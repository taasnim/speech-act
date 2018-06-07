from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import math
import numpy as np
import tensorflow as tf
import optparse
import sys
import math
import glob, os, csv, re
from collections import Counter
from sklearn import metrics

from utilities import aidr
from flip_gradient import flip_gradient


class dannModel(object):
    """domain adaptation model."""

    def __init__(self, E):
        self._build_model(E)

    def _build_model(self, E):
        self.X = tf.placeholder(tf.int32, [None, options.maxlen], name="input_data")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.y_values = tf.placeholder(tf.int32, [None])
        self.y = tf.one_hot(self.y_values, options.numClasses)


        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.lambda_ = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        # RNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            # embedding matrix
            E = tf.convert_to_tensor(E, tf.float32)
            W_embedding = tf.get_variable("W_embedding", initializer=E)
            print("Embedding shape: ", W_embedding.shape)

            print("Input data shape: ", self.X.shape)
            data = tf.nn.embedding_lookup(W_embedding, self.X)
            print("After word embedding input shape: ", data.shape)

            cell_fw = tf.contrib.rnn.LSTMCell(options.hidden_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=0.75)
            cell_bw = tf.contrib.rnn.LSTMCell(options.hidden_size)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=0.75)

            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, data, sequence_length=self.sequence_lengths, dtype=tf.float32)

            (c_fw, h_fw) = state_fw
            (c_bw, h_bw) = state_bw
            print("c_fw: ", c_fw.shape)
            print("c_bw: ", c_bw.shape)

            # The domain-invariant feature
            self.feature = tf.concat([c_fw, c_bw], axis=-1)
            print("feature shape: ", self.feature.shape)

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route of half of target examples (last one fourth of batch) differently
            # depending on train or test mode.
            all_features = self.feature
            source_features = tf.slice(self.feature, [0, 0], [options.minibatch_size // 2, -1])
            target_labeled_features = tf.slice(self.feature, [options.minibatch_size // 2, 0], [options.minibatch_size // 4, -1])
            print("All features: ", all_features.shape)
            print("source features: ", source_features.shape)
            print("target labeled features: ", target_labeled_features.shape)
            if self.train==True:
                classify_feats = tf.concat([source_features, target_labeled_features], 0)
            else:
                classify_feats = all_features
            all_labels = self.y
            source_labels = tf.slice(self.y, [0, 0], [options.minibatch_size // 2, -1])
            target_labeled_labels = tf.slice(self.y, [options.minibatch_size // 2, 0], [options.minibatch_size // 4, -1])
            print("All labels: ", all_labels.shape)
            print("source labels: ", source_labels.shape)
            print("target labelled labels: ", target_labeled_labels.shape)
            if self.train==True:
                self.classify_labels = tf.concat([source_labels, target_labeled_labels], 0)
            else:
                self.classify_labels = all_labels
            weight = tf.get_variable("l_w1",
                                     shape=[options.hidden_size + options.hidden_size, options.numClasses],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=101))
            bias = tf.get_variable("l_b1", shape=[options.numClasses], initializer=tf.constant_initializer(0.0))

            self.logits = (tf.matmul(classify_feats, weight) + bias)
            print("label Predictor: ", self.logits.shape)
            self.pred = tf.nn.softmax(self.logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.classify_labels)
            print("pred loss: ", self.pred_loss.shape)

        # MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.lambda_)

            d_W_fc0 = tf.get_variable("d_w1", shape=[options.hidden_size + options.hidden_size, 100],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=101))
            d_b_fc0 = tf.get_variable("d_b1", shape=[100], initializer=tf.constant_initializer(0.0))
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = tf.get_variable("d_w2", shape=[100, 2],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=101))
            d_b_fc1 = tf.get_variable("d_b2", shape=[2], initializer=tf.constant_initializer(0.0))
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1
            print("domain predictor: ", d_logits.shape)

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)
            print("domain loss: ", self.domain_loss.shape)


def mini_batches(X, Y, seq_len, mini_batch_size=32):
    """
    Creates a list of minibatches from (X, Y)

    Arguments:
    X -- input data [2D shape (num_sentences X maxlen)]
    Y -- label [list containing values 0-4 for 5 classes]
    seq_len -- length of each element in X
    mini_batch_size -- Size of each mini batch

    Returns:
    list of mini batches from the positive and negative documents.

    """
    m = X.shape[0]
    mini_batches = []

    num_complete_minibatches = int(math.floor(m / mini_batch_size))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        # mini_batch_Y_one_hot = tf.one_hot(mini_batch_Y, numClasses)
        mini_batch_seqlen = seq_len[k * mini_batch_size: k * mini_batch_size + mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_seqlen)
        mini_batches.append(mini_batch)

    return mini_batches


if __name__ == '__main__':
    parser = optparse.OptionParser("%prog [options]")

    # file related options
    parser.add_option("-g", "--log-file", dest="log_file", help="log file [default: %default]")
    parser.add_option("-d", "--data-dir", dest="data_dir",
                      help="directory containing train, test and dev file [default: %default]")
    parser.add_option("-D", "--data-spec", dest="data_spec",
                      help="specification for training data (in, out, in_out) [default: %default]")
    parser.add_option("-p", "--model-dir", dest="model_dir",
                      help="directory to save the best models [default: %default]")

    # network related
    parser.add_option("-t", "--max-tweet-length", dest="maxlen", type="int",
                      help="maximal tweet length (for fixed size input) [default: %default]")  # input size
    parser.add_option("-m", "--model-type", dest="model_type",
                      help="uni or bidirectional [default: %default]")  # uni, bi-directional
    parser.add_option("-r", "--recurrent-type", dest="recur_type",
                      help="recurrent types (lstm, gru, simpleRNN) [default: %default]")  # lstm, gru, simpleRNN
    parser.add_option("-v", "--vocabulary-size", dest="max_features", type="int",
                      help="vocabulary size [default: %default]")  # emb matrix row size
    parser.add_option("-e", "--emb-size", dest="emb_size", type="int",
                      help="dimension of embedding [default: %default]")  # emb matrix col size
    parser.add_option("-s", "--hidden-size", dest="hidden_size", type="int",
                      help="hidden layer size [default: %default]")  # size of the hidden layer
    parser.add_option("-o", "--dropout_ratio", dest="dropout_ratio", type="float",
                      help="ratio of cells to drop out [default: %default]")
    parser.add_option("-i", "--init-type", dest="init_type", help="random or pretrained [default: %default]")
    parser.add_option("-f", "--emb-file", dest="emb_file", help="file containing the word vectors [default: %default]")
    parser.add_option("-P", "--tune-emb", dest="tune_emb", action="store_false",
                      help="DON't tune word embeddings [default: %default]")
    parser.add_option("-z", "--num-class", dest="numClasses", type="int",
                      help="Number of output classes [default: %default]")
    parser.add_option("-E", "--eval-minibatches", dest="evalMinibatches", type="int",
                      help="After how many minibatch do we want to evaluate. [default: %default]")

    # learning related
    parser.add_option("-a", "--learning-algorithm", dest="learn_alg",
                      help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
    parser.add_option("-b", "--minibatch-size", dest="minibatch_size", type="int",
                      help="minibatch size [default: %default]")
    parser.add_option("-l", "--loss", dest="loss",
                      help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
    parser.add_option("-n", "--epochs", dest="epochs", type="int", help="nb of epochs [default: %default]")
    parser.add_option("-C", "--map-class", dest="map_class", type="int",
                      help="map classes to five labels [default: %default]")

    parser.set_defaults(
        data_dir="./data/input_to_DNNs/main_files/ta/"
        , data_spec="in"

        , model_dir="./saved_models/"
        , log_file="log"
        , learn_alg="momentum"  # momentum, sgd, adagrad, rmsprop, adadelta, adam (default)
        , loss="softmax_crossentropy"  # hinge, squared_hinge, binary_crossentropy (default)
        , minibatch_size=64
        , dropout_ratio=0.75
        , maxlen=100
        , epochs=10
        , max_features=10000
        , emb_size=300
        , hidden_size=128
        , model_type='bidirectional'  # bidirectional, unidirectional (default)
        , recur_type='lstm'  # gru, simplernn, lstm (default)
        , init_type='conv_glove'  # 'random', 'word2vec', 'glove', 'conv_word2vec', 'conv_glove', 'meta_conv',  'meta_orig'
        , emb_file="../data/unlabeled_corpus.vec"
        , tune_emb=True
        , map_class=0
        , numClasses=5
        , evalMinibatches=100
    )

    options, args = parser.parse_args(sys.argv)
    print("Using ", options.learn_alg, "optimizer.")

    (X_src, y_src), (X_train, y_train), (X_test, y_test), (X_dev, y_dev), max_features, E, label_id, sequence_len = \
        aidr.load_and_numberize_data_dann(path=options.data_dir, nb_words=options.max_features, maxlen=options.maxlen,
                                     init_type=options.init_type,
                                     dev_train_merge=1, embfile=None, map_labels_to_five_class=1)

    model = dannModel(E)

    learning_rate = tf.placeholder(tf.float32, [])

    prediction = model.logits

    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    if options.learn_alg == "adam":
        optimizer_regular = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(pred_loss)
        optimizer_dann = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
    elif options.learn_alg == "momentum":
        optimizer_regular = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        optimizer_dann = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    correctPred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(model.classify_labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    y_preds = tf.argmax(prediction, axis=1)

    init = tf.global_variables_initializer()
    m = X_src.shape[0]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        saver = tf.train.Saver()
        sess.run(init)

        best_accuracy = 0.
        best_macroF1 = 0.
        best_epoch = -1
        best_minibatch = -1

        for epoch in range(options.epochs):
            # randomly shuffle the source data
            np.random.seed(2018)
            np.random.shuffle(X_src)
            np.random.seed(2018)
            np.random.shuffle(y_src)
            np.random.seed(2018)
            np.random.shuffle(sequence_len['src_seq_len'])
            src_minibatches = mini_batches(X_src, y_src, seq_len=sequence_len['src_seq_len'],
                                           mini_batch_size=options.minibatch_size // 2)
            src_num_minibatches = len(src_minibatches)

            # randomly shuffle the target training data
            np.random.seed(2018)
            np.random.shuffle(X_train)
            np.random.seed(2018)
            np.random.shuffle(y_train)
            np.random.seed(2018)
            np.random.shuffle(sequence_len['train_seq_len'])
            target_train_minibatches = mini_batches(X_train, y_train, seq_len=sequence_len['train_seq_len'],
                                                    mini_batch_size=options.minibatch_size // 2)
            target_num_minibatches = len(target_train_minibatches)

            domain_labels = np.vstack([np.tile([1., 0.], [options.minibatch_size // 2, 1]),
                                       np.tile([0., 1.], [options.minibatch_size // 2, 1])])

            for (i, src_minibatch) in enumerate(src_minibatches):
                # Adaptation param and learning rate schedule as described in the DANN paper
                num_steps = (src_num_minibatches-1)*options.epochs
                p = float(epoch*(src_num_minibatches-1) + i) / num_steps
                lambda_ = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p) ** 0.75

                (src_minibatch_X, src_minibatch_y, src_minibatch_seqlen) = src_minibatch
                (target_train_minibatch_X, target_train_minibatch_y, target_train_minibatch_seqlen) = \
                target_train_minibatches[i%target_num_minibatches]

                X = np.vstack([src_minibatch_X, target_train_minibatch_X])
                Y = np.squeeze(np.vstack([np.reshape(np.asarray(src_minibatch_y), (-1, 1)),
                                          np.reshape(np.asarray(target_train_minibatch_y), (-1, 1))]))
                Z = np.squeeze(np.vstack([np.reshape(np.asarray(src_minibatch_seqlen), (-1, 1)),
                                          np.reshape(np.asarray(target_train_minibatch_seqlen), (-1, 1))]))
                '''
                _, batch_loss = sess.run([optimizer_regular, pred_loss],
                    feed_dict={model.X: X, model.y_values: Y, model.sequence_lengths: Z, model.domain: domain_labels,
                               model.train: True, model.lambda_: lambda_, learning_rate: lr})

                '''
                _, batch_loss = sess.run([optimizer_dann, total_loss],
                                         feed_dict={model.X: X, model.y_values: Y, model.sequence_lengths: Z,
                                                    model.domain: domain_labels,
                                                    model.train: True, model.lambda_: lambda_, learning_rate: lr})


                if ((i + 1) % options.evalMinibatches == 0 or i == len(src_minibatches) - 1):

                    test_acc, test_y_vals, test_y_preds = sess.run([accuracy, model.y_values, y_preds],
                    feed_dict={model.X: X_test, model.y_values: y_test,
                               model.sequence_lengths: sequence_len['test_seq_len'], model.train: False})

                    acc_test = metrics.accuracy_score(test_y_vals, test_y_preds)
                    # print("Test Accuracy: ", test_acc)

                    mic_p, mic_r, mic_f, sup = metrics.precision_recall_fscore_support(test_y_vals, test_y_preds,
                                                                                       average='micro')
                    mac_p, mac_r, mac_f, sup = metrics.precision_recall_fscore_support(test_y_vals, test_y_preds,
                                                                                       average='macro')

                    if (mac_f > best_macroF1):
                        best_accuracy = acc_test
                        best_macroF1 = mac_f
                        best_epoch = epoch
                        best_minibatch = i

                    print("\n\n##Epoch: ", epoch, "  Minibatch: ", i)

                    # print("Dev Accuracy: ", acc_dev)
                    print("Test Accuracy: ", acc_test)
                    print("Macro F-score: ", mac_f)
                    print("**Best so far** Epoch: ", best_epoch, " Minibatch: ", best_minibatch,
                          " Best Test acc: ", best_accuracy, " Best F1: ", best_macroF1, " **\n")
