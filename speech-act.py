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


def forward_propagation_bidirectional(input_data, sequence_lengths, E):
    # embedding matrix
    E = tf.convert_to_tensor(E, tf.float32)
    W_embedding = tf.get_variable("W_embedding", initializer=E)

    print("Input data shape: ", input_data.shape)
    data = tf.nn.embedding_lookup(W_embedding, input_data)
    print("After word embedding input shape: ", data.shape)

    if options.recur_type=='lstm':
        cell_fw = tf.contrib.rnn.LSTMCell(options.hidden_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=0.75)
        cell_bw = tf.contrib.rnn.LSTMCell(options.hidden_size)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=0.75)

        (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, data, sequence_length=sequence_lengths, dtype=tf.float32)

        (c_fw, h_fw) = state_fw
        (c_bw, h_bw) = state_bw
        print("h_fw: ", h_fw.shape)
        print("h_bw: ", h_bw.shape)

        h = tf.concat([h_fw, h_bw], axis=-1)
        print("h: ", h.shape)

    elif options.recur_type=='gru':
        cell_fw = tf.contrib.rnn.GRUCell(options.hidden_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=0.75)
        cell_bw = tf.contrib.rnn.GRUCell(options.hidden_size)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=0.75)

        (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, data, sequence_length=sequence_lengths, dtype=tf.float32)

        print("state_fw: ", state_fw.shape)
        print("state_bw: ", state_bw.shape)

        h = tf.concat([state_fw, state_bw], axis=-1)
        print("h: ", h.shape)

    weight = tf.get_variable("w", shape=[options.hidden_size + options.hidden_size, options.numClasses],
                             initializer=tf.contrib.layers.xavier_initializer(seed=101))
    bias = tf.get_variable("b", shape=[options.numClasses], initializer=tf.constant_initializer(0.0))

    prediction = (tf.matmul(h, weight) + bias)

    return prediction


def forward_propagation_unidirectional(input_data, sequence_lengths, E):
    # embedding matrix
    E = tf.convert_to_tensor(E, tf.float32)
    W_embedding = tf.get_variable("W_embedding", initializer=E)

    print("Input date shape: ", input_data.shape)
    data = tf.nn.embedding_lookup(W_embedding, input_data)
    print("After word embedding input shape: ", data.shape)

    lstmCell = tf.contrib.rnn.LSTMCell(options.hidden_size)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, state = tf.nn.dynamic_rnn(lstmCell, data, sequence_length=sequence_lengths, dtype=tf.float32)

    (c, h) = state
    print("c: ", c.shape)
    print("h: ", h.shape)

    weight = tf.get_variable("w", shape=[options.hidden_size, options.numClasses],
                             initializer=tf.contrib.layers.xavier_initializer(seed=101))
    bias = tf.get_variable("b", shape=[options.numClasses], initializer=tf.constant_initializer(0.0))

    prediction = (tf.matmul(h, weight) + bias)

    return prediction


def forward_propagation_mlp(input_data, sequence_lengths, E):
    # embedding matrix
    E = tf.convert_to_tensor(E, tf.float32)
    W_embedding = tf.get_variable("W_embedding", initializer=E)

    print("Input data shape: ", input_data.shape)
    data = tf.nn.embedding_lookup(W_embedding, input_data)
    print("After word embedding input shape: ", data.shape)

    c = tf.reshape(data, [-1, options.emb_size*options.maxlen])
    print("c: ", c.shape)

    w1 = tf.get_variable("w1", shape=[options.emb_size*options.maxlen, 1000],
                             initializer=tf.contrib.layers.xavier_initializer(seed=101))
    b1 = tf.get_variable("b1", shape=[1000], initializer=tf.constant_initializer(0.0))

    w2 = tf.get_variable("w2", shape=[1000, options.numClasses],
                         initializer=tf.contrib.layers.xavier_initializer(seed=101))
    b2 = tf.get_variable("b2", shape=[options.numClasses], initializer=tf.constant_initializer(0.0))

    A = tf.nn.relu(tf.matmul(c, w1) + b1)

    prediction = tf.matmul(A, w2) + b2

    return prediction


def forward_propagation_log_reg(input_data, sequence_lengths, E):
    # embedding matrix
    E = tf.convert_to_tensor(E, tf.float32)
    W_embedding = tf.get_variable("W_embedding", initializer=E)

    print("Input data shape: ", input_data.shape)
    data = tf.nn.embedding_lookup(W_embedding, input_data)
    print("After word embedding input shape: ", data.shape)

    c = tf.reshape(data, [-1, options.emb_size*options.maxlen])
    print("c: ", c.shape)

    weight = tf.get_variable("w", shape=[options.emb_size*options.maxlen, options.numClasses],
                             initializer=tf.contrib.layers.xavier_initializer(seed=101))
    bias = tf.get_variable("b", shape=[options.numClasses], initializer=tf.constant_initializer(0.0))

    prediction = (tf.matmul(c, weight) + bias)

    return prediction


def forward_propagation_averaging(input_data, sequence_lengths, E):
    # embedding matrix
    E = tf.convert_to_tensor(E, tf.float32)
    W_embedding = tf.get_variable("W_embedding", initializer=E)

    print("Input data shape: ", input_data.shape)
    data = tf.nn.embedding_lookup(W_embedding, input_data)
    print("After word embedding input shape: ", data.shape)

    c = tf.reduce_mean(data, axis=1)    #averaging the word embedding
    print("c: ", c.shape)

    weight = tf.get_variable("w", shape=[options.emb_size, options.numClasses],
                             initializer=tf.contrib.layers.xavier_initializer(seed=101))
    bias = tf.get_variable("b", shape=[options.numClasses], initializer=tf.constant_initializer(0.0))

    prediction = (tf.matmul(c, weight) + bias)

    return prediction


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
        mini_batch_seqlen = seq_len[k * mini_batch_size: k * mini_batch_size + mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_seqlen)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch_seqlen = seq_len[num_complete_minibatches * mini_batch_size: m]

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
        data_dir= "./data/input_to_DNNs/cat_MRDA/ta/"
        , data_spec="in"
        , model_dir="./saved_models/"
        , log_file="log"
        , learn_alg="adam"  # sgd, adagrad, rmsprop, adadelta, adam (default)
        , loss="softmax_crossentropy"  # hinge, squared_hinge, binary_crossentropy (default)
        , minibatch_size=32
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
        , map_class=1
        , numClasses=5
        , evalMinibatches=100
    )
    options, args = parser.parse_args(sys.argv)

    (X_train, y_train), (X_test, y_test), (X_dev, y_dev), max_features, E, label_id, sequence_len = \
        aidr.load_and_numberize_data(path=options.data_dir, nb_words=options.max_features, maxlen=options.maxlen,
                                     init_type=options.init_type,
                                     dev_train_merge=1, embfile=None, map_labels_to_five_class=options.map_class)

    # Placeholders
    input_data = tf.placeholder(tf.int32, [None, options.maxlen], name="input_data")
    sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
    y_values = tf.placeholder(tf.int32, [None])
    labels = tf.one_hot(y_values, options.numClasses)

    prediction = forward_propagation_bidirectional(input_data, sequence_lengths, E)
    #prediction = forward_propagation_unidirectional(input_data, sequence_lengths, E)
    #prediction = forward_propagation_averaging(input_data, sequence_lengths, E)
    #prediction = forward_propagation_mlp(input_data, sequence_lengths, E)
    #prediction = forward_propagation_log_reg(input_data, sequence_lengths, E)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    correctPred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    y_preds = tf.argmax(prediction, axis=1)

    init = tf.global_variables_initializer()
    m = X_train.shape[0]

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
            # randomly shuffle the training data
            np.random.seed(2018+epoch)
            np.random.shuffle(X_train)
            np.random.seed(2018+epoch)
            np.random.shuffle(y_train)
            np.random.seed(2018+epoch)
            np.random.shuffle(sequence_len['train_seq_len'])
            minibatch_cost = 0.
            num_minibatches = int(m / options.minibatch_size)
            train_minibatches = mini_batches(X_train, y_train, seq_len=sequence_len['train_seq_len'],
                                             mini_batch_size=options.minibatch_size)

            for (i, train_minibatch) in enumerate(train_minibatches):
                (train_minibatch_X, train_minibatch_y, train_minibatch_seqlen) = train_minibatch
                _, train_batch_loss, pr = sess.run([optimizer, loss, prediction], {input_data: train_minibatch_X,
                                                                                   y_values: train_minibatch_y,
                                                                                   sequence_lengths: train_minibatch_seqlen})

                if ((i + 1) % options.evalMinibatches == 0 or i == num_minibatches - 1):
                    test_acc, test_y_vals, test_y_preds = sess.run([accuracy, y_values, y_preds],
                                                                   {input_data: X_test,
                                                                    y_values: y_test,
                                                                    sequence_lengths: sequence_len['test_seq_len']})

                    acc_test = metrics.accuracy_score(test_y_vals, test_y_preds)
                    mic_p, mic_r, mic_f, sup = metrics.precision_recall_fscore_support(test_y_vals, test_y_preds,
                                                                                       average='micro')
                    mac_p, mac_r, mac_f, sup = metrics.precision_recall_fscore_support(test_y_vals, test_y_preds,
                                                                                       average='macro')
                    if (mac_f > best_macroF1):
                        best_accuracy = acc_test
                        best_macroF1 = mac_f
                        best_epoch = epoch
                        best_minibatch = i

                    print("\n\n##Epoch: ", epoch,"  Minibatch: ", i)
                    print("Test Accuracy: ", acc_test)
                    print("Macro F-score: ", mac_f)
                    print("**Best so far** Epoch: ", best_epoch, " Minibatch: ", best_minibatch,
                          " Best Test acc: ", best_accuracy, " Best F1: ", best_macroF1, " **\n")
