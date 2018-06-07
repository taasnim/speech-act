from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np
import string

import glob, os, csv, re
from collections import Counter
import codecs


def load_and_numberize_data(path="../data/", nb_words=None, maxlen=None, seed=2018,
                            start_char=1, oov_char=2, index_from=3, init_type="random",
                            embfile=None, dev_train_merge=0, map_labels_to_five_class=0,
                            out_model=None, out_vocab_file=None):

    """ numberize the train, dev and test files """

    # read the vocab from the entire corpus (train + test + dev)
    vocab = Counter()

    sentences_train = []
    y_train = []

    sentences_test  = []
    y_test = []

    sentences_dev   = []
    y_dev  = []

    for filename in glob.glob(os.path.join(path, '*.csv')):

        if re.search("(compress|emb)", filename):
            continue

        print("Reading vocabulary from" + filename)

        reader  = csv.reader(open(filename, 'r'))

        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue
            if re.search("train", filename.lower()):    
                sentences_train.append(row[1])
                y_train.append(row[2])    

            elif re.search("test", filename.lower()):    
                sentences_test.append(row[1])    
                y_test.append(row[2])    

            elif re.search("dev", filename.lower()):    
                sentences_dev.append(row[1])    
                y_dev.append(row[2])    

            for wrd in row[1].split():
                vocab[wrd] += 1


    full_vocab_size = len(vocab)
    full_vocab      = dict(vocab)

    print("Nb of sentences: train: " + str (len(sentences_train)) + " test: " + str (len(sentences_test)) + " dev: " + str (len(sentences_dev)))
    print("Total vocabulary size: " + str (full_vocab_size))

    if nb_words is None: # now take a fraction
        pr_perc = 100
        nb_words = len(vocab) 
    else:
        pr_perc  = nb_words
        nb_words = int(len(vocab) * (nb_words / 100.0))

#    if nb_words is None or nb_words > len(vocab):
#        nb_words = len(vocab) 

    vocab = dict(vocab.most_common(nb_words))
    print("Pruned vocabulary size: " + str(pr_perc) + "% =" + str(len(vocab)))

    #Create vocab dictionary that maps word to ID
#    vocab_list = vocab.keys()
    vocab_list = sorted(vocab.keys())
    vocab_list = ['00Padding00', '##START', 'OOV'] + vocab_list

    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

#    cPickle.dump(vocab_idmap, open("test.vocab", "wb"))    

    # Numberize the sentences
    X_train = numberize_sentences(sentences_train, vocab_idmap, oov_char=oov_char)
    X_test  = numberize_sentences(sentences_test,  vocab_idmap, oov_char=oov_char)
    X_dev   = numberize_sentences(sentences_dev,   vocab_idmap, oov_char=oov_char)


    #Create label dictionary that map label to ID
    merge_labels = None

    if map_labels_to_five_class:                
        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    "ques":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE 
                    "AA":"Polite",\
                    "P":"Polite",\
                    "appr":"Polite",\
                    # STATEMENT 
                    "S":"St",\
                    "st":"St",\
                    # RESPONSE 
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    "res":"Res",\
                    #SUGGESTION
                    "sug":"Sug",\
                    "AC":"Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        y_test  = remap_labels(y_test,  merge_labels=merge_labels)
        y_dev   = remap_labels(y_dev,   merge_labels=merge_labels)

#    label_list = list (set(y_train))
    label_list = sorted(list(set(y_train)))
    print("labels:", label_list)
    label_map = {}
    for lab_id, lab in enumerate(label_list):
        label_map[lab] = lab_id  

    # Numberize the labels
    (y_train, y_train_freq)   = numberize_labels(y_train, label_map)
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    (y_dev,   y_dev_freq)     = numberize_labels(y_dev,   label_map)


    assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    #randomly shuffle the training data
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    '''
    X_train, y_train = adjust_index(X_train, y_train, start_char=start_char, index_from=index_from, maxlen=maxlen)
    X_test,  y_test  = adjust_index(X_test,  y_test,  start_char=start_char, index_from=index_from, maxlen=maxlen)
    X_dev,   y_dev   = adjust_index(X_dev,   y_dev,   start_char=start_char, index_from=index_from, maxlen=maxlen)
    '''

    if dev_train_merge:
        X_train.extend(X_dev)
        y_train.extend(y_dev)
        #y_train=np.concatenate ((y_train, y_dev)) # need if y_train is numpy array

    train_seq_len = [len(X_train[i]) if len(X_train[i])<maxlen else maxlen for i in range(len(X_train))]
    test_seq_len = [len(X_test[i]) if len(X_test[i])<maxlen else maxlen for i in range(len(X_test))]
    dev_seq_len = [len(X_dev[i]) if len(X_dev[i])<maxlen else maxlen for i in range(len(X_dev))]
    sequence_len = {'train_seq_len': train_seq_len,
                    'test_seq_len': test_seq_len,
                    'dev_seq_len': dev_seq_len}

    X_train = pad_sequences(X_train, maxlen)
    X_test = pad_sequences(X_test, maxlen)
    X_dev = pad_sequences(X_dev, maxlen)


    # load the embeddeings
    if init_type.lower() == "word2vec":
        print("Loading word2vec!!")
        #E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding(vocabs=vocab_list,
                                           filename="../pretrained_embeddings/word2vec_googlenews/GoogleNews-vectors-negative300.bin.txt",
                                           pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "glove":
        print("Loading glove!!")
        #E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding(vocabs=vocab_list,
                                           filename="../pretrained_embeddings/Glove6B/glove.6B.300d.txt",
                                           pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "conv_word2vec":
        print("Loading conv_word2vec embedding!!")
        #E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding(vocabs=vocab_list,
                                           filename="../pretrained_embeddings/word2vec_source_code/word2vec_10_04_18.txt",
                                           pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "conv_glove":
        print("Loading conv_glove embedding!!")
        #E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding(vocabs=vocab_list,
                                      filename="../pretrained_embeddings/glove_source_code/glove_conv_12_04_18.txt",
                                       #filename="./utilities/glove_conv_12_04_18.txt",
                                      pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "meta_conv":
        print("Loading meta_conv embedding!!")
        #E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding_meta_conv(vocabs=vocab_list,
                                           word2vec_filename="../pretrained_embeddings/word2vec_source_code/word2vec_10_04_18.txt",
                                           glove_filename="../pretrained_embeddings/glove_source_code/glove_conv_12_04_18.txt",
                                           pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "meta_orig":
        print("Loading meta_orig embedding!!")
        #E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding_meta_orig(vocabs=vocab_list,
                                           word2vec_filename="../pretrained_embeddings/word2vec_googlenews/GoogleNews-vectors-negative300.bin.txt",
                                           glove_filename="../pretrained_embeddings/Glove6B/glove.6B.300d.txt",
                                           pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    #elif init_type.lower() != "random" and out_model:
        #E = get_embeddings_from_weight_file(out_model, vocab_idmap, out_vocab_file, index_from)
    else:
        np.random.seed(seed)
        emb_size=300
        E = 0.01 * np.random.uniform(-1.0, 1.0, (len(vocab_list), emb_size))
        print("Shape of random embedding: ", E.shape)


    return (X_train, y_train), (X_test, y_test), (X_dev, y_dev), nb_words + index_from, E, label_map, sequence_len


def load_and_numberize_data_dann(path="../data/", nb_words=None, maxlen=None, seed=2018,
                                 start_char=1, oov_char=2, index_from=3, init_type="random",
                                 embfile=None, dev_train_merge=0, map_labels_to_five_class=0,
                                 out_model=None, out_vocab_file=None):
    """ numberize the train, dev and test files """

    # read the vocab from the entire corpus (train + test + dev)
    vocab = Counter()

    sentences_train = []
    y_train = []

    sentences_test = []
    y_test = []

    sentences_dev = []
    y_dev = []

    for filename in glob.glob(os.path.join(path, '*.csv')):

        if re.search("(compress|emb)", filename):
            continue

        print("Reading vocabulary from" + filename)

        reader = csv.reader(open(filename, 'r'))

        for rowid, row in enumerate(reader):
            if rowid == 0:  # header
                continue
            if re.search("train", filename.lower()):
                sentences_train.append(row[1])
                y_train.append(row[2])

            elif re.search("test", filename.lower()):
                sentences_test.append(row[1])
                y_test.append(row[2])

            elif re.search("dev", filename.lower()):
                sentences_dev.append(row[1])
                y_dev.append(row[2])

            for wrd in row[1].split():
                vocab[wrd] += 1

    lines = [line.rstrip('\n') for line in open('./data/MRDA/Train.txt')]
    sentences_src = []
    y_src = []

    for i in range(len(lines)):
        row = re.split(r'\t+', lines[i])
        sentences_src.append(row[1])
        y_src.append(row[0])
        for wrd in row[1].split():
            vocab[wrd] += 1

    full_vocab_size = len(vocab)
    full_vocab = dict(vocab)

    print("Source Data:")
    print("Nb of sentences: train: " + str(len(sentences_src)))
    print("Target Data:")
    print(
        "Nb of sentences: train: " + str(len(sentences_train)) + " test: " + str(len(sentences_test)) + " dev: " + str(
            len(sentences_dev)))
    print("Total vocabulary size: " + str(full_vocab_size))

    if nb_words is None:  # now take a fraction
        pr_perc = 100
        nb_words = len(vocab)
    else:
        pr_perc = nb_words
        nb_words = int(len(vocab) * (nb_words / 100.0))

    #    if nb_words is None or nb_words > len(vocab):
    #        nb_words = len(vocab)

    vocab = dict(vocab.most_common(nb_words))
    print("Pruned vocabulary size: " + str(pr_perc) + "% =" + str(len(vocab)))

    # Create vocab dictionary that maps word to ID
    #    vocab_list = vocab.keys()
    vocab_list = sorted(vocab.keys())
    vocab_list = ['00Padding00', '##START', 'OOV'] + vocab_list

    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    #    cPickle.dump(vocab_idmap, open("test.vocab", "wb"))

    # Numberize the sentences
    X_train = numberize_sentences(sentences_train, vocab_idmap, oov_char=oov_char)
    X_test = numberize_sentences(sentences_test, vocab_idmap, oov_char=oov_char)
    X_dev = numberize_sentences(sentences_dev, vocab_idmap, oov_char=oov_char)

    X_src = numberize_sentences(sentences_src, vocab_idmap, oov_char=oov_char)

    # Create label dictionary that map label to ID
    merge_labels = None

    if map_labels_to_five_class:
        merge_labels = {
            # QUESTION/REQUEST
            "QH": "Ques", \
            "QO": "Ques", \
            "QR": "Ques", \
            "QW": "Ques", \
            "QY": "Ques", \
            "ques": "Ques", \
            # APPRECIATION/ASSESSMENT/POLITE
            "AA": "Polite", \
            "P": "Polite", \
            "appr": "Polite", \
            # STATEMENT
            "S": "St", \
            "st": "St", \
            # RESPONSE
            "A": "Res", \
            "R": "Res", \
            "U": "Res", \
            "res": "Res", \
            # SUGGESTION
            "sug": "Sug", \
            "AC": "Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        y_test = remap_labels(y_test, merge_labels=merge_labels)
        y_dev = remap_labels(y_dev, merge_labels=merge_labels)

        y_src = remap_labels(y_src, merge_labels=merge_labels)

    #    label_list = list (set(y_train))
    label_list = sorted(list(set(y_train)))
    #    print "labels:", label_list
    label_map = {}
    for lab_id, lab in enumerate(label_list):
        label_map[lab] = lab_id

        # Numberize the labels
    (y_train, y_train_freq) = numberize_labels(y_train, label_map)
    (y_test, y_test_freq) = numberize_labels(y_test, label_map)
    (y_dev, y_dev_freq) = numberize_labels(y_dev, label_map)

    (y_src, y_src_freq) = numberize_labels(y_src, label_map)

    assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    # randomly shuffle the training data
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    '''
    X_train, y_train = adjust_index(X_train, y_train, start_char=start_char, index_from=index_from, maxlen=maxlen)
    X_test,  y_test  = adjust_index(X_test,  y_test,  start_char=start_char, index_from=index_from, maxlen=maxlen)
    X_dev,   y_dev   = adjust_index(X_dev,   y_dev,   start_char=start_char, index_from=index_from, maxlen=maxlen)
    '''

    if dev_train_merge:
        X_train.extend(X_dev)
        y_train.extend(y_dev)
        # y_train=np.concatenate ((y_train, y_dev)) # need if y_train is numpy array

    train_seq_len = [len(X_train[i]) if len(X_train[i]) < maxlen else maxlen for i in range(len(X_train))]
    test_seq_len = [len(X_test[i]) if len(X_test[i]) < maxlen else maxlen for i in range(len(X_test))]
    dev_seq_len = [len(X_dev[i]) if len(X_dev[i]) < maxlen else maxlen for i in range(len(X_dev))]

    src_seq_len = [len(X_src[i]) if len(X_src[i]) < maxlen else maxlen for i in range(len(X_src))]
    sequence_len = {'train_seq_len': train_seq_len,
                    'test_seq_len': test_seq_len,
                    'dev_seq_len': dev_seq_len,
                    'src_seq_len': src_seq_len}

    X_train = pad_sequences(X_train, maxlen)
    X_test = pad_sequences(X_test, maxlen)
    X_dev = pad_sequences(X_dev, maxlen)

    X_src = pad_sequences(X_src, maxlen)

    # load the embeddeings
    if init_type.lower() == "word2vec":
        print("Loading word2vec!!")
        # E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding(vocabs=vocab_list,
                                      filename="../pretrained_embeddings/word2vec_googlenews/GoogleNews-vectors-negative300.bin.txt",
                                      pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "glove":
        print("Loading glove!!")
        # E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding(vocabs=vocab_list,
                                      filename="../pretrained_embeddings/Glove6B/glove.6B.300d.txt",
                                      pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "conv_word2vec":
        print("Loading conv_word2vec embedding!!")
        # E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding(vocabs=vocab_list,
                                      filename="../pretrained_embeddings/word2vec_source_code/word2vec_10_04_18.txt",
                                      pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "conv_glove":
        print("Loading conv_glove embedding!!")
        # E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding(vocabs=vocab_list,
                                      filename="../pretrained_embeddings/glove_source_code/glove_conv_12_04_18.txt",
                                      # filename="../pretrained_embedding_src/GloVe/glove_conv_12_04_18.txt",
                                      pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "meta_conv":
        print("Loading meta_conv embedding!!")
        # E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding_meta_conv(vocabs=vocab_list,
                                                word2vec_filename="../pretrained_embeddings/word2vec_source_code/word2vec_10_04_18.txt",
                                                glove_filename="../pretrained_embeddings/glove_source_code/glove_conv_12_04_18.txt",
                                                pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    elif init_type.lower() == "meta_orig":
        print("Loading meta_orig embedding!!")
        # E = load_emb(embfile, vocab_idmap, index_from)
        # "./glove/glove.6B.300d.txt"
        E = load_pretrained_embedding_meta_orig(vocabs=vocab_list,
                                                word2vec_filename="../pretrained_embeddings/word2vec_googlenews/GoogleNews-vectors-negative300.bin.txt",
                                                glove_filename="../pretrained_embeddings/Glove6B/glove.6B.300d.txt",
                                                pretrained_type=init_type)
        print("Shape of pretrained embedding: ", E.shape)

    # elif init_type.lower() != "random" and out_model:
    # E = get_embeddings_from_weight_file(out_model, vocab_idmap, out_vocab_file, index_from)
    else:
        np.random.seed(seed)
        emb_size = 300
        E = 0.01 * np.random.uniform(-1.0, 1.0, (len(vocab_list), emb_size))
        print("Shape of random embedding: ", E.shape)

    return (X_src, y_src), (X_train, y_train), (X_test, y_test), (X_dev, y_dev), nb_words + index_from, E, label_map, sequence_len


def load_data_unig(path="../data/", map_labels_to_five_class=0, add_feat=0, dev_train_merge=0, seed=113):

    """ get the training, test and dev data """

    U_train, U_test, U_dev = [], [], []
    y_train, y_test, y_dev = [], [], []

    for filename in glob.glob(os.path.join(path, '*unigram.compress.csv')):

        print("Reading data from" + filename)
        reader  = csv.reader(open(filename, 'r'))

        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue

            if re.search("train", filename.lower()):
                get_feat_unig(row[3], U_train)
                y_train.append(row[2])

            elif re.search("test", filename.lower()):
                get_feat_unig(row[3], U_test)
                y_test.append(row[2])

            elif re.search("dev", filename.lower()):
                get_feat_unig(row[3], U_dev)
                y_dev.append(row[2])

    print("Nb of sentences: train: " + str (len(U_train)) + " test: " + str (len(U_test)) + " dev: " + str (len(U_dev)))

    #binarize sunigram features
    allUnig = [item for sublist in (U_train + U_test + U_dev) for item in sublist]
    nb_items = max(allUnig)+1

#    print nb_items

    X_train = one_of_k_unig(U_train, nb_items)
    X_dev   = one_of_k_unig(U_dev, nb_items)
    X_test  = one_of_k_unig(U_test, nb_items)

    #-----------------------------------------------------------------------------

    #Create label dictionary that map label to ID
    merge_labels = None

    if map_labels_to_five_class:
        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    "ques":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE
                    "AA":"Polite",\
                    "P":"Polite",\
                    "appr":"Polite",\
                    # STATEMENT
                    "S":"St",\
                    "st":"St",\
                    # RESPONSE
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    "res":"Res",\
                    #SUGGESTION
                    "sug":"Sug",\
                    "AC":"Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        y_test  = remap_labels(y_test,  merge_labels=merge_labels)
        y_dev   = remap_labels(y_dev,   merge_labels=merge_labels)


    label_list = sorted ( list (set(y_train)))
    label_map  = {}
    for lab_id, lab in enumerate (label_list):
        label_map[lab] = lab_id

#    print label_map

    # Numberize the labels
    (y_train, y_train_freq)   = numberize_labels(y_train, label_map)
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    (y_dev,   y_dev_freq)     = numberize_labels(y_dev,   label_map)

    assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    #randomly shuffle the training data
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    if dev_train_merge==1:
        print("here")
        X_train = np.concatenate((X_train, X_dev), axis=0)
        y_train = np.concatenate((y_train, y_dev), axis=0)

    return (X_train, y_train), (X_test, y_test), (X_dev, y_dev), label_map



def get_embeddings_from_weight_file(weight_file_path, vocab_idmap, out_vocab_file, index_from=3):

    """
    Extract word embeddings from the model file
    Args:
      weight_file_path (str) : Path to the file to analyze
    """

    print("Loading pre-trained embeddings from out-domain model ......")

    import h5py

    #read out-model vocab
    out_vocab_idmap = cPickle.load(open(out_vocab_file, "rb"))    

    # get model embeddings
    f = h5py.File(weight_file_path)
    try:
        if len(f.items())==0:
            print("Invalid model file..")
            exit(1)
        E_out   = f.items()[0][1]['param_0'][:]

    finally:
        f.close()

    print(" Shape of out-domain emb matrix: " + str (E_out.shape))

    # init Embedding matrix
    row_nb   = index_from+len(vocab_idmap)    
    vec_size = E_out.shape[1]
    E        = 0.01 * np.random.uniform( -1.0, 1.0, (row_nb, vec_size) )

    # map embeddings..
    E[0:index_from] = E_out[0:index_from]

    wrd_found = {}
    for awrd in vocab_idmap.keys():
        wid = vocab_idmap[awrd] + index_from # wid in the in-dom vocab
        if out_vocab_idmap.has_key(awrd):
            wrd_found[awrd] = 1 
            wid_out = out_vocab_idmap[awrd] + index_from # wid in the out-dom emb
            E[wid]  = E_out[wid_out]

    print(" Number of words found in emb matrix: " + str (len (wrd_found)) + " of " + str (len(vocab_idmap)))
    print(" Shape of emb matrix: " + str (E.shape))

    return E    


def extract_and_write_vocab(datadir, nb_words, vocab_file, index_from=3, get_labels=0):

    """ numberize the train, dev and test files """

    # read the vocab from the entire corpus (train + test + dev)
    vocab = Counter()

    sentences_train = []
    y_train = []

    sentences_test  = []
    y_test = []

    sentences_dev   = []
    y_dev  = []

    for filename in glob.glob(os.path.join(datadir, '*.csv')):
        reader  = csv.reader(open(filename, 'rb'))

        if re.search("compress", filename.lower()):
            continue

        print("Reading vocabulary from" + filename)

        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue

            if re.search("train", filename.lower()):    
                sentences_train.append(row[1])
                y_train.append(row[2])    

            elif re.search("test", filename.lower()):    
                sentences_test.append(row[1])    
                y_test.append(row[2])    

            elif re.search("dev", filename.lower()):    
                sentences_dev.append(row[1])    
                y_dev.append(row[2])    

            for wrd in row[1].split():
                vocab[wrd] += 1

    print("Nb of sentences: train: " + str (len(sentences_train)) + " test: " + str (len(sentences_test)) + " dev: " + str (len(sentences_dev)))
    print("Total vocabulary size: " + str (len(vocab)))

    if nb_words is None: # now take a fraction
        nb_words = len(vocab) 
    else:
        pr_perc  = nb_words
        nb_words = int ( len(vocab) * (nb_words / 100.0) )    

    vocab = dict (vocab.most_common(nb_words))
    print("Pruned vocabulary size: " + str (pr_perc) + "% =" + str (len(vocab)))

    #Create vocab dictionary that maps word to ID
    print("Saving vocab to " + vocab_file)

    vocab_list = sorted ( vocab.keys() )
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    cPickle.dump(vocab_idmap, open(vocab_file, "wb"))    

    if get_labels:
        #Create label dictionary that map label to ID
        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    "ques":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE 
                    "AA":"Polite",\
                    "P":"Polite",\
                    "appr":"Polite",\
                    # STATEMENT 
                    "S":"St",\
                    "st":"St",\
                    # RESPONSE 
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    "res":"Res",\
                    #SUGGESTION
                    "sug":"Sug",\
                    "AC":"Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        label_list = sorted ( list (set(y_train)) )

        label_map  = {}
        for lab_id, lab in enumerate (label_list):
            label_map[lab] = lab_id

        return nb_words + index_from, label_map      






def load_and_numberize_testfile(datadir, nb_words, test_file, maxlen=None, start_char=1, oov_char=2, index_from=3, map_labels_to_five_class=1):

    """ numberize test file based on the source domain vocab """

    # read the vocab from the entire corpus (train + test + dev)
    print("Creating vocab for the source domain...")

    vocab = Counter()

    sentences_train = []
    y_train = []

    sentences_test  = []
    y_test = []

    sentences_dev   = []
    y_dev  = []

    for filename in glob.glob(os.path.join(datadir, '*.csv')):
#        print "Reading vocabulary from" + filename
        reader  = csv.reader(open(filename, 'rb'))

        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue
            if re.search("train", filename.lower()):    
                sentences_train.append(row[1])
                y_train.append(row[2])    

            elif re.search("test", filename.lower()):    
                sentences_test.append(row[1])    
                y_test.append(row[2])    

            elif re.search("dev", filename.lower()):    
                sentences_dev.append(row[1])    
                y_dev.append(row[2])    

            for wrd in row[1].split():
                vocab[wrd] += 1

    full_vocab_size = len(vocab)
    full_vocab      = dict (vocab) 

    print("Nb of sentences (source data): train: " + str (len(sentences_train)) + " test: " + str (len(sentences_test)) + " dev: " + str (len(sentences_dev)))
    print("Total vocabulary size: " + str (full_vocab_size))

    if nb_words is None: # now take a fraction
        nb_words = len(vocab) 
    else:
        pr_perc  = nb_words
        nb_words = int ( len(vocab) * (nb_words / 100.0) )    

    vocab = dict (vocab.most_common(nb_words))
    print("Pruned vocabulary size: " + str (pr_perc) + "% =" + str (len(vocab)))

    #Create vocab dictionary that maps words to IDs
    print("\n".join(vocab.keys()))
    exit(0)    

    vocab_list = vocab.keys()
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i


    # Numberize the test sentences
    print("Reading sentences from " + test_file)
    reader  = csv.reader(open(test_file, 'rb'))
    sentences_test  = []
    y_test = []
    tar_vocab = Counter()

    for rowid, row in enumerate (reader):
        if rowid == 0: #header
            continue
        sentences_test.append(row[1])    
        y_test.append(row[2])    

        for wrd in row[1].split():
            tar_vocab[wrd] += 1

    print("Numberizing the sentences...")
    X_test, OOV_nb = numberize_sentences_and_report_OOV(sentences_test,  vocab_idmap, oov_char=oov_char)
    print("Total Vocab (test file): " + str(len(tar_vocab)) + "\tTotal OOV: " + str(OOV_nb))

    #Create label dictionary that maps labels to IDs
    merge_labels = None

    if map_labels_to_five_class:                
        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    "ques":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE 
                    "AA":"Polite",\
                    "P":"Polite",\
                    "appr":"Polite",\
                    # STATEMENT 
                    "S":"St",\
                    "st":"St",\
                    # RESPONSE 
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    "res":"Res",\
                    #SUGGESTION
                    "sug":"Sug",\
                    "AC":"Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        y_test  = remap_labels(y_test,  merge_labels=merge_labels)

    label_list = list (set(y_train))

    if label_list != list (set(y_test)):
        print("Error!!! " + "trained model and test file have different label sets")
        print(label_list)
        print(list (set(y_test)))
        #raw_input(' ')

    label_map  = {}
    for lab_id, lab in enumerate (label_list):
        label_map[lab] = lab_id  

    # Numberize the labels
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    X_test,  y_test  = adjust_index(X_test,  y_test,  start_char=start_char, index_from=index_from, maxlen=maxlen)

    return (X_test, y_test), nb_words + index_from, label_map


def load_and_numberize_testfile_using_vocab(vocab_idmap, test_file, maxlen=None, start_char=1, oov_char=2, index_from=3, map_labels_to_five_class=1, get_item_info=0):

    """ numberize test file based on the source domain vocab """

    # Numberize the test sentences
    print("Reading sentences from " + test_file)
    reader  = csv.reader(open(test_file, 'rb'))
    sentences  = []
    y_test = []
    sentences_id = []
    tar_vocab = Counter()

    for rowid, row in enumerate (reader):
        if rowid == 0: #header
            continue

        sentences_id.append(row[0])    
        sentences.append(row[1])    
        y_test.append(row[2])    

        for wrd in row[1].split():
            tar_vocab[wrd] += 1

    print("Numberizing the sentences...")
    X_test, OOV_nb = numberize_sentences_and_report_OOV(sentences,  vocab_idmap, oov_char=oov_char)
    print("Total Vocab (test file): " + str(len(tar_vocab)) + "\tTotal OOV: " + str(OOV_nb))

    #Create label dictionary that maps labels to IDs
    merge_labels = None

    if map_labels_to_five_class:                
        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    "ques":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE 
                    "AA":"Polite",\
                    "P":"Polite",\
                    "appr":"Polite",\
                    # STATEMENT 
                    "S":"St",\
                    "st":"St",\
                    # RESPONSE 
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    "res":"Res",\
                    #SUGGESTION
                    "sug":"Sug",\
                    "AC":"Sug"}

        y_test  = remap_labels(y_test,  merge_labels=merge_labels)
        labels  = y_test

    label_list = sorted (list (set(y_test)))
#    label_list = list (set(y_test))

    label_map  = {}
    for lab_id, lab in enumerate (label_list):
        label_map[lab] = lab_id  

    # Numberize the labels
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    X_test,  y_test  = adjust_index(X_test,  y_test,  start_char=start_char, index_from=index_from, maxlen=maxlen)

    if get_item_info:
        return (X_test, y_test), len(vocab_idmap) + index_from, label_map, sentences_id, sentences, labels

    return (X_test, y_test), len(vocab_idmap) + index_from, label_map

def load_and_numberize_testfile_using_vocab_for_salvatore(vocab_idmap, test_file, maxlen=None, start_char=1, oov_char=2, index_from=3, map_labels_to_five_class=1, get_item_info=0):

    """ numberize test file based on the source domain vocab """

    # Numberize the test sentences
    print("Reading sentences from " + test_file)
    reader  = csv.reader(open(test_file, 'rb'), delimiter='\t')
    sentences  = []
    y_test = []
    tar_vocab = Counter()

    for rowid, row in enumerate (reader):

        if len(row) == 3:
            sentences.append(row[2])
            for wrd in row[2].split():
                tar_vocab[wrd] += 1

        else:
            print(row)
            print(" empty sentence... ")
#            raw_input(' ')        

        y_test.append(row[1])    


    print("Numberizing the sentences...")
    X_test, OOV_nb = numberize_sentences_and_report_OOV(sentences,  vocab_idmap, oov_char=oov_char)
    print("Total Vocab (test file): " + str(len(tar_vocab)) + "\tTotal OOV: " + str(OOV_nb))

    X_test,  y_test  = adjust_index(X_test,  y_test,  start_char=start_char, index_from=index_from)
    return X_test


def write_predicted_labels(test_file, y_pred, class_labels, pred_file):
    """ write predictions to a file """

    reader  = csv.reader(open(test_file, 'rb'))
    FW      = open(pred_file, 'wb')
    sentences    = []
#    sentences_id = []

    for rowid, row in enumerate (reader):
        if rowid == 0: #header
            continue
#        sentences_id.append(row[0])    
        sentences.append(row[1])
        FW.write(row[0] + "\t" + row[1] + "\t" + class_labels[y_pred[rowid-1]] + "\n")    

    assert  len(sentences) == y_pred.shape[0]
    FW.close()


def load_emb(embfile, vocab_idmap, index_from=3, start_char=1, oov_char=2, padding_char=0, vec_size=300):
    """ load the word embeddings """

    print("Loading pre-trained word2vec embeddings......")

    if embfile.endswith(".gz"):
        f = gzip.open(embfile, 'rb')
    else:
        f = open(embfile, 'rb')

    vec_size_got = int ( f.readline().strip().split()[1]) # read the header to get vec dim

    if vec_size_got != vec_size:
        print(" vector size provided and found in the file don't match!!!")
        #raw_input(' ')
        exit(1)

    # load Embedding matrix
    row_nb = index_from+len(vocab_idmap)    
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (row_nb, vec_size) )

    wrd_found = {}

    for line in f: # read from the emb file
        all_ent   = line.split()

        word = all_ent[0].lower()

        if vocab_idmap.has_key(word):
            wrd_found[word] = 1
            vec = map (float, all_ent[1:])
            wid = vocab_idmap[word] + index_from
            E[wid] = np.array(vec)

    f.close()
    print(" Number of words found in emb matrix: " + str (len (wrd_found)) + " of " + str (len(vocab_idmap)))

    return E        




def remap_labels(y, merge_labels=None):

    if not merge_labels:
        return y

    y_modified = []
    for alabel in y:
        if alabel in merge_labels:
            y_modified.append(merge_labels[alabel])
        else:
            y_modified.append(alabel)

    return y_modified        


def adjust_index(X, labels, start_char=1, index_from=3, maxlen=None):

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if start_char is not None: # add start of sentence char
        X = [[start_char] + [w + index_from for w in x] for x in X] # shift the ids to index_from; id 3 will be shifted to 3+index_from
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)

        X      = new_X
        labels = new_labels

    return (X, labels)     


def numberize_sentences(sentences, vocab_idmap, oov_char=2):  

    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
        for wrd in sent.split():
            #wrd_id = vocab_idmap[wrd] if vocab_idmap.has_key(wrd) else oov_char
            if wrd in vocab_idmap:
                wrd_id = vocab_idmap[wrd]
            else:
                wrd_id = oov_char
            tmp_list.append(wrd_id)

        sentences_id.append(tmp_list)

    return sentences_id    

def numberize_sentences_and_report_OOV(sentences, vocab_idmap, oov_char=2):  

    sentences_id=[]
    OOV_nb = 0  

    for sid, sent in enumerate (sentences):
        tmp_list = []
 #       print sent

        for wrd in sent.split():
            if vocab_idmap.has_key(wrd):
                wrd_id = vocab_idmap[wrd]
#                print wrd_id
            else:
                wrd_id = oov_char
                OOV_nb += 1
#                print wrd_id

            tmp_list.append(wrd_id)

#        print tmp_list
#        raw_input(' ')    
        sentences_id.append(tmp_list)

    return (sentences_id, OOV_nb)   


def numberize_labels(all_str_label, label_id_map):

    label_cnt = {}
    labels    = []
    
    for a_label in all_str_label:
        labels.append(label_id_map[a_label])

        if a_label in label_cnt:
            label_cnt[a_label] += 1
        else:
            label_cnt[a_label] = 1    

    return (labels, label_cnt)



def get_label(str_label):
    if  str_label == "informative":
        return 1
    elif str_label == "not informative":
        return 0
    else:
        print("Error!!! unknown label " + str_label)
        exit(1)        

def loadEmbedding(filename, vocabs, pretrained_type):
    vocab = []
    embd = []
    #file = open(filename,'r')
    file = codecs.open(filename, 'r', encoding='IBM858')
    for line in file.readlines():
        row = line.strip().split(' ')
        #vocab.append(row[0])
        if(row[0] in vocabs):
            #if (row[0] not in vocab):
            vocab.append(row[0])
            embd.append(row[1:])
    print('Loaded ', pretrained_type, '!')
    file.close()
    return vocab,embd


def load_pretrained_embedding(vocabs, filename, pretrained_type="glove"):
    table = str.maketrans({key: None for key in string.punctuation})
    vocabs = [s.translate(table).rstrip() for s in vocabs]

    if pretrained_type=="glove":
        vocabs = [x.lower() for x in vocabs]

    if pretrained_type=="word2vec":
        vocabs = [x.upper() for x in vocabs]

    vocab, embd = loadEmbedding(filename, vocabs, pretrained_type)

    print("Found ", len(vocab), " words in ", pretrained_type,  " out of total ", len(vocabs), " words.")

    embedding = []
    for entity in vocabs:
        if entity in vocab:
            index = vocab.index(entity)
            embedding.append(embd[index])
        else:
            e = 0.01 * np.random.uniform(-1.0, 1.0, len(embd[0]))
            embedding.append(e)

    embedding = np.asarray(embedding)

    return embedding


def load_pretrained_embedding_meta_conv(vocabs, word2vec_filename, glove_filename, pretrained_type="glove"):
    table = str.maketrans({key: None for key in string.punctuation})
    vocabs = [s.translate(table).rstrip() for s in vocabs]

    if pretrained_type=="glove":
        vocabs = [x.lower() for x in vocabs]

    if pretrained_type=="word2vec":
        vocabs = [x.upper() for x in vocabs]

    print("Loading conv_word2vec!")
    word2vec_vocab, word2vec_embd = loadEmbedding(word2vec_filename, vocabs, pretrained_type="conv_word2vec")
    print("Found ", len(word2vec_vocab), " words in conv_word2vec out of total ", len(vocabs), " words.")

    print("\n\nLoading conv_glove!")
    glove_vocab, glove_embd = loadEmbedding(glove_filename, vocabs, pretrained_type="conv_glove")
    print("Found ", len(glove_vocab), " words in conv_glove out of total ", len(vocabs), " words.")

    embedding = []
    meta_count = 0
    word2vec_count = 0
    glove_count = 0
    random_count = 0
    for entity in vocabs:
        if (entity in word2vec_vocab) and (entity in glove_vocab):
            word2vec_index = word2vec_vocab.index(entity)
            glove_index = glove_vocab.index(entity)
            meta_embd = np.array([word2vec_embd[word2vec_index], glove_embd[glove_index]]).astype(np.float)
            #print(len(word2vec_embd[word2vec_index]))
            #print(len(glove_embd[glove_index]))
            #print(np.shape(meta_embd))
            avg_embedding = np.average(meta_embd, axis=0)
            #print(np.shape(avg_embedding))
            embedding.append(avg_embedding)
            meta_count = meta_count + 1

        elif (entity in word2vec_vocab):
            word2vec_index = word2vec_vocab.index(entity)
            embedding.append(word2vec_embd[word2vec_index])
            word2vec_count = word2vec_count + 1

        elif (entity in glove_vocab):
            glove_index = glove_vocab.index(entity)
            embedding.append(glove_embd[glove_index])
            glove_count = glove_count + 1

        else:
            e = 0.01 * np.random.uniform(-1.0, 1.0, len(word2vec_embd[0]))
            embedding.append(e)
            random_count = random_count + 1

    embedding = np.asarray(embedding)

    print("Meta count: ", meta_count)
    print("Only word2vec count: ", word2vec_count)
    print("Only glove count: ",glove_count)
    print("Random count: ", random_count)


    return embedding


def load_pretrained_embedding_meta_orig(vocabs, word2vec_filename, glove_filename, pretrained_type="glove"):
    table = str.maketrans({key: None for key in string.punctuation})
    vocabs = [s.translate(table).rstrip() for s in vocabs]

    vocabs_word2vec = [x.upper() for x in vocabs]
    vocabs_glove = [x.lower() for x in vocabs]

    print("Loading word2vec!")
    word2vec_vocab, word2vec_embd = loadEmbedding(word2vec_filename, vocabs_word2vec, pretrained_type="word2vec")
    print("Found ", len(word2vec_vocab), " words in word2vec out of total ", len(vocabs), " words.")

    print("\n\nLoading glove!")
    glove_vocab, glove_embd = loadEmbedding(glove_filename, vocabs_glove, pretrained_type="conv_glove")
    print("Found ", len(glove_vocab), " words in glove out of total ", len(vocabs), " words.")

    embedding = []
    meta_count = 0
    word2vec_count = 0
    glove_count = 0
    random_count = 0
    for entity in vocabs:
        if (entity.upper() in word2vec_vocab) and (entity.lower() in glove_vocab):
            word2vec_index = word2vec_vocab.index(entity.upper())
            glove_index = glove_vocab.index(entity.lower())
            meta_embd = np.array([word2vec_embd[word2vec_index], glove_embd[glove_index]]).astype(np.float)
            # print(len(word2vec_embd[word2vec_index]))
            # print(len(glove_embd[glove_index]))
            # print(np.shape(meta_embd))
            avg_embedding = np.average(meta_embd, axis=0)
            # print(np.shape(avg_embedding))
            embedding.append(avg_embedding)
            meta_count = meta_count + 1

        elif (entity.upper() in word2vec_vocab):
            word2vec_index = word2vec_vocab.index(entity.upper())
            embedding.append(word2vec_embd[word2vec_index])
            word2vec_count = word2vec_count + 1

        elif (entity.lower() in glove_vocab):
            glove_index = glove_vocab.index(entity.lower())
            embedding.append(glove_embd[glove_index])
            glove_count = glove_count + 1

        else:
            e = 0.01 * np.random.uniform(-1.0, 1.0, len(word2vec_embd[0]))
            embedding.append(e)
            random_count = random_count + 1

    embedding = np.asarray(embedding)

    print("Meta count: ", meta_count)
    print("Only word2vec count: ", word2vec_count)
    print("Only glove count: ", glove_count)
    print("Random count: ", random_count)

    return embedding


def pad_sequences(X, maxlen=100):
    new_X = np.zeros((len(X), maxlen), dtype=int)
    #print(np.shape(new_X))

    for i in range(len(X)):
        if (len(X[i]) > maxlen):
            new_X[i] = X[i][0:maxlen]
        else:
            new_X[i][0:len(X[i])] = X[i]

    return new_X


def get_feat_unig(line, U):
    all_feats = line.split()
    U.append([int(afeat.split(":")[1]) for afeat in all_feats[0:]])  # get unigrams


def one_of_k_unig(X, nb_items):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''

    X_bin = np.zeros((len(X), nb_items))
    #    np.set_printoptions(threshold=np.nan)

    for sid, asen in enumerate(X):
        for wid in asen:
            X_bin[sid, wid] = 1.

    return X_bin