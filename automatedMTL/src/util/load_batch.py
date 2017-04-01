import cPickle as pickle
import numpy as np
import os
import random
from os.path import expanduser
from reformat import reformat_data
from load_util import class_look_up

# The files are named from 0.txt to n.txt
# This function returns a list of all shuffled file names 

def get_file_identifiers(data_path):
    ids = []
    f = open(expanduser(data_path))    
    for l in f.readlines():
        ids.append(int(l.split(" ")[0]))
    random.shuffle(ids)
    return ids

def get_classes(all_classes, id):
    return all_classes[id][0]

def get_text_by_batch(data_path, is_train, train_file, test_file, all_classes, start_idx, batch_size):
    if is_train:
        identifiers = train_file
    else:
        identifiers = test_file
    batch_identifiers = identifiers[start_idx: start_idx + batch_size]
    
    batch_text = []
    for idx in batch_identifiers:
        text = open(expanduser(data_path + get_classes(all_classes, idx)+ "/" + str(idx)+".txt"))
        batch_text.append(text.read())

    return batch_identifiers, batch_text

def load_data(data_path):
    
    all_classes = pickle.load(open(expanduser(data_path + '/classes.pkl')))
    test_file = get_file_identifiers(data_path + "/test_classes.txt")
    train_file = get_file_identifiers(data_path + "/train_classes.txt")
    return all_classes, train_file, test_file

def get_word2vec(data_path):
    # TO DO: download word2vec!
    word2vec_dic = pickle.load(open(expanduser(data_path)))
    return word2vec_dic

# Unknown symbols are UNK 
# Missing word symbols are zeros
# EOS are EOS

def encode_sequence(word2vec_dic, sequence, encode_dim, max_len):
    sequence_by_word = sequence.split(" ")
    encoded_seq = np.zeros((max_len, encode_dim))
    for i in range(len(sequence_by_word)):
        word = sequence_by_word[i]
        if word2vec_dic.get(word) == None:
            encoded_seq[i, :] = word2vec_dic["UNK"]
        else:
            if word != "REMOVE":
                encoded_seq[i, :] = word2vec_dic[word]
	    else:
		encoded_seq[i, :] = word2vec_dic["_"]
    return encoded_seq, len(sequence_by_word)

def encode_sequence_generation(word2vec_dic, sequence, encode_dim, max_len):
    sequence_by_word = sequence.split(" ")
    encoded_seq = np.zeros((max_len, encode_dim))
    for i in range(1, len(sequence_by_word)):
        word = sequence_by_word[i]
        if word2vec_dic.get(word) == None:
            encoded_seq[i-1, :] = word2vec_dic["UNK"]
        else:
            encoded_seq[i-1, :] = word2vec_dic[word]
    encoded_seq[len(sequence_by_word)-1, :] = word2vec_dic["EOS"]
    context_target = sequence_by_word[1:len(sequence_by_word)] + ["EOS"]

    return encoded_seq, context_target, len(sequence_by_word)

def oneHot(nclasses, idx):
    one_hot = np.zeros((nclasses))
    one_hot[idx] = 1
    return one_hot

def load_batch(n_classes, word2vec_dic, missing_word_dic, encode_dim, max_len, data_path, is_train, train_file, test_file, all_classes, start_idx, batch_size, automated_task):
    batch_identifiers, batch_text = get_text_by_batch(data_path, is_train, train_file, test_file, all_classes, start_idx, batch_size)
    encoded_batch = np.zeros((batch_size, max_len, encode_dim))
    batch_classes = np.zeros((batch_size, n_classes))
    batch_context_encoded = np.zeros((batch_size, encode_dim))
    if automated_task == "word generation": batch_context_encoded = np.zeros((batch_size, max_len, encode_dim))
    batch_context = []
    batch_length = []
    for i in range(batch_size):
        encoded_batch[i,:, :], text_length = encode_sequence(word2vec_dic, batch_text[i], encode_dim, max_len)
        batch_classes[i,:] = oneHot(n_classes, all_classes[batch_identifiers[i]][-1])
        if automated_task != "word generation":
            batch_context_encoded[i,:] = word2vec_dic[missing_word_dic[batch_identifiers[i]]]
            batch_context.append(missing_word_dic[batch_identifiers[i]])
        else:
	    batch_context_encoded[i, :, :], context_target, text_length = encode_sequence_generation(word2vec_dic, batch_text[i], encode_dim, max_len)
	    batch_context.append(context_target)
        batch_length.append(text_length)
    return encoded_batch, batch_classes, batch_context_encoded, batch_context, batch_identifiers, batch_text, batch_length

if __name__ == "__main__":
    data_path = "~/automatedMTL/data/rotten_tomato"
    max_length = reformat_data(data_path, False)
    class_look_up(data_path)
    data_stats = pickle.load(open(expanduser(data_path + "/stats.pkl")))
    n_classes, n_data, n_data_per_class, trainPercent, testPercent = data_stats['n_classes'], data_stats['n_data'], data_stats['n_data_per_class'],data_stats['trainPercent'], data_stats['testPercent']
    word2vec_dic = get_word2vec("~/tweetnet/data/word2vec_dict.pkl")
    missing_word_dic = pickle.load(open(expanduser(data_path + "/missing_word_dic.pkl")))
    for epoch in range(3):
        dic = {}
        all_classes, train_file, test_file = load_data(data_path)
        start_idx = 0
        for minibatch in range(73):
            encoded_batch, batch_classes, batch_context_encoded, batch_context, batch_identifier, batch_text, batch_length = load_batch(n_classes, word2vec_dic, missing_word_dic, 300, max_length, data_path+"/Train/", 1, train_file, test_file, all_classes, start_idx, 128, automated_task="word generation")
            start_idx += 128
            print batch_context
            for i in batch_identifier:
                if dic.get(i) != None: print "Wrong"
                else: dic[i] = 1
