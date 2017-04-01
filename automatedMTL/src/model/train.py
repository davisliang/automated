import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
from os.path import expanduser
import sys
import mcrnn_model
from mcrnn_model import model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))
from tf_utils import fcLayer, createLSTMCell, applyActivation, predictionLayer
from load_batch import get_file_identifiers, get_classes, load_data, get_word2vec, load_batch

def get_data(data_path):
    data_stats = pickle.load(open(expanduser(data_path + "/rt_stats.pkl")))
    max_length, nPos, nNeg, trainPercent, testPercent = data_stats["longest"], data_stats[0], data_stats[1], data_stats['trainPercent'], data_stats['testPercent']
    word2vec_dic = get_word2vec("~/tweetnet/data/word2vec_dict.pkl")
    missing_word_dic = pickle.load(open(expanduser(data_path + "/missing_word_dic.pkl")))
    nTest = int(testPercent*nPos) + int(testPercent*nNeg)
    nTrain = nPos + nNeg - nTest

    return max_length, nPos, nNeg, trainPercent, testPercent, word2vec_dic, missing_word_dic, nTest, nTrain


def trainModel():
    
    M = model()
    data_path = "~/automatedMTL/data/rotten_tomato"
    max_length, nPos, nNeg, trainPercent, testPercent, word2vec_dic, missing_word_dic, nTest, nTrain = get_data(data_path)
    
    x = tf.placeholder(tf.float32, shape=(None, M.max_length, M.feature_length))
    y_context = tf.placeholder(tf.float32, shape=(None, M.context_dim))
    y_task = tf.placeholder(tf.float32, shape=(None, M.task_dim))
    
    optimizer1 = tf.train.AdamOptimizer(learning_rate=M.context_lr)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=M.lr)
    is_train = tf.placeholder(tf.int32)
    n_train_batches = np.ceil(nTrain / M.batch_size).astype(int)
    keep_prob = tf.placeholder(tf.float32)
    
    context_cost, task_cost, task_output, context_output = M.buildModel(x, y_context, y_task, is_train, keep_prob)
    train_step1 = optimizer1.minimize(context_cost)
    train_step2 = optimizer2.minimize(task_cost)

    # Start running operations on the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    with sess.as_default():
        for epoch in range(100):
            taskCost = 0
            contextCost = 0

            all_classes, train_file, test_file = load_data(data_path)
            start_idx = 0 
            for minibatch in range(n_train_batches):
                encoded_batch, batch_classes, batch_missing_word_encoded, batch_missing_word, batch_identifier, batch_text, batch_length = load_batch(word2vec_dic, missing_word_dic, M.feature_length, max_length, data_path+"/Train/", 1, train_file, test_file, all_classes, start_idx, M.batch_size)
                start_idx += M.batch_size
        
                feed_dict = {x: encoded_batch, y_context: batch_missing_word_encoded, y_task: batch_classes, is_train:1, keep_prob:0.5}
                
		train_step1.run(feed_dict=feed_dict)
	        context_cost_val, _, _ = sess.run(fetches = [context_cost, task_cost, task_output], feed_dict=feed_dict)
                contextCost += context_cost_val

                train_step2.run(feed_dict=feed_dict)
	        _, task_cost_val, _ = sess.run(fetches = [context_cost, task_cost, task_output], feed_dict=feed_dict)
                taskCost += task_cost_val

                #if minibatch !=0 and minibatch % 100 == 0:
                print "Minibatch ", minibatch, " Missing Word: ", contextCost , " Classification: ", taskCost 
                contextCost = 0
                taskCost = 0

            start_idx = 0
	    accuracy = 0

            for i in range(nTest):
                encoded_batch, batch_classes, batch_missing_word_encoded, batch_missing_word, batch_identifier, batch_text, batch_length = load_batch(word2vec_dic, missing_word_dic, M.feature_length, max_length, data_path+"/Test/", 0, train_file, test_file, all_classes, start_idx, 1)
		start_idx += 1
                feed_dict = {x:encoded_batch, y_context: batch_missing_word_encoded, y_task: batch_classes, is_train:0, keep_prob:0.5}
                task_output_val = sess.run(fetches = [task_output], feed_dict=feed_dict)
		accuracy += is_correct(batch_classes, task_output_val)
            print "The accuracy in epoch ", epoch, " is: ", accuracy * 1.0 / nTest

def is_correct(target, output):
    prediction = np.argmax(output)
    target = np.argmax(target)
    #print prediction, target
    return prediction == target

            
if __name__ == "__main__":
    trainModel()    
