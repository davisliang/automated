import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
from os.path import expanduser
import sys
#from mcrnn_model_gen2 import model
from ccrnn_model_gen import model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))
from tf_utils import fcLayer, createLSTMCell, applyActivation, predictionLayer
from load_batch_val import get_file_identifiers, get_classes, load_data, get_word2vec, load_batch
#from reformat import reformat_data
from load_util import class_look_up

def get_data(data_path):
    data_stats = pickle.load(open(expanduser(data_path + "/stats.pkl")))
    n_classes, n_data, n_data_per_class, trainPercent, testPercent = data_stats['n_classes'], data_stats['n_data'], data_stats['n_data_per_class'],data_stats['trainPercent'], data_stats['testPercent']
    word2vec_dic = get_word2vec("~/tweetnet/data/word2vec_dict.pkl")
    missing_word_dic = pickle.load(open(expanduser(data_path + "/missing_word_dic.pkl")))
    n_test = int(testPercent * n_data)
    n_train = n_data - n_test

    return n_classes, word2vec_dic, n_test, n_train, missing_word_dic

def get_data_ag(data_path):

    data_stats = pickle.load(open(expanduser(data_path + "/stats.pkl")))
    word2vec_dic = get_word2vec("~/tweetnet/data/word2vec_dict.pkl")

    n_classes, n_data, n_data_per_class, n_train_data, n_test_data, n_validation_data, max_length = data_stats['n_classes'], data_stats['n_data'], data_stats['n_data_per_class'],data_stats['n_train_data'], data_stats['n_test_data'], data_stats["n_validation_data"], data_stats['max_length']
    return n_classes, word2vec_dic, n_train_data, n_test_data, n_validation_data



def trainModel(M):

    # M = model()
    data_path = "~/tweetnet/automatedMTL/data/ag_news_csv"

    # Reformat the data according to the secondary task
    # Create class look up table
    # max_length = reformat_data(data_path, M.secondary_task == "missing word")
    # class_look_up(data_path)

    if "rotten_tomato" in data_path:
        n_classes, word2vec_dic, n_test, n_train, missing_word_dic = get_data(data_path)

    else:
        n_classes, word2vec_dic, n_train, n_test, n_validation = get_data_ag(data_path)
    x = tf.placeholder(tf.float32, shape=(None, M.max_length, M.feature_length))
    if M.secondary_task == "missing word":
        y_context = tf.placeholder(tf.float32, shape=(None, M.context_dim))
    else:
        y_context = tf.placeholder(tf.float32, shape=(None, M.max_length, M.feature_length))
    y_task = tf.placeholder(tf.float32, shape=(None, M.task_dim))

    is_train = tf.placeholder(tf.int32)
    n_train_batches = np.ceil(n_train / M.batch_size).astype(int)
    n_test_batches = np.ceil(n_test / M.batch_size).astype(int)
    if "rotten_tomato" not in data_path:
        n_val_batches = np.ceil(n_validation / M.batch_size).astype(int)
    global_step = tf.Variable(0, trainable=False)
    keep_prob = tf.placeholder(tf.float32)
    context_lr = tf.placeholder(tf.float32)
    task_lr = tf.placeholder(tf.float32)
    decay_context_lr = tf.train.exponential_decay(context_lr, global_step, 500, 0.96, staircase=True)
    decay_task_lr = tf.train.exponential_decay(task_lr, global_step, 500, 0.96, staircase=True)
    optimizer1 = tf.train.AdamOptimizer(learning_rate=context_lr)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=task_lr)

    context_cost, task_cost, task_output, context_output = M.buildModel(x, y_context, y_task, is_train, keep_prob)
    
    context_vars = []
    task_vars = []
    for var in tf.trainable_variables():
	if "context" not in var.name: task_vars.append(var)
	if "task" not in var.name: context_vars.append(var)
    for var in context_vars:
        print "Context variable: ", var.name
    print ("\n")
    for var in task_vars:
	print "Task variables: ", var.name

    if M.is_multi_task:
        context_gradients = optimizer1.compute_gradients(context_cost, context_vars)
        for i, (cg, cv) in enumerate(context_gradients):
            if cg is not None:
                context_gradients[i] = (tf.clip_by_norm(cg, 1.0), cv)
        train_step1 = optimizer1.apply_gradients(context_gradients, global_step=global_step)
    task_gradients = optimizer2.compute_gradients(task_cost, task_vars)
    for i, (tg, tv) in enumerate(task_gradients):
        if tg is not None:
            task_gradients[i] = (tf.clip_by_norm(tg, 1.0), tv)
    train_step2 = optimizer2.apply_gradients(task_gradients, global_step=global_step)

    validation_accuracy_list = np.zeros((M.n_epoch))
    test_accuracy_list = np.zeros((M.n_epoch))
    # Start running operations on the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        print "Total number of epoch is: ", M.n_epoch
        for epoch in range(M.n_epoch):
            taskCost = 0
            contextCost = 0

            all_classes, train_file, test_file, val_file = load_data(data_path)
            start_idx = 0
            #for minibatch in range(10):
	    for minibatch in range(n_train_batches):
                encoded_batch, batch_classes, batch_context_encoded, batch_context, batch_identifier, batch_text, batch_length = load_batch(n_classes, word2vec_dic, {}, M.feature_length, M.max_length, data_path+"/Train/", 1, 0, train_file, test_file, val_file, all_classes, start_idx, M.batch_size, M.secondary_task)
                start_idx += M.batch_size
	
		if M.is_multi_task:
                    feed_dict = {x: encoded_batch, y_context: batch_context_encoded, y_task: batch_classes, is_train:1, keep_prob:0.5, context_lr:(1-epoch*1.0/M.n_epoch)*M.lr, task_lr:epoch*1.0/M.n_epoch*M.lr}
                else:
		    feed_dict = {x: encoded_batch, y_context: batch_context_encoded, y_task: batch_classes, is_train:1, keep_prob:0.5, context_lr: 0.0, task_lr:M.lr}
                if M.is_multi_task:
                    train_step1.run(feed_dict=feed_dict)
	            context_cost_val, _, _ = sess.run(fetches = [context_cost, task_cost, task_output], feed_dict=feed_dict)
                    contextCost += context_cost_val

                train_step2.run(feed_dict=feed_dict)
	        _, task_cost_val, _ = sess.run(fetches = [context_cost, task_cost, task_output], feed_dict=feed_dict)
                taskCost += task_cost_val

                # print "Minibatch ", minibatch, " Missing Word: ", contextCost , " Classification: ", taskCost
                contextCost = 0
                taskCost = 0

            start_idx = 0
	    validation_accuracy = 0
            if "rotten_tomato" not in data_path:
                for i in range(n_val_batches):
                    encoded_batch, batch_classes, batch_context_encoded, batch_context, batch_identifier, batch_text, batch_length = load_batch(n_classes, word2vec_dic, {}, M.feature_length, M.max_length, data_path+"/Validation/", 0, 1, train_file, test_file,val_file, all_classes, start_idx, M.batch_size, M.secondary_task)
		    start_idx += M.batch_size
                    feed_dict = {x:encoded_batch, y_context: batch_context_encoded, y_task: batch_classes, is_train:0, keep_prob:1.0}
                    task_output_val = sess.run(fetches = [task_output], feed_dict=feed_dict) #task output val is list of a single element, which is a numpy array of suze (batch_size, n_classes)
		    task_output_val = task_output_val[0] 
		    batch_accuracy = is_correct(M, batch_classes, task_output_val)
		    validation_accuracy += batch_accuracy
		    #print "Batch accuracy is: ", batch_accuracy
	        validation_accuracy_list[epoch] = validation_accuracy * 1.0 / n_val_batches
                print "\n"
                print "The validation accuracy in epoch ", epoch, " is: ", validation_accuracy * 1.0 / n_val_batches

	    start_idx = 0
	    test_accuracy = 0
            for i in range(n_test_batches):
                encoded_batch, batch_classes, batch_context_encoded, batch_context, batch_identifier, batch_text, batch_length = load_batch(n_classes, word2vec_dic, {}, M.feature_length, M.max_length, data_path+"/Test/", 0, 0, train_file, test_file, val_file, all_classes, start_idx, M.batch_size, M.secondary_task)
		start_idx += M.batch_size
                feed_dict = {x:encoded_batch, y_context: batch_context_encoded, y_task: batch_classes, is_train:0, keep_prob:1.0}
                task_output_val = sess.run(fetches = [task_output], feed_dict=feed_dict) #task output val is list of a single element, which is a numpy array of suze (batch_size, n_classes)
		task_output_val = task_output_val[0] 
		batch_accuracy = is_correct(M, batch_classes, task_output_val)
		test_accuracy += batch_accuracy
		#print "Batch accuracy is: ", batch_accuracy
	    test_accuracy_list[epoch] = test_accuracy * 1.0 / n_test_batches
            print "The test_accuracy in epoch ", epoch, " is: ", test_accuracy * 1.0 / n_test_batches
        tf.reset_default_graph()
        return validation_accuracy_list, test_accuracy_list


def is_correct(M, target, output):
    batch_accuracy = 0
    for r in range(M.batch_size):
        prediction = np.argmax(output[r,:])
	label = np.argmax(target[r,:])
	#print prediction, label
	if prediction == label:
	    batch_accuracy += 1
    return batch_accuracy * 1.0 / M.batch_size
    #prediction = np.argmax(output)
    #target = np.argmax(target)
    #print prediction, target
    #return prediction == target


if __name__ == "__main__":
    M = model()
    trainModel(M)
