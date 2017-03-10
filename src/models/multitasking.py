import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
from os.path import expanduser
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))
from tf_utils import fcLayer, createLSTMCell, applyActivation, predictionLayer
from predContext import predContext, createHtDict

# Model params
# 0 -- shared;  1 -- context;  2 -- task
fc_activation = "tanh"
output_activation = "tanh"
dropout = 0.0
lstm_size_0 = 512
lstm_size_1 = 512
lstm_size_2 = 512
n_layer_0 = 1
n_layer_1 = 1
n_layer_2 = 1
branch1_fc = 512
branch2_fc = 512

# Data params
train_data_path = "~/tweetnet/data/train_data.pkl"
test_data_path = "~/tweetnet/data/test_data.pkl"
batch_size = 128
n_steps = 40
feature_length = 66
context_dim = 300
task_dim = 300

# Hyper- params
lr = 0.0001
n_epoch = 500
topN = 4

def buildModel(x, y_context, y_task, is_train, scope="multiTask"):
     
    # Assume the input shape is (batch_size, n_steps, feature_length) 
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    print x.get_shape()
    # Reshaping to (n_steps*batch_size, feature_length)
    x = tf.reshape(x, [-1, feature_length])
    # Split to get a list of "n_steps" tensors of shape (batch_size, feature_length)
    print x.get_shape()
    x = tf.split(x, n_steps, 0)

    # Create lstm cell for the shared layer 
    lstm_cell_0, state_0 = createLSTMCell(batch_size, lstm_size_0, n_layer_0, forget_bias=0.0)
    # Create lstm cell for branch 1 
    lstm_cell_1, state_1 = createLSTMCell(batch_size, lstm_size_1, n_layer_1, forget_bias=0.0)
    # Create lstm cells for branch 2
    lstm_cell_2, state_2 = createLSTMCell(batch_size, lstm_size_2, n_layer_2, forget_bias=0.0)

    combined_cost = tf.constant(0)
    cost1 = tf.constant(0)
    cost2 = tf.constant(0)

    for time_step in range(n_steps):
        with tf.variable_scope("SharedLSTM"):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output_0, state_0) = lstm_cell_0(x[time_step], state_0)
        
        #with tf.variable_scope("Branch_context"):         
        #    if time_step > 0: 
        #        tf.get_variable_scope().reuse_variables()
        #    (cell_output_1, state_1) = lstm_cell_1(cell_output_0, state_1)
        with tf.variable_scope("Branch_context_fc"):
            if time_step == n_steps - 1:
                fc_out1 = fcLayer(x=cell_output_0, in_shape=lstm_size_0, out_shape=branch1_fc, activation=fc_activation, dropout=dropout, is_train=is_train, scope="fc1")
                cost1, output1 = predictionLayer(x=fc_out1, y=y_context, in_shape=branch1_fc, out_shape=y_context.get_shape()[-1].value, activation=output_activation)

        #with tf.variable_scope("Branch_task"):
        #    if time_step > 0: 
        #        tf.get_variable_scope().reuse_variables()
        #    (cell_output_2, state_2) = lstm_cell_2(cell_output_0, state_2) 
        with tf.variable_scope("Branch_task_fc"):
            if time_step == n_steps - 1:
                fc_out2 = fcLayer(x=cell_output_0, in_shape=lstm_size_0, out_shape=branch2_fc, activation=fc_activation, dropout=dropout, is_train=is_train, scope="fc2")
                cost2, output2 = predictionLayer(x=fc_out2, y=y_task, in_shape=branch1_fc, out_shape=y_task.get_shape()[-1].value, activation=output_activation)

    combined_cost = cost1 + cost2
    
    return combined_cost, cost1, cost2, output2, output1

            

def trainModel(train_path = train_data_path, test_path = test_data_path):
    
    # Load data as np arrays
    train_data = pickle.load(open(expanduser(train_path)))
    trainX, trainY_task, trainY_context = train_data[0], train_data[1], train_data[2]
    
    test_data = pickle.load(open(expanduser(test_path)))
    testX, testY_task, testY_context = test_data[0], test_data[1], test_data[2]
   
    htDic, testTweets, testHashtags, testMw, testTweetSequence, testHashtagSequence, testMwSequence, testStartIdx = prepareForTest() 

    
    # place holder for X and Y
    x = tf.placeholder(tf.float32, shape=(batch_size, n_steps, feature_length))
    y_context = tf.placeholder(tf.float32, shape=(batch_size, context_dim))
    y_task = tf.placeholder(tf.float32, shape=(batch_size, task_dim))

    # Setting up training variables
    optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
    is_train = tf.placeholder(tf.int32)
    n_batches = np.ceil(len(trainX) / batch_size).astype(int)
    
    # Build model and apply optimizer
    total_cost, cost1, cost2, output2, output1 = buildModel(x, y_context, y_task, is_train)

    # Minimize losses
    train_step1 = optimizer.minimize(cost1)
    train_step2 = optimizer.minimize(cost2)
 
    # Start running operations on the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    trainables = tf.trainable_variables()
    for var in trainables:
        print var.name

    with sess.as_default():
        for epoch in range(n_epoch):
            taskCost = 0
            contextCost = 0
            epochTask = 0
            epochContext = 0
            for batch in range(n_batches):
                startIdx = batch*batch_size
                train_x = trainX[startIdx : startIdx+batch_size, :, :]
                train_y_context = trainY_context[startIdx : startIdx+batch_size, :]
                train_y_task = trainY_task[startIdx : startIdx+batch_size, :]        
                
                feed_dict = {x: train_x, y_context: train_y_context, y_task: train_y_task, is_train: 1}

                train_step2.run(feed_dict=feed_dict)
                combined_cost, _, cost_task, taskOutput = sess.run(fetches = [total_cost, cost1, cost2, output2], feed_dict=feed_dict)
                taskCost += cost_task
            
                train_step1.run(feed_dict=feed_dict)
                combined_cost, cost_context, _, taskOutput = sess.run(fetches = [total_cost, cost1, cost2, output2], feed_dict=feed_dict)
                contextCost += cost_context
                epochTask += cost_task
                epochContext += cost_context
                
                if batch !=0 and batch % 100 == 0:
                    print "Minibatch ", batch, " Missing Word: ", contextCost / 100, " Hashtag: ", taskCost / 100
                    contextCost = 0
                    taskCost = 0
            print "Epoch ", epoch, "Missing Word: ", epochContext / n_batches, " Hashtag: ", epochTask / n_batches


            # At the end of each epoch, run a forward pass of all testing data

    	    #tweetStartIdx = 0
    	    tweetCnt = 0
            correctCnt = 0
            n_test_batches = np.ceil(len(testTweetSequence) / batch_size).astype(int)

    	    for batch in range(n_test_batches):
                startIdx = batch*batch_size
                test_x = testX[startIdx : startIdx+batch_size, :, :]
                test_y_context = testY_context[startIdx : startIdx+batch_size, :]
                test_y_task = testY_task[startIdx : startIdx+batch_size, :]

                feed_dict = {x: test_x, y_context: test_y_context, y_task: test_y_task} 

        	combined_cost, cost_context, cost_task, taskOutput = sess.run(fetches = [total_cost, cost1, cost2, output2], feed_dict=feed_dict)
                print taskOutput.shape
                for i in range(batch_size):
        	    if testTweetSequence[startIdx+i][-1] == chr(3):
            	        topNht, isCorrect, topNdist = predContext(htDic, np.reshape(taskOutput[i,:], [1,task_dim]), topN, testHashtags[tweetCnt])
            	        #tweetStartIdx = testIdx + 1
                        if isCorrect: correctCnt += 1

                        print "Tweet: ", testTweets[tweetCnt]
                        print "True label is: ", testHashtags[tweetCnt]
                        print "Predicted labels are: ", topNht

                        tweetCnt += 1

            accuracy = correctCnt * 1.0 / len(testTweets)
            print "Testing accuracy is: ", accuracy
 


def prepareForTest(dataset_path="~/tweetnet/data/text_data.pkl"):
    
    text_data = pickle.load(open(expanduser(dataset_path)))
    testTweets, testHashtags, testMw, testTweetSequence, testHashtagSequence, testMwSequence, testStartIdx = text_data[0],text_data[1],text_data[2],text_data[3],text_data[4],text_data[5],text_data[6]

    dictionary = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl")))
    htDic = createHtDict(dictionary, testHashtags)
    return htDic, testTweets, testHashtags, testMw, testTweetSequence, testHashtagSequence, testMwSequence, testStartIdx




trainModel(train_path = train_data_path, test_path=test_data_path)
