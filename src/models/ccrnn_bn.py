# Add batch normalization to the input to fully conntect layers and prediction layers
# No batch norm on Recurrent layer
# When using batch normalization, make sure that keep_prob is set to 1.0 so that
# no dropout will be applied to fc layer

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
body_lstm_size = 128
context_lstm_size = 128
task_lstm_size = 128
body_n_layer = 1
context_n_layer = 1
task_n_layer = 1
context_branch_fc = 512
task_branch_fc = 512

# Data params
train_data_path = "~/tweetnet/data/train_data.pkl"
test_data_path = "~/tweetnet/data/test_data.pkl"
batch_size = 128
n_steps = 40
feature_length = 66
context_dim = 300
task_dim = 300

# Hyper- params
lr = 0.01
context_lr = lr
n_epoch = 500
topN = 4
keep_prob_val = 1.0



def buildModel(x, y_context, y_task, is_train, keep_prob, scope="multiTask"):
     
    # Assume the input shape is (batch_size, n_steps, feature_length) 

    #TASK = primary task, CONTEXT = secondary task
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, feature_length)
    x = tf.reshape(x, [-1, feature_length])
    # Split to get a list of "n_steps" tensors of shape (batch_size, feature_length)
    x = tf.split(x, n_steps, 0)

    # Create lstm cell for the shared layer 
    body_lstm_cell, body_state = createLSTMCell(batch_size, body_lstm_size, body_n_layer, forget_bias=0.0)
    # Create lstm cell for branch 1 
    context_lstm_cell, context_state = createLSTMCell(batch_size, context_lstm_size, context_n_layer, forget_bias=0.0)
    # Create lstm cells for branch 2
    task_lstm_cell, task_state = createLSTMCell(batch_size, task_lstm_size, task_n_layer, forget_bias=0.0)

    #combined_cost = tf.constant(0)
    context_cost = tf.constant(0)
    task_cost = tf.constant(0)

    #IMPLEMENTATION NOTES
    #No idea how code compiles in mcrnn... indentation makes 0 sense.
    #cant get context_output to next layer so have to use separate for loops... 
    #need to cast as float32.
    #we should try both top and bottom outputs for our targets.
    for time_step in range(n_steps):
    	# first, we construct the context lstm
        print time_step
    	with tf.variable_scope("context_branch"):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
	    (context_cell_output, context_state) = context_lstm_cell(x[time_step], context_state)
        
	with tf.variable_scope("context_fc"): 
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            context_fc_bn = tf.contrib.layers.batch_norm(context_cell_output, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn2')
            context_fc_out = fcLayer(x=context_fc_bn, in_shape=context_lstm_size, out_shape=context_branch_fc, activation=fc_activation, dropout=keep_prob, is_train=is_train, scope="fc1")
            
            context_pred_bn = tf.contrib.layers.batch_norm(context_fc_out, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn3')
            context_cost, context_output = predictionLayer(x=context_pred_bn, y=y_context, in_shape=context_branch_fc, out_shape=y_context.get_shape()[-1].value, activation=output_activation)

        # then make the body where the input is the concatenation of both text and context_output         
    	with tf.variable_scope("body_lstm"):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            #body_input = tf.concat([x[time_step],context_cell_output], 1)

            (body_cell_output, body_state) = body_lstm_cell(x[time_step], body_state)

        # finally make the output task cell.
        with tf.variable_scope("task_branch"):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

    	    task_input = tf.concat([body_cell_output, context_cell_output], 1)
            (task_cell_output,task_state) = task_lstm_cell(task_input, task_state)

        with tf.variable_scope("task_fc"):
            if time_step == n_steps - 1:
                task_fc_bn = tf.contrib.layers.batch_norm(task_cell_output, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn6')
                task_fc_out = fcLayer(x=task_fc_bn, in_shape=task_lstm_size, out_shape=task_branch_fc, activation=fc_activation, dropout=keep_prob, is_train=is_train, scope="fc2")
                task_pred_bn = tf.contrib.layers.batch_norm(task_fc_out, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn7')
                task_cost, task_output = predictionLayer(x=task_pred_bn, y=y_task, in_shape=task_branch_fc, out_shape=y_task.get_shape()[-1].value, activation=output_activation)

    return context_cost, task_cost, task_output, context_output

def print_config():
	print "-"*100
	print "Learning rate for task: ", lr
	print "Learning rate for context: "
	print "Body LSTM  size: ", body_lstm_size
	print "Context LSTM size: ", context_lstm_size
	print "Task LSTM size: ", task_lstm_size
	print "Context fc size: ", context_branch_fc
	print "Task fc size: ", task_branch_fc
	print "Number of steps: ", n_steps
	print "Batch size: ", batch_size

	print "-"*100

def trainModel(train_path = train_data_path, test_path = test_data_path):
    
    print print_config
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
    optimizer1 = tf.train.AdamOptimizer(learning_rate=lr) 
    optimizer2 = tf.train.AdamOptimizer(learning_rate=context_lr)
    is_train = tf.placeholder(tf.bool)
    n_batches = np.ceil(len(trainX) / batch_size).astype(int)
    keep_prob = tf.placeholder(tf.float32)
    
    # Build model and apply optimizer
    context_cost, task_cost, task_output, context_output = buildModel(x, y_context, y_task, is_train, keep_prob)

    # Minimize losses
    train_step1 = optimizer1.minimize(context_cost)
    train_step2 = optimizer2.minimize(task_cost)
 
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
                
                feed_dict = {x: train_x, y_context: train_y_context, y_task: train_y_task, is_train: True, keep_prob: keep_prob_val}

                #feed dict same for both train_step1 and train_step 2?
                train_step2.run(feed_dict=feed_dict)
                _, cost_task, taskOutput = sess.run(fetches = [context_cost, task_cost, task_output], feed_dict=feed_dict)
                taskCost += cost_task
            	epochTask += cost_task

                train_step1.run(feed_dict=feed_dict)
                cost_context, _, taskOutput = sess.run(fetches = [context_cost, task_cost, task_output], feed_dict=feed_dict)
                contextCost += cost_context
                epochContext += cost_context
                
                if batch !=0 and batch % 100 == 0:
                    print "Minibatch ", batch, " Missing word: ", contextCost / 100, " Hashtag: ", taskCost / 100
                    contextCost = 0
                    taskCost = 0
            print "Epoch ", epoch, "Missing word: ", epochContext / n_batches, " Hashtag: ", epochTask / n_batches


            # At the end of each epoch, run a forward pass of all testing data
            tweetCnt = 0
            correctCnt = 0
            n_test_batches = np.ceil(len(testTweetSequence) / batch_size).astype(int)

            for batch in range(n_test_batches):
                startIdx = batch*batch_size
                test_x = testX[startIdx : startIdx+batch_size, :, :]
                test_y_context = testY_context[startIdx : startIdx+batch_size, :]
                test_y_task = testY_task[startIdx : startIdx+batch_size, :]

                feed_dict = {x: test_x, y_context: test_y_context, y_task: test_y_task, is_train: True, keep_prob:1.0} 

                cost_context, cost_task, taskOutput = sess.run(fetches = [context_cost, task_cost, task_output], feed_dict=feed_dict)
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
