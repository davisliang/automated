import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell
import numpy as np

def buildModel(x, y_context, y_task, is_train, scope="lstmLayer"):
     
    # Assume the input shape is (batch_size, n_steps, feature_length) 
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, feature_length)
    x = tf.reshape(x, [-1, feature_length])
    # Split to get a list of "n_steps" tensors of shape (batch_size, feature_length)
    x = tf.split(x, n_steps, 0)

    # Create lstm cell for the shared layer 
    lstm_cell_0, state_0 = createLSTMCell(batch_size, lstm_size_0, n_layer_0, forget_bias)
    # Create lstm cell for branch 1 
    lstm_cell_1, state_1 = createLSTMCell(batch_size, lstm_size_1, n_layer_1, forget_bias)
    # Create lstm cells for branch 2
    lstm_cell_2, state_2 = createLSTMCell(batch_size, lstm_size_2, n_layer_2, forget_bias)

    combined_cost = tf.constant(0)
    cost1 = tf.constant(0)
    cost2 = tf.constant(0)

    for time_step in range(n_steps):
        with tf.variable_scope("SharedLSTM"):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output_0, state_0) = lstm_cell_0(x[i], state_0)
        
        with tf.variable_scope("Branch_context"):         
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output_1, state_1) = lstm_cell_1(cell_output_0, state_1)
            if time_step == n_step - 1:
                fc_out1 = fcLayer(x=cell_output_1, in_shape=lstm_size_1, out_shape=branch1_hidden, activation=fc_activation, dropout=dropout, is_train, scope="fc1")
                cost1 = predictionLayer(x=fc_out1, y=y_context, in_shape=lstm_size_1, out_shape=y_context.get_shape[-1].value, activation=output_activation)

        with tf.variable_scope("Branch_task"):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output_2, state_2) = lstm_cell_2(cell_output_0, state_2) 
            if time_step == n_step - 1:
                fc_out2 = fcLayer(x=cell_output_2, in_shape=lstm_size_2, out_shape=branch2_hidden, activation=fc_activation, dropout=dropout, is_train, scope="fc2")
                cost2 = predictionLayer(x=fc_out2, y=y_task, in_shape=lstm_size_2, out_shape=y_task.get_shape[-1].value, activation=output_activation)

    combined_cost = cost1 + cost2
    return combined_cost, cost1, cost2

def trainModel(dataset_path = dataset_path):
    
    x = tf.placeholder(tf.float32, shape=(batch_size, n_steps, feature_length))
    y_context = tf.placeholder(tf.float32, shape=(batch_size, context_dim))
    y_task = tf.placeholder(tf.float32, shape=(batch_size, task_dim))

    optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
    is_train = tf.placeholder(tf.int32)
  
    total_cost, cost1, cost2 = buildModel(x, y, is_train)
    train_step = optimizer.minimize(total_cost)
 
    # Start running operations on the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    with sess.as_default():
        for epoch in range(n_epoch):
            


# Model params
fc_activation = "relu"
output_activation = "tanh"
dropout = 0.9
lstm_size_0 = 512
lstm_size_1 = 512
lstm_size_2 = 512
branch1_fc = 512
branch2_fc = 512

# Data params
batch_size = 128
n_step = 40
feature_length = 66
context_dim = 300
task_dim = 300

# Hyper- params
lr = 0.001
n_epoch = 500
