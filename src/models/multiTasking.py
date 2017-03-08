import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell
import numpy as np

def buildModel(x, y, lstm_size, n_steps, feature_length, initializer, activation, scope="lstmLayer"):
     
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

    for time_step in range(n_steps):
        with tf.variable_scope("SharedLSTM"):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output_0, state_0) = lstm_cell_0(x[i], state_0)
        
        with tf.variable_scope("Branch1"):         
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output_1, state_1) = lstm_cell_1(cell_output_0, state_1)
            if time_step == n_step - 1:
                fc_out1 = fcLayer(x=cell_output_1, in_shape=lstm_size_1, out_shape=512, activation="relu", dropout=0.9, scope="fc1")
                cost1 = predictionLayer(x=fc_out1, y=y, in_shape=lstm_size_1, out_shape=y.get_shape[-1].value, activation="tanh")
                combined_cost = combined_cost + cost1

        with tf.variable_scope("Branch2"):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output_2, state_2) = lstm_cell_2(cell_output_0, state_2) 
            if time_step == n_step - 1:
                fc_out2 = fcLayer(x=cell_output_2, in_shape=lstm_size_2, out_shape=512, activation="relu", dropout=0.9, scope="fc2")
                cost2 = predictionLayer(x=fc_out2, y=y, in_shape=lstm_size_2, out_shape=y.get_shape[-1].value, activation="tanh")
                combined_cost = combined_cost + cost2

    return combined_cost

def trainModel():
    
    x = tf.placeholder()
    y = tf.placeholder()
    
