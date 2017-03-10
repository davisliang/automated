import tensorflow as tf
import numpy as np

def fcLayer(x, in_shape, out_shape, activation, dropout, is_train, scope="fc"):
    
    x = tf.reshape(x, [-1, in_shape])
 
    with tf.variable_scope(scope):
        w = tf.get_variable(name="w", shape = [in_shape, out_shape], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-2))
        b = tf.get_variable(name="b", shape= [out_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        fc = tf.add(tf.matmul(x, w), b)

        with tf.variable_scope("activation"):
            output = applyActivation(fc, activation)
            #out_op = tf.cond(tf.less(is_train, tf.constant(1)), lambda: tf.nn.dropout(output, dropout), lambda: tf.identity(output))
            out_op = output 
    return out_op

def createLSTMCell(batch_size, lstm_size, n_layers, forget_bias):

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=forget_bias)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell  for i in range(n_layers)], state_is_tuple=True)
    state = lstm_cell.zero_state(batch_size, tf.float32)
   
    return lstm_cell, state

def applyActivation(x, activation):

    if activation == "tanh":
        return tf.nn.tanh(x)
    elif activation == "relu":
        return tf.nn.relu(x)
    elif activation == "sigmoid":
        return tf.nn.sigmoid(x)
    elif activation == "relu6":
        return tf.nn.relu6(x)
    else: return None


def predictionLayer(x, y, in_shape, out_shape, activation, scope="prediction"):
    
    x = tf.reshape(x, [-1, in_shape])

    with tf.variable_scope(scope):
        w = tf.get_variable(name=scope+"w", shape = [in_shape, out_shape], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-2))
        b = tf.get_variable(name=scope+"b", shape= [out_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(x, w), b)
        output = applyActivation(logits, activation)
        # Compute the mean-squared-error
        cost = tf.reduce_mean(tf.square(tf.subtract(y , output)))

    return cost, output
