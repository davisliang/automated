import tensorflow as tf
import numpy as np

def fcLayer(x, in_shape, out_shape, activation, dropout, is_train, scope="fc"):
    
    x = tf.reshape(x, [-1, in_shape])
 
    with tf.variable_scope(scope):
        w = tf.get_variable(name="w", shape = [in_shape, out_shape], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b", shape= [out_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        fc = tf.add(tf.matmul(x, w), b)

        with tf.variable_scope("activation"):
            output = applyActivation(fc, activation)
            #out_op = tf.nn.dropout(output, dropout)
            out_op = output

    return out_op

def createGRUCell(batch_size, lstm_size):
    gru_cell = tf.contrib.rnn.GRUCell(num_units=lstm_size, activation=tf.tanh)
    state=gru_cell.zero_state(batch_size, tf.float32)

    return gru_cell, state

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
    elif activation == "softmax":
	return tf.nn.softmax(x)
    else: return None

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def predictionLayer(x, y, in_shape, out_shape, activation, scope="prediction"):
    
    x = tf.reshape(x, [-1, in_shape])

    with tf.variable_scope(scope):
        w = tf.get_variable(name=scope+"w", shape = [in_shape, out_shape], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name=scope+"b", shape= [out_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(x, w), b)
        output = applyActivation(logits, activation)
    return output, logits

def compute_cost(logit, y, out_type, max_length, batch_size, embed_dim, activation):
    if out_type=="last_only":
        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)
        cost = tf.reduce_mean(cost, reduction_indices=1)
    else:
        pred_out = applyActivation(logit, activation)
	pred_out = tf.reshape(pred_out, [batch_size, max_length, embed_dim])
        mse = tf.reduce_mean(tf.square(tf.subtract(y, pred_out)), reduction_indices=2)
	mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
        mse *= mask
        mse = tf.reduce_sum(mse, reduction_indices=1)
        mse /= tf.cast(length(y), tf.float32)
        cost = mse
    cost = tf.reduce_mean(cost, reduction_indices=0)
    print "final cost shape: ", cost.get_shape()
    return cost 
