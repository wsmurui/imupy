import tensorflow as tf
import numpy as np

class PDRModel(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        hidden_size = config.hidden_size
        data_size = config.data_size

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)

        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("input_data", [data_size, hidden_size], tf.float32)
        input_data = tf.placeholder(dtype=tf.float32, shape=[None])
        inout_embedding = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inout_embedding, config.keep_prob)