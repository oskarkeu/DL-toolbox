from __future__ import print_function
import tensorflow as tf
import numpy as np


class FeedForward:

    def __init__(self, dim_specs, activation, output_activation):

        """

        :param dim_specs:  list containing dimensions of input, hidden layers and output
        :param activation:  activation function for each layer e.g. tf.nn.relu
        :param output_activation:   output activation function e.g. tf.nn.softmax
        """

        self.output_activation = output_activation
        self.n_layers = len(dim_specs) - 1
        self.output_dim = dim_specs[-1]
        self.input_dim = dim_specs[0]
        self.activation = activation
        self.dim_specs = dim_specs
        self.current = self.inputs
        self.data_set = None
        self.train_op = None
        self.output = None
        self.loss = None

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim])

        self.var_list = [tf.Variable(tf.truncated_normal(shape=[self.dim_specs[i-1], self.dim_specs[i]], stddev=0.1))
                         for i in range(1, self.n_layers)]

        self.bias_list = [tf.Variable(tf.constant(0.1, shape=[self.dim_specs[i]])) for i in range(1, self.n_layers)]

    def build_model(self, layer, output_layer):

        """

        :param layer: an instance of the Layer class for the hidden layers of the model
        :param output_layer: an instance of the Layer class for the output layer of the model
        """
        for i in range(self.n_layers - 1):
            self.current = layer.feed_forward(self.current, self.var_list[i], self.bias_list[i])
        self.current = output_layer.feed_forward(self.current, self.var_list[-1], self.bias_list[-1])
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.current)

    def train_model(self, data_inputs, data_labels, batch_size, lr, iters, eval_int):

        """

        :param data_inputs: input data matrix; rank 2 np_array of shape (batch size, feature size)
        :param data_labels: label data matrix; rank 2 np_array of shape (batch size, number of classes)
        :param batch_size: number of samples to estimate gradient for each training iteration
        :param lr: learning rate - how much to move in gradient direction in parameter space
        :param iters: number of training iterations
        :param eval_int: samples in between printing data about training progress
        """
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.data_set = zip(data_inputs, data_labels)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iters):
                np.random.shuffle(self.data_set)
                batch_x, batch_y = zip(*self.data_set[:batch_size])
                train_feed = {self.inputs: batch_x, self.labels: batch_y}
                if i % eval_int == 0:
                    print("Loss: ", self.loss.eval(sess, train_feed))
                sess.run(self.train_op, train_feed)



    def predict(self):
        pass


class Layer:

    def __init__(self, activation):
        self.activation = activation

    def feed_forward(self, layer_inputs, layer_weight, layer_bias):
        return self.activation(tf.matmul(layer_inputs, layer_weight) + layer_bias)
