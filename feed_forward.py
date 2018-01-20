
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
        self.n_layers = len(dim_specs)
        self.output_dim = dim_specs[-1]
        self.input_dim = dim_specs[0]
        self.activation = activation
        self.dim_specs = dim_specs
        self.data_set = None
        self.train_op = None
        self.output = None
        self.loss = None

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim])

        self.var_list = [tf.Variable(tf.truncated_normal(shape=[self.dim_specs[i-1], self.dim_specs[i]], stddev=0.1))
                         for i in range(1, self.n_layers)]

        self.bias_list = [tf.Variable(tf.constant(0.1, shape=[1, self.dim_specs[i]])) for i in range(1, self.n_layers)]

        self.current = self.inputs

    def build_model(self, layer, output_layer):

        """

        :param layer: an instance of the Layer class for the hidden layers of the model
        :param output_layer: an instance of the Layer class for the output layer of the model
        """
        for i in range(self.n_layers - 2):
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
        self.data_set = list(zip(data_inputs, data_labels))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iters):
                np.random.shuffle(self.data_set)
                batch_x, batch_y = list(zip(*self.data_set[:batch_size]))
                train_feed = {self.inputs: batch_x, self.labels: batch_y}
                if i % eval_int == 0:
                    print("Loss: ", self.loss.eval(sess, train_feed))
                sess.run(self.train_op, train_feed)

    def predict(self):
        pass


class LadderNetwork:

    def __init__(self, dim_specs, activation, output_activation, noise_std):
        self.corrupted_encoder = LadderEncoder(dim_specs, activation, output_activation, noise_std)
        self.clean_encoder = LadderEncoder(dim_specs, activation, output_activation, 0)
        self.rung = LadderRung(activation, noise_std)
        self.top_rung = LadderRung(output_activation, noise_std)
        self.corrupted_encoder.build_encoder(self.rung, self.top_rung)
        self.supervised_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.corrupted_encoder.labels,
                                                                       logits=self.corrupted_encoder.current)


class LadderEncoder(FeedForward):

    def __init__(self, dim_specs, activation, output_activation, noise_std):

        """
        :param noise_std: standard deviation of gaussian noise used for corrupting layers
        """
        super(FeedForward).__init__(dim_specs, activation, output_activation)

        self.noise_std = noise_std

        self.scaling_list = [tf.Variable(tf.truncated_normal(shape=[self.dim_specs[i-1, i]], stddev=0.1))
                             for i in range(self.n_layers)]

        self.stored_values = []
        self.stored_means = []
        self.stored_stds = []

    def build_encoder(self, layer, output_layer):
        self.current = layer.corrupt(self.current, layer.noise_std, self.input_dim)
        for i in range(self.n_layers - 2):
            self.current = layer.climb_up(self.current, self.var_list[i], self.bias_list[i],
                                       self.scaling_list[i], self.dim_specs[i])
        self.current = output_layer.climb_up(self.current, self.var_list[-1], self.bias_list[-1],
                                          self.scaling_list[-1], self.dim_specs[-1])


class LadderDecoder:

    def __init__(self, dim_specs):
        self.dim_specs = dim_specs
        self.n_layers = len(dim_specs)
        self.denoising_weights = [tf.Variable(tf.truncated_normal(shape=[dim_specs[i], 10]))
                                  for i in range(self.n_layers)]


class Layer:

    def __init__(self, activation):
        self.activation = activation

    def feed_forward(self, layer_inputs, layer_weight, layer_bias):
        return self.activation(tf.matmul(layer_inputs, layer_weight) + layer_bias)


class LadderRung(Layer):

    def __init__(self, activation, noise_std):
        super(Layer).__init__(activation)
        self.noise_std = noise_std

    def climb_up(self, layer_inputs, layer_weights, layer_bias, layer_scaling, latent_dim):
        latent_raw = tf.matmul(layer_inputs, layer_weights)
        batch_mean, batch_std = tf.nn.moments(latent_raw, axes=0)
        latent_normalized = (latent_raw - batch_mean) / tf.sqrt(batch_std)
        latent_corrupted = self.corrupt(latent_normalized, latent_dim)
        latent_final = tf.multiply(layer_scaling, layer_bias + latent_corrupted)
        return latent_final, latent_corrupted, batch_mean, batch_std

    def climb_down(self, corrupted_inputs, layer_weights):
        projection = tf.matmul(corrupted_inputs, layer_weights)
        batch_mean, batch_std = tf.nn.moments(projection, axes=0)
        projection_normalized = (projection - batch_mean) / batch_std

    def corrupt(self, values, dim):
        return values + tf.random_normal(shape=[dim], mean=0, stddev=self.noise_std)

    @staticmethod
    def expressive_nonlinearity(u, a):
        return a[0] * tf.nn.sigmoid(a[1] * u + a[2]) + a[3] * u + a[4]


