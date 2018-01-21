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
        self.decoder = LadderDecoder(dim_specs, self.clean_encoder.stored_values, self.clean_encoder.stored_means,
                                     self.clean_encoder.stored_stds)
        self.decoder.build_decoder(self.rung, self.corrupted_encoder.current)
        self.semi_supervised_loss = self.supervised_loss + self.decoder.unsupervised_loss

    def train_ladder_network(self):
        pass


class LadderEncoder(FeedForward):

    def __init__(self, dim_specs, activation, output_activation, noise_std):

        """
        :param noise_std: standard deviation of gaussian noise used for corrupting layers
        """
        super().__init__(dim_specs, activation, output_activation)

        self.noise_std = noise_std

        self.scaling_list = [tf.constant(1.0, shape=[1, self.dim_specs[i]]) for i in range(1, self.n_layers)]

        self.stored_values = []
        self.stored_means = []
        self.stored_stds = []

    def build_encoder(self, layer, output_layer):
        self.current = layer.corrupt(self.current, self.input_dim)
        for i in range(self.n_layers - 2):
            self.current = layer.climb_up(self.current, self.var_list[i], self.bias_list[i],
                                       self.scaling_list[i], self.dim_specs[i+1])
        self.current = output_layer.climb_up(self.current, self.var_list[-1], self.bias_list[-1],
                                          self.scaling_list[-1], self.dim_specs[-1])


class LadderDecoder:

    def __init__(self, dim_specs, skip_values, skip_means, skip_stds):
        self.dim_specs = dim_specs
        self.n_layers = len(dim_specs)
        self.skip_values = skip_values
        self.skip_means = skip_means
        self.skip_stds = skip_stds
        self.denoising_weights = [tf.Variable(tf.truncated_normal(shape=[dim_specs[i], 10]))
                                  for i in range(1, self.n_layers)]
        self.decoder_weights = [tf.Variable(tf.truncated_normal(shape=[dim_specs[self.n_layers - i - 1]], stddev=0.1))
                                for i in range(1, self.n_layers)]
        self.unsupervised_loss = None

    def build_decoder(self, layer, top):
        current_normal, current = layer.climb_down(top, self.dim_specs[0],
                                   self.skip_means[0], self.skip_stds[0], self.denoising_weights[0])
        self.unsupervised_loss = tf.reduce_sum(tf.square(current_normal - self.skip_values[0]), reduction_indices=1)
        for i in range(self.n_layers - 2):
            current_normal, current = layer.climb_down(top, self.dim_specs[i], self.skip_means[i], self.skip_stds[i],
                             self.denoising_weights[i], self.decoder_weights[i])
            self.unsupervised_loss += tf.reduce_sum(tf.square(current_normal - self.skip_values[i]),
                                                    reduction_indices=1)


class Layer:

    def __init__(self, activation):
        self.activation = activation

    def feed_forward(self, layer_inputs, layer_weight, layer_bias):
        return self.activation(tf.matmul(layer_inputs, layer_weight) + layer_bias)

    @staticmethod
    def normalize(values):
        batch_mean = tf.reduce_mean(values, reduction_indices=0)
        batch_std = tf.sqrt(tf.reduce_mean(tf.map_fn(lambda z: z - batch_mean, values), reduction_indices=0))
        return (values - batch_mean) / batch_std, batch_mean, batch_std


class LadderRung(Layer):

    def __init__(self, activation, noise_std):
        super().__init__(activation)
        self.noise_std = noise_std

    def climb_up(self, layer_inputs, layer_weights, layer_bias, layer_scaling, latent_dim):
        latent_raw = tf.matmul(layer_inputs, layer_weights)
        latent_normalized, batch_mean, batch_std = self.normalize(latent_raw)
        latent_corrupted = self.corrupt(latent_normalized, latent_dim)
        latent_final = tf.multiply(layer_scaling, layer_bias + latent_corrupted)
        return latent_final, latent_corrupted, batch_mean, batch_std

    def climb_down(self, corrupted_inputs, latent_dim, skip_mean, skip_std, denoising_weights, layer_weights=None):
        if layer_weights is None: projection = corrupted_inputs
        else: projection = tf.matmul(corrupted_inputs, layer_weights)
        projection_normalized, _, _ = self.normalize(projection)
        reconstruction = tf.constant(0, shape=[0, latent_dim])
        for i in range(latent_dim):
            denoising_func_1 = self.expressive_nonlinearity(projection_normalized[:, i], denoising_weights[i, :5])
            denoising_func_2 = self.expressive_nonlinearity(projection_normalized[:, i], denoising_weights[i, 5:])
            reconstruction_i = (corrupted_inputs[:, i] - denoising_func_1) * denoising_func_2 + denoising_func_1
            reconstruction = tf.concat([reconstruction, reconstruction_i], axis=1)
        return (reconstruction - skip_mean) / skip_std, reconstruction

    def corrupt(self, values, dim):
        return values + tf.random_normal(shape=[1, dim], mean=0, stddev=self.noise_std)

    @staticmethod
    def expressive_nonlinearity(u, a):
        return a[0] * tf.nn.sigmoid(a[1] * u + a[2]) + a[3] * u + a[4]


# Simple unit test for compiling ladder network

ladder = LadderNetwork([5, 6, 7, 8, 9, 10], tf.nn.relu, tf.nn.softmax, 0.3)

print(ladder.semi_supervised_loss)  # Should print out tensor object
