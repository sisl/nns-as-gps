import numpy as np
import tensorflow as tf
import pdb


class EnsembleMSE(object):

    def __init__(self,
        sess,
        obs_dim,
        output_dim,
        name,
        ensemble_size = 10,
        hidden_sizes = [64,32,32,32],
        learning_rate = 1e-3,
        batch_size = 4,
        validation_ratio = 0.0,
        max_epochs = 300,
        use_bn = False,
        dropout_rate = 0.0,
        weight_reg = 1e-3,
        ):

        self.sess = sess
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.ensemble_size = ensemble_size
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_ratio = validation_ratio
        self.max_epochs = max_epochs
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.weight_reg = weight_reg

        self.inputs = tf.placeholder(tf.float32, [None, obs_dim], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, output_dim], name='outputs')

        self.phase = tf.placeholder(tf.bool, name='phase')

        self.outputs = [None] * ensemble_size

        for en in range(ensemble_size):
            with tf.variable_scope(name+str(en)) as scope:
                hidden = tf.layers.dense(
                    self.inputs, hidden_sizes[0], 
                    name = 'h0', 
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer=tf.contrib.layers.xavier_initializer())


                if(use_bn):
                    hidden_bn = tf.contrib.layers.batch_norm(hidden, 
                                              center=True, scale=True, 
                                              is_training=self.phase)
                else:
                    hidden_bn = hidden

                hidden_relu = tf.nn.relu(hidden_bn)

                hidden_ = tf.layers.dropout(hidden_relu, rate = dropout_rate, training=self.phase)

                for i in range(1, len(hidden_sizes)):
                    hidden = tf.layers.dense(
                    hidden_, hidden_sizes[i], 
                    name = 'h%d' % i, 
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer=tf.contrib.layers.xavier_initializer())

                    if(use_bn):
                        hidden_bn = tf.contrib.layers.batch_norm(hidden, 
                                                  center=True, scale=True, 
                                                  is_training=self.phase)
                    else:
                        hidden_bn = hidden

                    hidden_relu = tf.nn.relu(hidden_bn)

                    hidden_ = tf.layers.dropout(hidden_relu, rate = dropout_rate, training=self.phase)


                self.outputs[en] = tf.layers.dense(hidden_, output_dim, name='output', 
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.losses = [tf.nn.l2_loss(output - self.labels) for output in self.outputs]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optims = [tf.train.AdamOptimizer(learning_rate=learning_rate, 
                name='adam_emse'+str(en)).minimize(
                loss) for en, loss in enumerate(self.losses)]
            # self.optims = [tf.train.GradientDescentOptimizer(learning_rate=learning_rate, 
            #     name='adam_emse'+str(en)).minimize(
            #     loss) for en, loss in enumerate(self.losses)]

        return

    def load_data(self, data):
        self.dataset_X = data[0]
        self.dataset_Y = data[1]


    def split_train_test(self):
        N = self.dataset_X.shape[0]

        if(self.validation_ratio > 0.0):

            Ntrain = int(N * (1-self.validation_ratio))
            Ntest = N - Ntrain

            shuffle_inds = np.arange(N)
            np.random.shuffle(shuffle_inds)
            self.train_data_X = self.dataset_X[shuffle_inds[:Ntrain], :]
            self.train_data_Y = self.dataset_Y[shuffle_inds[:Ntrain], :]
            self.test_data_X = self.dataset_X[shuffle_inds[Ntrain:], :]
            self.test_data_Y = self.dataset_Y[shuffle_inds[Ntrain:], :]
        else:
            self.train_data_X = self.dataset_X
            self.train_data_Y = self.dataset_Y
            self.test_data_X = None
            self.test_data_Y = None

    def train(self, model_name = None, 
        verbose=False):
        # if model_name == None then it will not save

        min_test_loss = np.Inf

        for epoch in range(self.max_epochs):
            Ntrain = self.train_data_X.shape[0]
            num_batches = Ntrain // self.batch_size


            for en in range(self.ensemble_size):
                # shuffle train data
                shuffle_inds = np.arange(Ntrain)
                np.random.shuffle(shuffle_inds)
                self.train_data_X = self.train_data_X[shuffle_inds,:]
                self.train_data_Y = self.train_data_Y[shuffle_inds,:]
                for idx in range(num_batches):
                    batch_ind_start = idx*self.batch_size
                    batch_ind_end = (idx+1)*self.batch_size
                    batch_X = self.train_data_X[batch_ind_start:batch_ind_end,:]
                    batch_Y = self.train_data_Y[batch_ind_start:batch_ind_end,:]

                    self.sess.run(self.optims[en], 
                        feed_dict = dict(zip([self.inputs, self.labels, self.phase], 
                                            [batch_X, batch_Y, True])))

                if verbose: print('Epoch %d Train Loss: %.3f' % (epoch, 
                    self.eval_performance_single(self.train_data_X, self.train_data_Y,en)))

                if(self.test_data_X is not None):
                    test_loss = self.eval_performance_single(self.test_data_X, self.test_data_Y,en)
                    if verbose: print('Epoch %d Test Loss: %.3f' % (epoch, test_loss))


    def eval_performance_single(self, x, labels, en):

        N = x.shape[0]

        return self.sess.run(self.losses[en], 
            feed_dict = dict(zip([self.inputs, self.labels, self.phase], 
                                [x, labels, False]))) / N


    def eval(self, x):
        means_en = self.sess.run(self.outputs, 
            feed_dict = dict(zip([self.inputs, self.phase], 
                                [x, False])))

        means_en = np.squeeze(np.array(means_en))
        means = np.mean(means_en,axis=0)
        sigsqs = np.std(means_en,axis=0) ** 2

        return means, sigsqs

    def get_output(self,x):
        x = np.expand_dims(x, axis=0)
        outputs = self.sess.run(self.outputs, 
            feed_dict = dict(zip([self.inputs, self.phase], 
                                [x, False])))

        outputs = np.array(outputs)

        # reshape to samples of each output dim
        outputs = outputs.reshape(-1, outputs.shape[-1]).T

        mean_outputs = np.mean(outputs,axis=1)
        cov_outputs = np.cov(outputs)

        if(self.output_dim == 1):
            cov_outputs = cov_outputs.reshape((1,1))

        return mean_outputs, dict(mean=mean_outputs,cov=cov_outputs)

class EnsembleNLL(object):
    def __init__(self,
        sess,
        obs_dim,
        output_dim,
        name,
        ensemble_size = 10,
        hidden_sizes = [64,32,32,32],
        learning_rate = 1e-3,
        batch_size = 4,
        validation_ratio = 0.0,
        max_epochs = 300,
        use_bn = False,
        dropout_rate = 0.0,
        weight_reg = 1e-3,
        ):

        self.sess = sess
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.ensemble_size = ensemble_size
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_ratio = validation_ratio
        self.max_epochs = max_epochs
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.weight_reg = weight_reg

        self.inputs = tf.placeholder(tf.float32, [None, obs_dim], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, output_dim], name='outputs')

        self.phase = tf.placeholder(tf.bool, name='phase')

        self.means = [None] * ensemble_size
        self.vars = [None] * ensemble_size

        self.losses = [None] * ensemble_size

        for en in range(ensemble_size):
            with tf.variable_scope(name+str(en)) as scope:
                hidden = tf.layers.dense(
                    self.inputs, hidden_sizes[0], 
                    name = 'h0', 
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer=tf.contrib.layers.xavier_initializer())


                if(use_bn):
                    hidden_bn = tf.contrib.layers.batch_norm(hidden, 
                                              center=True, scale=True, 
                                              is_training=self.phase)
                else:
                    hidden_bn = hidden

                hidden_relu = tf.nn.relu(hidden_bn)

                hidden_ = tf.layers.dropout(hidden_relu, rate = dropout_rate, training=self.phase)

                for i in range(1, len(hidden_sizes)-1):
                    hidden = tf.layers.dense(
                    hidden_, hidden_sizes[i], 
                    name = 'h%d' % i, 
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer=tf.contrib.layers.xavier_initializer())

                    if(use_bn):
                        hidden_bn = tf.contrib.layers.batch_norm(hidden, 
                                                  center=True, scale=True, 
                                                  is_training=self.phase)
                    else:
                        hidden_bn = hidden

                    hidden_relu = tf.nn.relu(hidden_bn)

                    hidden_ = tf.layers.dropout(hidden_relu, rate = dropout_rate, training=self.phase)
                i = len(hidden_sizes)-1
                hidden = tf.layers.dense(
                    hidden_, hidden_sizes[-1]*2, 
                    name = 'h%d' % i, 
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer=tf.contrib.layers.xavier_initializer())

                if(use_bn):
                    hidden_bn = tf.contrib.layers.batch_norm(hidden, 
                                              center=True, scale=True, 
                                              is_training=self.phase)
                else:
                    hidden_bn = hidden

                hidden_relu = tf.nn.relu(hidden_bn)

                hidden_ = tf.layers.dropout(hidden_relu, rate = dropout_rate, training=self.phase)

                mu_, rawvar_ = tf.split(hidden_, [hidden_sizes[-1], hidden_sizes[-1]], axis=1)

                mu = tf.layers.dense(mu_, output_dim, name='outputmu', 
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer=tf.contrib.layers.xavier_initializer())

                rawvar = tf.layers.dense(rawvar_, output_dim, name='outputrw', 
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
                sigsq = tf.nn.softplus(rawvar) + 1e-6 # enforce positivity and numerical stability

                self.means[en] = mu
                self.vars[en] = sigsq

                # Negative log likelihood loss
                self.losses[en] = 0.5*tf.reduce_mean( tf.log(sigsq) + (self.labels - mu)**2 / sigsq ) 

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optims = [tf.train.AdamOptimizer(learning_rate=learning_rate, 
                name='adam_enll'+str(en)).minimize(
                loss) for en, loss in enumerate(self.losses)]
            # self.optims = [tf.train.GradientDescentOptimizer(learning_rate=learning_rate, 
            #     name='sgd_enll'+str(en)).minimize(
            #     loss) for en, loss in enumerate(self.losses)]

        return

    def load_data(self, data):
        self.dataset_X = data[0]
        self.dataset_Y = data[1]


    def split_train_test(self):
        N = self.dataset_X.shape[0]

        if(self.validation_ratio > 0.0):

            Ntrain = int(N * (1-self.validation_ratio))
            Ntest = N - Ntrain

            shuffle_inds = np.arange(N)
            np.random.shuffle(shuffle_inds)
            self.train_data_X = self.dataset_X[shuffle_inds[:Ntrain], :]
            self.train_data_Y = self.dataset_Y[shuffle_inds[:Ntrain], :]
            self.test_data_X = self.dataset_X[shuffle_inds[Ntrain:], :]
            self.test_data_Y = self.dataset_Y[shuffle_inds[Ntrain:], :]
        else:
            self.train_data_X = self.dataset_X
            self.train_data_Y = self.dataset_Y
            self.test_data_X = None
            self.test_data_Y = None



    def train(self, model_name = None, 
        verbose=False):
        # if model_name == None then it will not save

        min_test_loss = np.Inf

        for epoch in range(self.max_epochs):
            Ntrain = self.train_data_X.shape[0]
            num_batches = Ntrain // self.batch_size


            for en in range(self.ensemble_size):
                # shuffle train data
                shuffle_inds = np.arange(Ntrain)
                np.random.shuffle(shuffle_inds)
                self.train_data_X = self.train_data_X[shuffle_inds,:]
                self.train_data_Y = self.train_data_Y[shuffle_inds,:]
                for idx in range(num_batches):
                    batch_ind_start = idx*self.batch_size
                    batch_ind_end = (idx+1)*self.batch_size
                    batch_X = self.train_data_X[batch_ind_start:batch_ind_end,:]
                    batch_Y = self.train_data_Y[batch_ind_start:batch_ind_end,:]

                    self.sess.run(self.optims[en], 
                        feed_dict = dict(zip([self.inputs, self.labels, self.phase], 
                                            [batch_X, batch_Y, True])))

                if verbose: print('Epoch %d Train Loss: %.3f' % (epoch, 
                    self.eval_performance_single(self.train_data_X, self.train_data_Y,en)))

                if(self.test_data_X is not None):
                    test_loss = self.eval_performance_single(self.test_data_X, self.test_data_Y,en)
                    if verbose: print('Epoch %d Test Loss: %.3f' % (epoch, test_loss))


    def eval_performance_single(self, x, labels, en):

        N = x.shape[0]

        return self.sess.run(self.losses[en], 
            feed_dict = dict(zip([self.inputs, self.labels, self.phase], 
                                [x, labels, False]))) / N

    def eval(self, x):
        means_en, sigsqs_en = self.sess.run([self.means, self.vars], 
            feed_dict = dict(zip([self.inputs, self.phase], 
                                [x, False])))

        means_en = np.squeeze(np.array(means_en))
        sigsqs_en = np.squeeze(np.array(sigsqs_en))

        means = np.mean(means_en,axis=0)
        sigsqs = np.mean(sigsqs_en + means_en**2,axis=0) - means**2

        return means, sigsqs

    def get_output(self,x):
        x = np.expand_dims(x, axis=0)
        means, sigsqs = self.eval(x)

        means = np.array(means)
        sigsqs = np.array(sigsqs)

        # reshape to samples of each output dim
        means = means.reshape(-1, means.shape[-1]).T
        sigsqs = means.reshape(-1, sigsqs.shape[-1]).T

        mean_outputs = np.mean(means,axis=1)
        cov_outputs = np.mean(sigsqs + means**2,axis=1) - mean_outputs**2

        if(self.output_dim == 1):
            cov_outputs = cov_outputs.reshape((1,1))


        return mean_outputs, dict(mean=mean_outputs,cov=cov_outputs)

class MCDropoutMSE(object):

    def __init__(self,
        sess,
        obs_dim,
        output_dim,
        name,
        hidden_sizes = [64,32,32,32],
        learning_rate = 1e-3,
        batch_size = 4,
        validation_ratio = 0.0,
        max_epochs = 300,
        use_bn = False,
        dropout_rate = 0.0,
        weight_reg = 1e-3,
        ):

        self.sess = sess
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_ratio = validation_ratio
        self.max_epochs = max_epochs
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.weight_reg = weight_reg

        self.inputs = tf.placeholder(tf.float32, [None, obs_dim], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, output_dim], name='outputs')

        self.phase = tf.placeholder(tf.bool, name='phase')

        self.output = None

        with tf.variable_scope(name) as scope:
            hidden = tf.layers.dense(
                self.inputs, hidden_sizes[0], 
                name = 'h0', 
                kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                kernel_initializer=tf.contrib.layers.xavier_initializer())


            if(use_bn):
                hidden_bn = tf.contrib.layers.batch_norm(hidden, 
                                          center=True, scale=True, 
                                          is_training=self.phase)
            else:
                hidden_bn = hidden

            hidden_relu = tf.nn.relu(hidden_bn)

            hidden_ = tf.layers.dropout(hidden_relu, rate = dropout_rate, training=True)

            for i in range(1, len(hidden_sizes)):
                hidden = tf.layers.dense(
                hidden_, hidden_sizes[i], 
                name = 'h%d' % i, 
                kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                kernel_initializer=tf.contrib.layers.xavier_initializer())

                if(use_bn):
                    hidden_bn = tf.contrib.layers.batch_norm(hidden, 
                                              center=True, scale=True, 
                                              is_training=self.phase)
                else:
                    hidden_bn = hidden

                hidden_relu = tf.nn.relu(hidden_bn)

                hidden_ = tf.layers.dropout(hidden_relu, rate = dropout_rate, training=True)


            self.output = tf.layers.dense(hidden_, output_dim, name='output', 
                kernel_regularizer=tf.contrib.layers.l1_regularizer(weight_reg),
                kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.output - self.labels))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optim = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_dmse').minimize(
                self.loss)
            # self.optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='sgd_dmse').minimize(
            #     self.loss)

        self.saver = tf.train.Saver()

        return

    def load_data(self, data):
        self.dataset_X = data[0]
        self.dataset_Y = data[1]


    def split_train_test(self):
        N = self.dataset_X.shape[0]

        if(self.validation_ratio > 0.0):

            Ntrain = int(N * (1-self.validation_ratio))
            Ntest = N - Ntrain

            shuffle_inds = np.arange(N)
            np.random.shuffle(shuffle_inds)
            self.train_data_X = self.dataset_X[shuffle_inds[:Ntrain], :]
            self.train_data_Y = self.dataset_Y[shuffle_inds[:Ntrain], :]
            self.test_data_X = self.dataset_X[shuffle_inds[Ntrain:], :]
            self.test_data_Y = self.dataset_Y[shuffle_inds[Ntrain:], :]
        else:
            self.train_data_X = self.dataset_X
            self.train_data_Y = self.dataset_Y
            self.test_data_X = None
            self.test_data_Y = None



    def train(self, model_name = None, 
        verbose=False):
        # if model_name == None then it will not save

        min_test_loss = np.Inf

        for epoch in range(self.max_epochs):
            Ntrain = self.train_data_X.shape[0]
            num_batches = Ntrain // self.batch_size


            shuffle_inds = np.arange(Ntrain)
            np.random.shuffle(shuffle_inds)
            self.train_data_X = self.train_data_X[shuffle_inds,:]
            self.train_data_Y = self.train_data_Y[shuffle_inds,:]
            for idx in range(num_batches):
                batch_ind_start = idx*self.batch_size
                batch_ind_end = (idx+1)*self.batch_size
                batch_X = self.train_data_X[batch_ind_start:batch_ind_end,:]
                batch_Y = self.train_data_Y[batch_ind_start:batch_ind_end,:]

                self.sess.run(self.optim, 
                    feed_dict = dict(zip([self.inputs, self.labels, self.phase], 
                                        [batch_X, batch_Y, True])))

            if verbose: print('Epoch %d Train Loss: %.3f' % (epoch, 
                self.eval_performance_single(self.train_data_X, self.train_data_Y)))

            if(self.test_data_X is not None):
                test_loss = self.eval_performance_single(self.test_data_X, self.test_data_Y)
                if verbose: print('Epoch %d Test Loss: %.3f' % (epoch, test_loss))



    def eval_performance_single(self, x, labels):

        N = x.shape[0]

        return self.sess.run(self.loss, 
            feed_dict = dict(zip([self.inputs, self.labels, self.phase], 
                                [x, labels, False]))) / N

    def eval(self, x):
        samps = self.sess.run(self.output, 
            feed_dict = dict(zip([self.inputs, self.phase], 
                                [np.vstack([x]*100), False])))
        samps = samps.reshape([100]+list(x.shape))
        samps = np.squeeze(np.array(samps))

        means = np.mean(samps,axis=0)
        sigsqs = np.std(samps,axis=0) ** 2

        return means, sigsqs

    def get_output(self,x):
        x = np.expand_dims(x, axis=0)
        outputs = self.sess.run(self.output, 
            feed_dict = dict(zip([self.inputs, self.phase], 
                                [np.array([x]*10), False])))

        outputs = np.array(outputs)

        # reshape to samples of each output dim
        outputs = outputs.reshape(-1, outputs.shape[-1]).T

        mean_outputs = np.mean(outputs,axis=1)
        cov_outputs = np.cov(outputs)

        if(self.output_dim == 1):
            cov_outputs = cov_outputs.reshape((1,1))


        return mean_outputs, dict(mean=mean_outputs,cov=cov_outputs)
