import tensorflow as tf
import tf_slim as slim


def lrelu(input, leak=0.2, scope='lrelu'):
    return tf.compat.v1.maximum(input, leak * input)


class Model(object):
    """Tensorflow model
    """

    def __init__(self, iterations, mode='train'):
        self.no_classes = 18
        self.x_size = 80
        self.y_size = 6
        self.no_channels = 1

        # training iterations
        self.train_iters = iterations
        # number of samples in each batch
        self.batch_size = 16 # 32
        # learning rate min
        self.learning_rate_min = 0.0001
        # learning rate max
        self.learning_rate_max = 1.0
        # gamma
        self.gamma = 1.0

    '''
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.Conv2D(64,(2, 2), activation='relu', input_shape=self.x_train[0].shape))
    self.model.add(tf.keras.layers.Dropout(0.1))

    self.model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu'))
    self.model.add(tf.keras.layers.Dropout(0.2))

    self.model.add(tf.keras.layers.Flatten())

    self.model.add(tf.keras.layers.Dense(256, activation='relu'))
    self.model.add(tf.keras.layers.Dropout(0.5))

    self.model.add(tf.keras.layers.Dense(18, activation='softmax'))
    '''

    def encoder(self, data, reuse=False, return_feat=False):
        with tf.compat.v1.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.compat.v1.nn.relu):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.compat.v1.nn.relu, padding='VALID'):
                    net = slim.conv2d(data, 64, (2, 2), scope='conv1')
                    # net = slim.dropout(net, keep_prob=0.9)
                    net = slim.conv2d(net, 128, (2, 2), scope='conv2')
                    # net = slim.dropout(net, keep_prob=0.8)
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 512, scope='fc1')
                    net = slim.fully_connected(net, 512, scope='fc2')
                    # net = slim.dropout(net, keep_prob=0.5)
                    if return_feat:
                        return net
                    net = slim.fully_connected(net, self.no_classes, activation_fn=None, scope='fco')
                    return net

    def build_model(self):
        tf.compat.v1.disable_eager_execution()
        # images placeholder
        self.z = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.x_size, self.y_size, self.no_channels], 'z')
        # labels placeholder
        self.labels = tf.compat.v1.placeholder(tf.compat.v1.int64, [None], 'labels')

        # images-for-gradient-ascent variable
        self.z_hat = tf.compat.v1.get_variable('z_hat', [self.batch_size, self.x_size, self.y_size, self.no_channels])
        # op to assign the value fed to self.z to the variable
        self.z_hat_assign_op = self.z_hat.assign(self.z)

        self.logits = self.encoder(self.z)
        self.logits_hat = self.encoder(self.z_hat, reuse=True)

        # for evaluation
        self.pred = tf.compat.v1.argmax(self.logits, 1)
        self.correct_pred = tf.compat.v1.equal(self.pred, self.labels)
        self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(self.correct_pred, tf.compat.v1.float32))

        # variables for the minimizer are the net weights, variables for the maximizer are the data's pixels
        t_vars = tf.compat.v1.trainable_variables()
        min_vars = [var for var in t_vars if 'z_hat' not in var.name]
        max_vars = [var for var in t_vars if 'z_hat' in var.name]

        # loss for the minimizer
        self.min_loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)

        # first term of the loss for the maximizer (== loss for the minimizer)
        self.max_loss_1 = slim.losses.sparse_softmax_cross_entropy(self.logits_hat, self.labels)

        # second term of the loss for the maximizer
        self.max_loss_2 = slim.losses.mean_squared_error(self.encoder(self.z, reuse=True, return_feat=True),
                                                         self.encoder(self.z_hat, reuse=True, return_feat=True))

        # final loss for the maximizer
        self.max_loss = self.max_loss_1 - self.gamma * self.max_loss_2

        # we use Adam for the minimizer and vanilla gradient ascent for the maximizer
        self.min_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate_min)
        self.max_optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate_max)

        # minimizer
        self.min_train_op = slim.learning.create_train_op(self.min_loss, self.min_optimizer,
                                                          variables_to_train=min_vars)
        # maximizer (-)
        self.max_train_op = slim.learning.create_train_op(-self.max_loss, self.max_optimizer,
                                                          variables_to_train=max_vars)

        min_loss_summary = tf.compat.v1.summary.scalar('min_loss', self.min_loss)
        max_loss_summary = tf.compat.v1.summary.scalar('max_loss', self.max_loss)

        accuracy_summary = tf.compat.v1.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.compat.v1.summary.merge([min_loss_summary, max_loss_summary, accuracy_summary])
