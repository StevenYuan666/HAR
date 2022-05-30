import tensorflow as tf
import numpy as np
import numpy.random as npr
import os
import scipy.io
import sys
import glob
from numpy.linalg import norm
from scipy import misc
import tf_slim as slim

from model_unseen import Model
from data_preprocessing import get_phone_data, get_watch_data

class Trainer:
    def __init__(self, flag, k=1):
        self.flag = flag
        if flag == "phone":
            print("Loading phone data.")
            self.x_train, self.x_validation, self.x_test, self.y_train, self.y_validation, self.y_test = get_phone_data()
            self.x_train = self.x_train
            self.y_train = self.y_train
            self.x_validation = self.x_validation
            self.y_validation = self.y_validation
            print("Loaded phone data")
        elif flag == "watch":
            print("Loading watch data")
            self.x_train, self.x_validation, self.x_test, self.y_train, self.y_validation, self.y_test = get_watch_data()
            self.x_train = self.x_train
            self.y_train = self.y_train
            self.x_validation = self.x_validation
            self.y_validation = self.y_validation
            print("Loaded watch data")
        else:
            raise Exception("flag should be either 'phone' or 'watch'")

        # training iterations
        # self.train_iters = self.x_train.shape[0] + 1
        self.train_iters = 2000 # 做一个early stop
        # number of samples in each batch
        self.batch_size = 32
        # gamma
        self.gamma = 1.0
        # max iterations
        self.T_adv = 15
        # min iterations
        self.T_min = 100
        # number of adversarial phases
        self.k = k

        self.log_dir = "./exp_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.model_save_path = "./model_params_earlystop"
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def train(self):

        print("Building Model")
        self.model = Model(iterations=self.x_train.shape[0] + 1)
        self.model.build_model()
        print("Built")

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto()) as sess:
            # tf.compat.v1.global_variables_initializer().run()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()

            summary_writer = tf.compat.v1.summary.FileWriter(logdir=self.log_dir, graph=tf.compat.v1.get_default_graph())

            counter_k = 0
            print("Start Training")
            for t in range(self.train_iters):
                if ((t + 1) % self.T_min == 0) and (counter_k < self.k):
                    print("Generating adversarial data [iter %d]" % counter_k)
                    for start, end in zip(range(0, self.x_train.shape[0], self.batch_size), range(self.batch_size, self.x_train.shape[0], self.batch_size)):
                        feed_dict = {self.model.z: self.x_train[start:end], self.model.labels: self.y_train[start:end]}

                        # assigning the current batch of data to the variable to learn z_hat
                        sess.run(self.model.z_hat_assign_op, feed_dict)
                        for n in range(self.T_adv):  # running T_adv gradient ascent steps
                            sess.run(self.model.max_train_op, feed_dict)

                        # tmp variable with the learned data
                        learnt_data_tmp = sess.run(self.model.z_hat, feed_dict)

                        # stacking the learned data and corresponding labels to the original dataset
                        self.x_train = np.vstack((self.x_train, learnt_data_tmp))
                        self.y_train = np.hstack((self.y_train, self.y_train[start:end]))

                        # shuffling the dataset
                    rnd_indices = list(range(len(self.x_train)))
                    npr.shuffle(rnd_indices)
                    self.x_train = self.x_train[rnd_indices]
                    self.y_train = self.y_train[rnd_indices]

                    counter_k += 1

                i = t % int(self.x_train.shape[0] / self.batch_size)

                # current batch of data and labels
                batch_z = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                batch_labels = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]

                feed_dict = {self.model.z: batch_z, self.model.labels: batch_labels}

                # running a step of gradient descent
                sess.run([self.model.min_train_op, self.model.min_loss], feed_dict)

                # evaluating the model
                if t % 100 == 0:
                    summary, min_l, max_l, acc = sess.run(
                        [self.model.summary_op, self.model.min_loss, self.model.max_loss, self.model.accuracy],
                        feed_dict)

                    train_rand_idxs = np.random.permutation(self.x_train.shape[0])[:100]
                    test_rand_idxs = np.random.permutation(self.x_validation.shape[0])[:100]

                    train_acc, train_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss],
                                                         feed_dict={self.model.z: self.x_train[train_rand_idxs],
                                                                    self.model.labels: self.y_train[
                                                                        train_rand_idxs]})
                    test_acc, test_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss],
                                                       feed_dict={self.model.z: self.x_validation[test_rand_idxs],
                                                                  self.model.labels: self.y_validation[
                                                                      test_rand_idxs]})

                    summary_writer.add_summary(summary, t)
                    print(
                        'Step: [%d/%d] train_min_loss: [%.4f] train_acc: [%.4f] test_min_loss: [%.4f] test_acc: [%.4f]' % (
                        t + 1, self.train_iters, train_min_loss, train_acc, test_min_loss, test_acc))

            print('Saving')
            saver.save(sess, os.path.join(self.model_save_path, self.flag + '_k' + str(self.k), self.flag + '_encoder_' + str(self.k)))

    def test(self):
        # build a graph
        # print('Building model')
        # self.model.build_model()
        # print('Built')

        with tf.compat.v1.Session() as sess:
            tf.compat.v1.global_variables_initializer().run()

            print('Loading pre-trained model.')
            variables_to_restore = slim.get_model_variables(scope='encoder')
            restorer = tf.compat.v1.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(self.model_save_path, self.flag + '_k' + str(self.k), self.flag + '_encoder_' + str(self.k)))

            N = 100  # set accordingly to GPU memory
            target_accuracy = 0
            target_loss = 0

            print('Calculating accuracy')

            for test_data_batch, test_labels_batch in zip(np.array_split(self.x_test, N),
                                                            np.array_split(self.y_test, N)):
                feed_dict = {self.model.z: test_data_batch, self.model.labels: test_labels_batch}
                target_accuracy_tmp, target_loss_tmp = sess.run([self.model.accuracy, self.model.min_loss], feed_dict)
                target_accuracy += target_accuracy_tmp / float(N)
                target_loss += target_loss_tmp / float(N)

        print('Target accuracy: [%.4f] target loss: [%.4f]' % (target_accuracy, target_loss))
