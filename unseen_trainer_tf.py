import tensorflow as tf
import numpy as np
import numpy.random as npr
import os
import scipy.io
import sys
import glob
from numpy.linalg import norm
from scipy import misc

from model_unseen import Model
from data_preprocessing import get_phone_data, get_watch_data

class Trainer:
    def __init__(self, flag):
        if flag == "phone":
            print("Loading phone data.")
            self.x_train, self.x_test, self.y_train, self.y_test = get_phone_data()
            self.x_train = self.x_train[:10000]
            self.y_train = self.y_train[:10000]
            self.x_test = self.x_test
            self.y_test = self.y_test
            print("Loaded phone data")
        elif flag == "watch":
            print("Loading watch data")
            self.x_train, self.x_test, self.y_train, self.y_test = get_watch_data()
            self.x_train = self.x_train[:10000]
            self.y_train = self.y_train[:10000]
            self.x_test = self.x_test
            self.y_test = self.y_test
            print("Loaded watch data")
        else:
            raise Exception("flag should be either 'phone' or 'watch'")

        self.model = Model()
        self.model.build_model()

        # training iterations
        self.train_iters = 10001
        # number of samples in each batch
        self.batch_size = 32
        # gamma
        self.gamma = 1.0
        # max iterations
        self.T_adv = 15
        # min iterations
        self.T_min = 100
        # number of adversarial phases
        self.k = 1

        self.log_dir = "./exp_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.model_save_path = "./model_params"
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def train(self):

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto()) as sess:
            # tf.compat.v1.global_variables_initializer().run()
            sess.run(tf.compat.v1.global_variables_initializer())
            # saver = tf.compat.v1.train.Saver(max_to_keep=0)

            summary_writer = tf.compat.v1.summary.FileWriter(logdir=self.log_dir, graph=tf.compat.v1.get_default_graph())

            counter_k = 0
            print("Start Training")
            for t in range(self.train_iters):
                if ((t + 1) % self.T_min == 0) and (counter_k < self.k):
                    print("Generating adversarial data [iter %d]" % counter_k)
                    for start, end in zip(range(0, self.x_train.shape[0], self.batch_size), range(self.batch_size, self.x_train.shape[0], self.batch_size)):
                        feed_dict = {self.model.z: self.x_train[start:end], self.model.labels: self.y_train[start:end]}

                        # assigning the current batch of images to the variable to learn z_hat
                        sess.run(self.model.z_hat_assign_op, feed_dict)
                        for n in range(self.T_adv):  # running T_adv gradient ascent steps
                            sess.run(self.model.max_train_op, feed_dict)

                        # tmp variable with the learned images
                        learnt_imgs_tmp = sess.run(self.model.z_hat, feed_dict)

                        # stacking the learned images and corresponding labels to the original dataset
                        self.x_train = np.vstack((self.x_train, learnt_imgs_tmp))
                        self.y_train = np.hstack((self.y_train, self.y_train[start:end]))

                        # shuffling the dataset
                    rnd_indices = list(range(len(self.x_train)))
                    npr.shuffle(rnd_indices)
                    self.x_train = self.x_train[rnd_indices]
                    self.y_train = self.y_train[rnd_indices]

                    counter_k += 1

                i = t % int(self.x_train.shape[0] / self.batch_size)

                # current batch of images and labels
                batch_z = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                batch_labels = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]

                feed_dict = {self.model.z: batch_z, self.model.labels: batch_labels}

                # running a step of gradient descent
                sess.run([self.model.min_train_op, self.model.min_loss], feed_dict)

                # evaluating the model
                if t % 250 == 0:
                    summary, min_l, max_l, acc = sess.run(
                        [self.model.summary_op, self.model.min_loss, self.model.max_loss, self.model.accuracy],
                        feed_dict)

                    train_rand_idxs = np.random.permutation(self.x_train.shape[0])[:100]
                    test_rand_idxs = np.random.permutation(self.x_test.shape[0])[:100]

                    train_acc, train_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss],
                                                         feed_dict={self.model.z: self.x_train[train_rand_idxs],
                                                                    self.model.labels: self.y_train[
                                                                        train_rand_idxs]})
                    test_acc, test_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss],
                                                       feed_dict={self.model.z: self.x_test[test_rand_idxs],
                                                                  self.model.labels: self.y_test[
                                                                      test_rand_idxs]})

                    summary_writer.add_summary(summary, t)
                    print(
                        'Step: [%d/%d] train_min_loss: [%.4f] train_acc: [%.4f] test_min_loss: [%.4f] test_acc: [%.4f]' % (
                        t + 1, self.train_iters, train_min_loss, train_acc, test_min_loss, test_acc))

            print('Saving')
            # saver.save(sess, os.path.join(self.model_save_path, 'encoder'))

