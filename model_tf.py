# import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
import os
import csv
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from data_preprocessing import get_phone_data, get_watch_data


class Model:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu', input_shape=self.x_train[0].shape))
        self.model.add(tf.keras.layers.Dropout(0.1))

        self.model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(18, activation='softmax'))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=15, validation_data=(self.x_test, self.y_test),
                                 verbose=1)
        return history

    # Plot learning Curve of traning and validation values

    @classmethod
    def plot_learningCurve(cls, history, epochs):
        # Accuracy values
        epoch_range = range(1, epochs + 1)
        plt.plot(epoch_range, history.history['accuracy'])
        plt.plot(epoch_range, history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        # loss values
        plt.plot(epoch_range, history.history['loss'])
        plt.plot(epoch_range, history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    # '''
    x_train, x_test, y_train, y_test = get_phone_data()
    phone_model = Model(x_train=x_train[:10000], x_test=x_test, y_train=y_train[:10000], y_test=y_test)
    phone_history = phone_model.train()
    Model.plot_learningCurve(phone_history, epochs=15)
    # '''

    '''
    x_train, x_test, y_train, y_test = get_watch_data()
    watch_model = Model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    watch_history = watch_model.train()
    Model.plot_learningCurve(watch_history, epochs=15)
    '''
