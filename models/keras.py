# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.callbacks import Callback
from keras.optimizers import Adam, RMSprop

OPTIMIZERS = {
    'adam': Adam,
    'rmsprop': RMSprop
}


class KerasTestCallback(Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.test_performance = []
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

    def on_epoch_end(self, epoch, logs=None):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.test_performance.append("%.4f%%" % (score[1] * 100))
        print("Test %s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


class KerasModels:
    @staticmethod
    def get_keras_model(layers, embedding_options=None, activation='sigmoid', optimizer='rmsprop', learning_rate=0.001):
        model = Sequential()

        if embedding_options is not None:
            model.add(Embedding(**embedding_options))

        for layer in layers:
            model.add(layer)

        model.add(Dense(1, activation=activation))

        model.compile(
            optimizer=OPTIMIZERS[optimizer](learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model
