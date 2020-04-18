# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Dense, Input, Embedding
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, Sequential
import matplotlib.pyplot as plt
import tensorflow as tf


class TestDataCallback(Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.accuracy = []
        self.loss = []
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.loss.append(score[0])
        self.accuracy.append(score[1])
        print(f"test_{self.model.metrics_names[0]}: {score[0]:.4f}, test_{self.model.metrics_names[1]}: {score[1]:.4f}")


class Keras:
    OPTIMIZERS = {
        'adam': Adam,
        'rmsprop': RMSprop
    }

    @staticmethod
    def get_model(layers, embedding_options=None, activation='sigmoid', optimizer='rmsprop', learning_rate=0.001):
        model = Sequential()

        if embedding_options is not None:
            model.add(Embedding(**embedding_options))

        for layer in layers:
            model.add(layer)

        model.add(Dense(1, activation=activation))

        model.compile(
            optimizer=Keras.OPTIMIZERS[optimizer](learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def _plot(history, string):
        legend = []
        val_str = f'val_{string}'
        test_str = f'test_{string}'

        if history.get(string):
            plt.plot(history[string])
            legend.append(string)
        if history.get(val_str):
            plt.plot(history[val_str])
            legend.append(val_str)
        if history.get(test_str):
            plt.plot(history[test_str])

        plt.xlabel('Epochs')
        plt.legend(legend)
        plt.show()

    @staticmethod
    def draw_graph(history):
        Keras._plot(history, 'accuracy')
        Keras._plot(history, 'loss')

    @staticmethod
    def get_bert_model(bert_layer, input_length, optimizer='rmsprop', learning_rate=2e-6):
        input_word_ids = Input(shape=(input_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(input_length,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(input_length,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)

        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        model.compile(
            optimizer=Keras.OPTIMIZERS[optimizer](learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model
