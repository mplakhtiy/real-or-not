# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM


class KerasModels:
    @staticmethod
    def get_keras_model(layers, embedding_options=None, activation='sigmoid', optimizer='rmsprop'):  # optimizer='adam'
        model = Sequential()

        if embedding_options is not None:
            model.add(Embedding(**embedding_options))

        for layer in layers:
            model.add(layer)

        model.add(Dense(1, activation=activation))

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def get_binary_classification_model(embedding_options):
        return KerasModels.get_keras_model([Flatten()], embedding_options=embedding_options)

    @staticmethod
    def get_mlp_for_binary_classification_model(embedding_options, dense_units=None):
        if dense_units is None:
            dense_units = []

        layers = [Flatten()]

        for index, units in enumerate(dense_units):
            layers.append(Dense(units, activation='relu'))

        return KerasModels.get_keras_model(layers, embedding_options=embedding_options)

    @staticmethod
    def get_lstm_model(embedding_options, lstm_units):
        return KerasModels.get_keras_model([LSTM(lstm_units)], embedding_options=embedding_options)
