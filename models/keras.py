# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Embedding


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
