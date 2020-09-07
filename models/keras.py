# -*- coding: utf-8 -*-
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D, Dropout, GlobalMaxPooling1D, GRU, Input, Embedding
from tensorflow.keras.layers import Bidirectional, Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPool1D
import matplotlib.pyplot as plt
import tensorflow as tf


class TestDataCallback(Callback):
    def __init__(self, x_test, y_test, is_history=True, is_predictions=False):
        super().__init__()
        self.accuracy = []
        self.loss = []
        self.predictions = []
        self._is_history = is_history
        self._is_predictions = is_predictions
        self.x_test = x_test
        self.y_test = y_test

    @staticmethod
    def _flatten_predictions(predictions):
        return [round(float(prediction[0]), 6) for prediction in predictions]

    def on_epoch_end(self, epoch, logs=None):
        if self._is_history:
            score = self.model.evaluate(self.x_test, self.y_test, verbose=1)
            self.loss.append(score[0])
            self.accuracy.append(score[1])
        if self._is_predictions:
            self.predictions.append(
                TestDataCallback._flatten_predictions(self.model.predict(self.x_test).tolist())
            )


class Keras:
    OPTIMIZERS = {
        'adam': Adam,
        'rmsprop': RMSprop
    }

    DEFAULTS = {
        'ACTIVATION': 'sigmoid',
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
    }

    @staticmethod
    def get_sequential_model(layers, config):
        if layers is None or config is None:
            raise ValueError('Layers and config can not be None!')

        model = Sequential()

        if config.get('EMBEDDING_OPTIONS') is not None:
            model.add(Embedding(**config['EMBEDDING_OPTIONS']))

        for layer in layers:
            model.add(layer)

        model.add(Dense(
            1,
            activation=config.get('ACTIVATION', Keras.DEFAULTS['ACTIVATION'])
        ))

        model.compile(
            optimizer=Keras.OPTIMIZERS[
                config.get('OPTIMIZER', Keras.DEFAULTS['OPTIMIZER'])
            ](
                learning_rate=config.get(
                    'LEARNING_RATE',
                    Keras.DEFAULTS['LEARNING_RATE']
                )
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def _get_lstm_model(config):
        return Keras.get_sequential_model(
            [LSTM(config['LSTM_UNITS'])],
            config,
        )

    @staticmethod
    def _get_lstm_dropout_model(config):
        return Keras.get_sequential_model(
            [
                SpatialDropout1D(config['DROPOUT']),
                LSTM(config['LSTM_UNITS'], dropout=config['DROPOUT'], recurrent_dropout=config['DROPOUT'])
            ],
            config
        )

    @staticmethod
    def _get_bi_lstm_model(config):
        layers = [Bidirectional(LSTM(config['LSTM_UNITS']))]

        if config.get('DROPOUT'):
            layers.append(Dropout(config['DROPOUT']))

        return Keras.get_sequential_model(
            layers,
            config
        )

    @staticmethod
    def _get_lstm_cnn_model(config):
        layers = [
            Conv1D(filters=config['CONV_FILTERS'], kernel_size=config['CONV_KERNEL_SIZE'], activation='relu'),
            MaxPooling1D(),
            LSTM(config['LSTM_UNITS'])
        ]

        if config.get('DROPOUT') is not None:
            layers = [Dropout(config['DROPOUT'])] + layers

        return Keras.get_sequential_model(
            layers,
            config
        )

    @staticmethod
    def _get_fast_text_model(config):
        return Keras.get_sequential_model(
            [GlobalAveragePooling1D(), Dense(config['DENSE_UNITS'], activation='relu')],
            config
        )

    @staticmethod
    def _get_rcnn_model(config):
        layers = [
            Conv1D(filters=config['CONV_FILTERS'], kernel_size=config['CONV_KERNEL_SIZE'], activation='relu'),
            MaxPooling1D(pool_size=config['MAX_POOLING_POOL_SIZE']),
            LSTM(config['LSTM_UNITS']),
        ]

        if config.get('DROPOUT') is not None:
            layers = [Dropout(config['DROPOUT'])] + layers

        return Keras.get_sequential_model(
            layers,
            config
        )

    @staticmethod
    def _get_cnn_model(config):
        layers = [
            Conv1D(filters=config['CONV_FILTERS'], kernel_size=config['CONV_KERNEL_SIZE'], activation='relu'),
            GlobalMaxPooling1D(),
            Dense(config['DENSE_UNITS'], activation='relu'),
        ]

        if config.get('DROPOUT') is not None:
            layers = [Dropout(config['DROPOUT'])] + layers

        return Keras.get_sequential_model(
            layers,
            config
        )

    @staticmethod
    def _get_rnn_model(config):
        return Keras.get_sequential_model(
            [
                Bidirectional(LSTM(config['LSTM_UNITS'])),
                Dense(config['DENSE_UNITS'], activation='relu')
            ],
            config
        )

    @staticmethod
    def _get_gru_model(config):
        layers = [
            Bidirectional(GRU(config['GRU_UNITS'], return_sequences=True)),
            GlobalMaxPool1D(),
            Dense(config['DENSE_UNITS'], activation='relu'),
        ]

        if config.get('DROPOUT') is not None:
            layers = layers + [Dropout(config['DROPOUT'])]

        return Keras.get_sequential_model(
            layers,
            config
        )

    @staticmethod
    def get_model(config):
        models = {
            'LSTM': Keras._get_lstm_model,
            'LSTM_DROPOUT': Keras._get_lstm_dropout_model,
            'BI_LSTM': Keras._get_bi_lstm_model,
            'LSTM_CNN': Keras._get_lstm_cnn_model,
            'FASTTEXT': Keras._get_fast_text_model,
            'RCNN': Keras._get_rcnn_model,
            'CNN': Keras._get_cnn_model,
            'RNN': Keras._get_rnn_model,
            'GRU': Keras._get_gru_model,
        }

        return models[config['TYPE']](config)

    @staticmethod
    def get_bert_model(
            bert_layer,
            input_length,
            optimizer='rmsprop',
            learning_rate=2e-6
    ):
        input_word_ids = Input(
            shape=(input_length,), dtype=tf.int32, name="input_word_ids"
        )
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

    @staticmethod
    def _plot(history, string):
        legend = []
        val_str = f'val_{string}'
        test_str = f'test_{string}'

        if history.get(string):
            plt.plot(history[string], color='#1f77b4')
            legend.append('train')
        if history.get(val_str):
            plt.plot(history[val_str], color='#ff7f0e')
            legend.append('validation')
        if history.get(test_str):
            plt.plot(history[test_str], color='#2ca02c')
            legend.append('test')

        plt.xlabel('Epochs')
        plt.legend(legend)

    @staticmethod
    def draw_graph(history):
        Keras._plot(history, 'accuracy')
        Keras._plot(history, 'loss')
        plt.show()

    @staticmethod
    def fit(model, data, config):
        is_with_test_data = len(data) == 6

        if is_with_test_data:
            x_train, y_train, x_val, y_val, x_test, y_test = data
        else:
            x_train, y_train, x_val, y_val = data

        callbacks = []

        if is_with_test_data:
            test_data_callback = TestDataCallback(
                x_test=x_test,
                y_test=y_test,
                is_history=False,
                is_predictions=True,
            )
            callbacks.append(test_data_callback)

        if config.get('DIR') is not None and config.get('PREFIX') is not None:
            suffix = '-e{epoch:03d}-a{accuracy:03f}-va{val_accuracy:03f}-ta.h5'
            callbacks.append(ModelCheckpoint(
                config['DIR'] + config['PREFIX'] + suffix,
                verbose=1,
                monitor='val_loss',
                save_best_only=True,
                mode='auto'
            ))

        history = model.fit(
            x=x_train, y=y_train,
            batch_size=config['BATCH_SIZE'],
            epochs=config['EPOCHS'],
            verbose=1,
            validation_data=(
                x_val,
                y_val
            ),
            callbacks=callbacks
        )

        model_history = history.history.copy()

        if is_with_test_data:
            # model_history['test_loss'] = test_data_callback.loss
            # model_history['test_accuracy'] = test_data_callback.accuracy
            pass

        model_history = {
            k: [round(float(v), 6) for v in data] for k, data in model_history.items()
        }

        if is_with_test_data:
            model_history['val_predictions'] = test_data_callback.predictions

        return model_history
