# -*- coding: utf-8 -*-
import numpy as np
from tweets import TweetsVectorization, tweets_preprocessor
from models import KerasModels, KerasTestCallback
from utils import draw_keras_graph, log
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Embedding, LSTM
from keras.layers import Bidirectional, GRU, Conv1D, GlobalAveragePooling1D, Dropout, MaxPool1D
from data import train_data as data, test_data_with_target

########################################################################################################################

VOCABULARY = {
    'WORDS_REPUTATION_FILTER': 0,
    'SORT_VOCABULARY': False,
    'REVERSE_SORT': False,
    'TWEETS_FOR_VOCABULARY_BASE': None,
    'ADD_START_SYMBOL': False
}

DATA = {
    'TRAIN_PERCENTAGE': 0.8,
    'SHUFFLE_DATA': True,
    'PREPROCESS_OPTRIONS': {
        'add_link_flag': True,
        'add_user_flag': True,
        'add_hash_flag': True,
        'add_number_flag': True,
        'remove_links': True,
        'remove_users': True,
        'remove_hash': True,
        'unslang': True,
        'split_words': True,
        'stem': True,
        'remove_punctuations': True,
        'remove_numbers': True,
        'to_lower_case': True,
        'remove_stop_words': True,
        'remove_not_alpha': True,
        'join': False
    }
}

########################################################################################################################

data['preprocessed'] = tweets_preprocessor.preprocess(data.text, DATA['PREPROCESS_OPTRIONS'])
test_data_with_target['preprocessed'] = tweets_preprocessor.preprocess(
    test_data_with_target.text,
    DATA['PREPROCESS_OPTRIONS']
)

vocabulary = TweetsVectorization.get_vocabulary(
    tweets=data.preprocessed[data.target == 1] if VOCABULARY['TWEETS_FOR_VOCABULARY_BASE'] is True else
    (data.preprocessed[data.target == 0] if VOCABULARY['TWEETS_FOR_VOCABULARY_BASE'] is False else data.preprocessed),
    vocabulary_filter=VOCABULARY['WORDS_REPUTATION_FILTER'],
    sort=VOCABULARY['SORT_VOCABULARY'],
    reverse=VOCABULARY['REVERSE_SORT'],
    add_start_symbol=VOCABULARY['ADD_START_SYMBOL']
)

x, y = TweetsVectorization.get_prepared_data_based_on_vocabulary_indexes(
    tweets=data.preprocessed,
    target=data.target.values,
    vocabulary=vocabulary,
)

x_train, y_train, x_val, y_val = TweetsVectorization.get_train_test_split(
    x=x, y=y, train_percentage=DATA['TRAIN_PERCENTAGE'], shuffle_data=DATA['SHUFFLE_DATA']
)

x_test, y_test = TweetsVectorization.get_prepared_data_based_on_vocabulary_indexes(
    tweets=test_data_with_target.preprocessed,
    target=test_data_with_target.target.values,
    vocabulary=vocabulary,
)

########################################################################################################################

MODEL = {
    'BATCH_SIZE': 512,
    'EPOCHS': 30,
    'VERBOSE': 1,
    'OPTIMIZER': 'rmsprop',
    'SHUFFLE': True,
    'EMBEDDING_OPTIONS': {
        'input_dim': len(vocabulary),
        'output_dim': 16,
        'input_length': len(x_train[0])
    }
}

########################################################################################################################

x_train = TweetsVectorization.to_same_length(x_train, MODEL['EMBEDDING_OPTIONS']['input_length'])
x_val = TweetsVectorization.to_same_length(x_val, MODEL['EMBEDDING_OPTIONS']['input_length'])
x_test = TweetsVectorization.to_same_length(x_test, MODEL['EMBEDDING_OPTIONS']['input_length'])

########################################################################################################################

'''Binary classification model'''
LAYERS = [Flatten()]

'''Multi layer binary classification model'''
# DENSE_LAYERS_UNITS = [16]
# LAYERS = [Flatten()] + [Dense(unit, activation='relu') for unit in DENSE_LAYERS_UNITS]

'''LSTM model'''
# LSTM_UNITS = 256
# LAYERS = [LSTM(LSTM_UNITS)]

'''Bidirectional LSTM with Dense layer model'''
# LSTM_UNITS = 32
# DENSE_UNITS = 24
# LAYERS = [
#     Bidirectional(LSTM(LSTM_UNITS)),
#     Dense(DENSE_UNITS, activation='relu')
# ]

'''Multilayer bidirectional LSTM with Dense layers model'''
# LSTM_LAYERS_UNITS = [64, 32]
# DENSE_LAYERS_UNITS = [4]
# LAYERS = [Bidirectional(LSTM(units, return_sequences=True if i < (len(LSTM_LAYERS_UNITS) - 1) else None)) for i, units
#           in enumerate(LSTM_LAYERS_UNITS)] + [Dense(units, activation='relu') for units in DENSE_LAYERS_UNITS]

'''GRU model'''
# GRU_UNITS = 64
# DENSE_UNITS = 32
# LAYERS = [
#     Bidirectional(GRU(GRU_UNITS)),
#     Dense(DENSE_UNITS, activation='relu')
# ]

'''ConvD model'''
# CONVD_OPTIONS = {
#     'filters': 64,
#     'kernel_size': 10,
#     'activation': 'relu'
# }
# DENSE_UNITS = 16
# LAYERS = [
#     Conv1D(**CONVD_OPTIONS),
#     GlobalAveragePooling1D(),
#     Dense(DENSE_UNITS, activation='relu')
# ]

'''Convd, LSTM, Dropout'''
# DROPOUT_PERCENTAGE = 0.2
# CONVD_OPTIONS = {
#     'filters': 64,
#     'kernel_size': 5,
#     'activation': 'relu'
# }
# POOL_SIZE = 4
# LSTM_UNITS = 64
# LAYERS = [
#     Dropout(DROPOUT_PERCENTAGE),
#     Conv1D(**CONVD_OPTIONS),
#     MaxPool1D(POOL_SIZE),
#     LSTM(LSTM_UNITS),
# ]

'''Model'''

# DENSE_UNITS = 64
# LAYERS = [
#     GlobalAveragePooling1D(),
#     Dense(DENSE_UNITS, activation='relu'),
# ]

########################################################################################################################

# checkpoint = ModelCheckpoint(
#     './data/models/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
#     verbose=1,
#     monitor='val_loss',
#     save_best_only=True,
#     mode='auto'
# )

test_callback = KerasTestCallback(
    x_test=np.array(x_test),
    y_test=np.array(y_test)
)

########################################################################################################################

model = KerasModels.get_keras_model(
    layers=LAYERS,
    embedding_options=MODEL['EMBEDDING_OPTIONS'],
    optimizer=MODEL['OPTIMIZER']
)

history = model.fit(
    x=np.array(x_train),
    y=np.array(y_train),
    batch_size=MODEL['BATCH_SIZE'],
    epochs=MODEL['EPOCHS'],
    verbose=MODEL['VERBOSE'],
    shuffle=MODEL['SHUFFLE'],
    validation_data=(
        np.array(x_val),
        np.array(y_val)
    ),
    callbacks=[test_callback]
)

draw_keras_graph(history)

log(
    data=DATA,
    vocabulary=VOCABULARY,
    model=MODEL,
    model_history=history.history,
    model_config=model.get_config(),
    test_performance=test_callback.test_performance
)
