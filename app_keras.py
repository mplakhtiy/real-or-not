# -*- coding: utf-8 -*-
import numpy as np
from tweets import TweetsVectorization, tweets_preprocessor
from models import KerasModels
from utils import draw_keras_graph, log
from keras.layers import Dense, Flatten, Embedding, LSTM
from keras.layers import Bidirectional, GRU, Conv1D, GlobalAveragePooling1D, Dropout, MaxPool1D
from data import data

########################################################################################################################

# Shuffle Data
# data = data.sample(frac=1).reset_index(drop=True)

WORDS_REPUTATION_FILTER = 0
TRAIN_PERCENTAGE = 0.8
ADD_START_SYMBOL = True
PREPROCESS_OPTRIONS = {
    'remove_links': True,
    'remove_users': True,
    'remove_hash': False,
    'unslang': False,
    'split_words': True,
    'stem': False,
    'remove_punctuations': True,
    'remove_numbers': False,
    'to_lower_case': True,
    'remove_stop_words': True,
    'remove_not_alpha': False,
    'join': False
}

x_train, y_train, x_val, y_val, vocabulary, max_vector_len = TweetsVectorization.get_prepared_data_based_on_words_indexes(
    tweets_preprocessor=tweets_preprocessor,
    tweets=data.text,
    target=data.target,
    preprocess_options=PREPROCESS_OPTRIONS,
    tweets_for_words_base=data.text[data.target == 1],
    words_reputation_filter=WORDS_REPUTATION_FILTER,
    train_percentage=TRAIN_PERCENTAGE,
    add_start_symbol=ADD_START_SYMBOL
)

########################################################################################################################

BATCH_SIZE = 512
EPOCHS = 20
VERBOSE = 1
EMBEDING_DIM = 256
OPTIMIZER = 'rmsprop'
INPUT_LENGTH = max_vector_len
EMBEDDING_OPTIONS = {
    'input_dim': len(vocabulary),
    'output_dim': EMBEDING_DIM,
    'input_length': INPUT_LENGTH
}

########################################################################################################################

'''Binary classification model'''
LAYERS = [Flatten()]

'''Multi layer binary classification model'''
# DENSE_LAYERS_UNITS = [32]
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

########################################################################################################################

model = KerasModels.get_keras_model(
    layers=LAYERS,
    embedding_options=EMBEDDING_OPTIONS,
    optimizer=OPTIMIZER
)

history = model.fit(
    x=np.array(x_train),
    y=np.array(y_train),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    shuffle=True,
    validation_data=(
        np.array(x_val),
        np.array(y_val)
    )
)

draw_keras_graph(history)

log(
    preprocess_options=PREPROCESS_OPTRIONS,
    model_history=history.history,
    model_config=model.get_config(),
    words_reputation_filter=WORDS_REPUTATION_FILTER,
    train_percentage=TRAIN_PERCENTAGE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    embedding_dim=EMBEDING_DIM,
    vocabulary_len=len(vocabulary),
    add_start_symbol=ADD_START_SYMBOL,
    input_len=INPUT_LENGTH,
    optimizer=OPTIMIZER
)
