# -*- coding: utf-8 -*-
import numpy as np
from tweets import TweetsVectorization, tweets_preprocessor
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Embedding, LSTM, SpatialDropout1D
from keras.layers import Bidirectional, GRU, Conv1D, GlobalAveragePooling1D, Dropout, MaxPool1D
from data import train_data as data, test_data_with_target as test_data
from models import KerasModels, KerasTestCallback
from utils import log, draw_keras_graph, relable, get_glove_from_txt
from sklearn.model_selection import train_test_split

DATA = {
    'GLOVE_SIZE': 300,
    'VALIDATION_PERCENTAGE': 0.2,
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
        'stem': False,
        'remove_punctuations': True,
        'remove_numbers': True,
        'to_lower_case': True,
        'remove_stop_words': True,
        'remove_not_alpha': True,
        'join': False
    }
}

# GLOVE_FILE_PATH = f'./data/glove/glove.twitter.27B.{DATA["GLOVE_SIZE"]}d.txt'
# GLOVE_FILE_PATH = f'./data/glove/glove.6b.{DATA["GLOVE_SIZE"]}d.txt'
GLOVE_FILE_PATH = f'./data/glove/glove.42B.{DATA["GLOVE_SIZE"]}d.txt'

########################################################################################################################

data['preprocessed'] = tweets_preprocessor.preprocess(data.text, DATA['PREPROCESS_OPTRIONS'])

relable(data)

test_data['preprocessed'] = tweets_preprocessor.preprocess(
    test_data.text,
    DATA['PREPROCESS_OPTRIONS']
)

glove_embeddings = get_glove_from_txt(GLOVE_FILE_PATH)

########################################################################################################################

keras_tokenizer = Tokenizer()

keras_tokenizer.fit_on_texts(data.preprocessed)

sequences = keras_tokenizer.texts_to_sequences(data.preprocessed)
sequences_test = keras_tokenizer.texts_to_sequences(test_data.preprocessed)

MAX_LEN = TweetsVectorization.get_max_vector_len(sequences)

sequences_padded = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
sequences_test_padded = pad_sequences(sequences_test, maxlen=MAX_LEN, truncating='post', padding='post')

word_index = keras_tokenizer.word_index

WORD_INDEX_SIZE = len(word_index) + 1

sorted_missing, train_glove_vocab_coverage, train_glove_text_coverage = TweetsVectorization.check_embeddings_coverage(
    keras_tokenizer.word_counts,
    glove_embeddings
)

print(
    'GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(
        train_glove_vocab_coverage, train_glove_text_coverage
    )
)

embedding_matrix = TweetsVectorization.get_embedding_matrix(
    word_index=word_index,
    word_index_size=WORD_INDEX_SIZE,
    glove_embeddings=glove_embeddings,
    glove_size=DATA['GLOVE_SIZE'],
)

x_train, x_val, y_train, y_val = train_test_split(
    sequences_padded,
    data['target_relabeled'].values,
    test_size=DATA['VALIDATION_PERCENTAGE']
)

x_test = sequences_test_padded
y_test = test_data.target.values

########################################################################################################################

MODEL = {
    'BATCH_SIZE': 8,
    'EPOCHS': 50,
    'VERBOSE': 1,
    'OPTIMIZER': 'rmsprop',
    'LEARNING_RATE': 0.0001,
    'SHUFFLE': True,
    'EMBEDDING_OPTIONS': {
        'input_dim': len(word_index) + 1,
        'output_dim': DATA['GLOVE_SIZE'],
        'input_length': MAX_LEN,
        'embeddings_initializer': Constant(embedding_matrix),
        'trainable': False
    }
}

########################################################################################################################

'''Binary classification model'''
# LAYERS = [Flatten()]

'''Model'''
LAYERS = [
    SpatialDropout1D(0.2),
    LSTM(50, dropout=0.2, recurrent_dropout=0.2)
]

'''Multi layer binary classification model'''
# DENSE_LAYERS_UNITS = [16]
# LAYERS = [Flatten()] + [Dense(unit, activation='relu') for unit in DENSE_LAYERS_UNITS]

'''LSTM model'''
# LSTM_UNITS = 256
# LAYERS = [LSTM(LSTM_UNITS)]

'''Bidirectional LSTM with Dense layer model'''
# LSTM_UNITS = 50
# DENSE_UNITS = 25
# LAYERS = [
#     Bidirectional(LSTM(LSTM_UNITS)),
#     Dense(DENSE_UNITS, activation='relu')
# ]

'''Multilayer bidirectional LSTM with Dense layers model'''
# LSTM_LAYERS_UNITS = [50, 25]
# DENSE_LAYERS_UNITS = [10]
# LAYERS = [Bidirectional(LSTM(units, return_sequences=True if i < (len(LSTM_LAYERS_UNITS) - 1) else None)) for i, units
#           in enumerate(LSTM_LAYERS_UNITS)] + [Dense(units, activation='relu') for units in DENSE_LAYERS_UNITS]

'''GRU model'''
# GRU_UNITS = 100
# DENSE_UNITS = 50
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
#     'filters': 50,
#     'kernel_size': 5,
#     'activation': 'relu'
# }
# POOL_SIZE = 4
# LSTM_UNITS = 25
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
    optimizer=MODEL['OPTIMIZER'],
    learning_rate=MODEL['LEARNING_RATE']
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

draw_keras_graph(history, test_callback.test_performance)

log(
    file='app_glove.py',
    data=DATA,
    model=MODEL,
    model_history=history.history,
    model_config=model.get_config(),
    test_performance=test_callback.test_performance
)
