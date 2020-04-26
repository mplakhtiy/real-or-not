# -*- coding: utf-8 -*-
from tweets import Helpers, tweets_preprocessor
from models import Keras, TestDataCallback
from utils import log, get_glove_embeddings
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, LSTM, SpatialDropout1D
from tensorflow.keras.layers import Bidirectional, Conv1D, GlobalAveragePooling1D
from data import train_data as data, test_data_with_target as test_data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.sequence import pad_sequences

########################################################################################################################
USE_GLOVE = True

DATA = {
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

########################################################################################################################

data['preprocessed'] = tweets_preprocessor.preprocess(
    data.text,
    DATA['PREPROCESS_OPTRIONS']
)

Helpers.correct_data(data)

test_data['preprocessed'] = tweets_preprocessor.preprocess(
    test_data.text,
    DATA['PREPROCESS_OPTRIONS']
)

########################################################################################################################

keras_tokenizer = Tokenizer()
keras_tokenizer.fit_on_texts(data.preprocessed)

sequences = keras_tokenizer.texts_to_sequences(data.preprocessed)
sequences_test = keras_tokenizer.texts_to_sequences(test_data.preprocessed)

MAX_LEN = Helpers.get_max_vector_len(sequences)

sequences_padded = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
sequences_test_padded = pad_sequences(sequences_test, maxlen=MAX_LEN, truncating='post', padding='post')

WORD_INDEX_SIZE = len(keras_tokenizer.word_index) + 1

x = sequences_padded
y = data.target_relabeled.values

x_test = sequences_test_padded
y_test = test_data.target.values

########################################################################################################################

MODEL = {
    'BATCH_SIZE': 32,
    'EPOCHS': 20,
    'VERBOSE': 1,
    'OPTIMIZER': 'rmsprop',
    'LEARNING_RATE': 0.001,
    'SHUFFLE': True,
    'EMBEDDING_OPTIONS': {
        'input_dim': WORD_INDEX_SIZE,
        'output_dim': 50,
        'input_length': MAX_LEN
    },
    'VALIDATION_PERCENTAGE': 0.2
}

########################################################################################################################

if USE_GLOVE:
    DATA['GLOVE_SIZE'] = 100
    DATA['GLOVE'] = f'glove.twitter.27B.{DATA["GLOVE_SIZE"]}d.txt'
    # DATA['GLOVE'] = f'glove.6b.{DATA["GLOVE_SIZE"]}d.txt'
    # DATA['GLOVE'] = f'glove.42B.{DATA["GLOVE_SIZE"]}d.txt'
    GLOVE_FILE_PATH = f'./data/glove/{DATA["GLOVE"]}'
    glove_embeddings = get_glove_embeddings(GLOVE_FILE_PATH)
    train_glove_vocab_coverage, train_glove_text_coverage, _ = Helpers.check_embeddings_coverage(
        keras_tokenizer.word_counts,
        glove_embeddings
    )
    print(
        'GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(
            train_glove_vocab_coverage, train_glove_text_coverage
        )
    )
    embedding_matrix = Helpers.get_embedding_matrix(
        word_index=keras_tokenizer.word_index,
        word_index_size=WORD_INDEX_SIZE,
        glove_embeddings=glove_embeddings,
        glove_size=DATA['GLOVE_SIZE'],
    )
    MODEL['EMBEDDING_OPTIONS']['output_dim'] = DATA['GLOVE_SIZE']
    MODEL['EMBEDDING_OPTIONS']['embeddings_initializer'] = Constant(embedding_matrix)
    MODEL['EMBEDDING_OPTIONS']['trainable'] = False

########################################################################################################################

MODELS_LAYERS = {
    'FLATTEN_LAYER': [Flatten()],
    'LSTM_WITH_SPATIAL_DROPOUT': [
        SpatialDropout1D(0.2),
        LSTM(50, dropout=0.2, recurrent_dropout=0.2)
    ],
    'BIDIRECTIONAL_LSTM': [Bidirectional(LSTM(50))],
    'CNN': [
        Conv1D(**{'filters': 50, 'kernel_size': 5, 'activation': 'relu'}),
        GlobalAveragePooling1D(),
        Dense(10, activation='relu')
    ]
}

########################################################################################################################

checkpoint = ModelCheckpoint(
    './data/models/keras/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
    mode='auto'
)

test_data_callback = TestDataCallback(
    x_test=x_test,
    y_test=y_test
)

########################################################################################################################
model = Keras.get_model(
    layers=MODELS_LAYERS['DENSE'],
    embedding_options=MODEL['EMBEDDING_OPTIONS'],
    optimizer=MODEL['OPTIMIZER'],
    learning_rate=MODEL['LEARNING_RATE']
)

history = model.fit(
    x, y,
    batch_size=MODEL['BATCH_SIZE'],
    epochs=MODEL['EPOCHS'],
    verbose=MODEL['VERBOSE'],
    shuffle=MODEL['SHUFFLE'],
    validation_split=MODEL['VALIDATION_PERCENTAGE'],
    callbacks=[test_data_callback]  # checkpoint
)

model_history = history.history.copy()
model_history['test_loss'] = test_data_callback.loss
model_history['test_accuracy'] = test_data_callback.accuracy

Keras.draw_graph(model_history)

log(
    file='app_keras.py',
    data=DATA,
    model=MODEL,
    model_history=model_history,
    model_config=model.get_config(),
)
