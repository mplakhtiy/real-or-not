# -*- coding: utf-8 -*-
from tweets import Helpers, tweets_preprocessor
from models import Keras, TestDataCallback
from utils import log, get_glove_embeddings, ensure_path_exists
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D, Dropout, GlobalMaxPooling1D, GRU
from tensorflow.keras.layers import Bidirectional, Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data import train, validation, test

########################################################################################################################

USE_GLOVE = False
P_A = [
    {'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
     'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
     'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
     'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True, 'join': False},
    {'add_link_flag': False, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
     'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
     'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
     'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True, 'join': False},
    {'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
     'add_keyword_flag': True, 'add_location_flag': True, 'remove_links': True, 'remove_users': True,
     'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
     'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True, 'join': False},
    {'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
     'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
     'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
     'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True, 'join': False},
    {'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
     'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
     'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
     'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True, 'join': False},
    {'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
     'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
     'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': False,
     'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
     'join': False}
]

N = 6

DATA = {
    'PREPROCESS_OPTRIONS': P_A[N]
}

########################################################################################################################

train['preprocessed'] = tweets_preprocessor.preprocess(
    train.text,
    DATA['PREPROCESS_OPTRIONS'],
    keywords=train.keyword,
    locations=train.location
)

validation['preprocessed'] = tweets_preprocessor.preprocess(
    validation.text,
    DATA['PREPROCESS_OPTRIONS'],
    keywords=validation.keyword,
    locations=validation.location
)

test['preprocessed'] = tweets_preprocessor.preprocess(
    test.text,
    DATA['PREPROCESS_OPTRIONS'],
    keywords=test.keyword,
    locations=test.location
)

########################################################################################################################

keras_tokenizer = Tokenizer()
keras_tokenizer.fit_on_texts(train.preprocessed)

sequences_train = keras_tokenizer.texts_to_sequences(train.preprocessed)
sequences_validation = keras_tokenizer.texts_to_sequences(validation.preprocessed)
sequences_test = keras_tokenizer.texts_to_sequences(test.preprocessed)

MAX_LEN = Helpers.get_max_vector_len(sequences_train)

x_train = pad_sequences(sequences_train, maxlen=MAX_LEN, truncating='post', padding='post')
y_train = train.target.values
x_val = pad_sequences(sequences_validation, maxlen=MAX_LEN, truncating='post', padding='post')
y_val = validation.target.values
x_test = pad_sequences(sequences_test, maxlen=MAX_LEN, truncating='post', padding='post')
y_test = test.target.values

WORD_INDEX_SIZE = len(keras_tokenizer.word_index) + 1

########################################################################################################################

MODEL = {
    'BATCH_SIZE': 16,
    'EPOCHS': 8,
    'VERBOSE': 1,
    'OPTIMIZER': 'rmsprop',
    'LEARNING_RATE': 1e-4,
    'SHUFFLE': True,
    'EMBEDDING_OPTIONS': {
        'input_dim': WORD_INDEX_SIZE,
        'output_dim': 256,
        'input_length': MAX_LEN
    },
    'TYPE': 'GRU'
}

########################################################################################################################

if USE_GLOVE:
    DATA['GLOVE_SIZE'] = 100
    DATA['GLOVE'] = f'glove.twitter.27B.{DATA["GLOVE_SIZE"]}d.txt'
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
    'LSTM': [
        LSTM(64)
    ],
    'LSTM_DROPOUT': [
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2)
    ],
    'BI_LSTM': [
        Bidirectional(LSTM(64)),
        Dropout(0.2)
    ],
    'FASTTEXT': [
        GlobalAveragePooling1D(),
        Dense(64, activation='relu')
    ],
    'RCNN': [
        Dropout(0.25),
        Conv1D(filters=128, kernel_size=5, activation='relu', strides=1, padding='valid'),
        MaxPooling1D(pool_size=4),
        LSTM(128),
    ],
    'CNN': [
        Dropout(0.25),
        Conv1D(filters=128, kernel_size=5, activation='relu', strides=1, padding='valid'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
    ],
    'RNN': [
        Bidirectional(LSTM(128)),
        Dense(128, activation='relu')
    ],
    'GRU': [
        Bidirectional(GRU(128, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(64, activation="relu"),
        Dropout(0.1)
    ],
}

########################################################################################################################

MODELS_DIR_SAVE_PATH = f'./data/models/keras/{MODEL["TYPE"]}/{N}/'
ensure_path_exists(MODELS_DIR_SAVE_PATH)

MODEL_PREFIX = f'model-{MODEL["OPTIMIZER"]}-bs{MODEL["BATCH_SIZE"]}-lr{MODEL["LEARNING_RATE"]}-len{MODEL["EMBEDDING_OPTIONS"]["output_dim"]}'

checkpoint = ModelCheckpoint(
    MODELS_DIR_SAVE_PATH + MODEL_PREFIX + '-e{epoch:03d}-a{accuracy:03f}-va{val_accuracy:03f}-ta.h5',
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
    layers=MODELS_LAYERS[MODEL['TYPE']],
    embedding_options=MODEL['EMBEDDING_OPTIONS'],
    optimizer=MODEL['OPTIMIZER'],
    learning_rate=MODEL['LEARNING_RATE']
)

history = model.fit(
    x_train, y_train,
    batch_size=MODEL['BATCH_SIZE'],
    epochs=MODEL['EPOCHS'],
    verbose=MODEL['VERBOSE'],
    shuffle=MODEL['SHUFFLE'],
    validation_data=(
        x_val,
        y_val
    ),
    callbacks=[checkpoint, test_data_callback]
)

model_history = history.history.copy()
model_history['test_loss'] = test_data_callback.loss
model_history['test_accuracy'] = test_data_callback.accuracy

Keras.draw_graph(model_history)

log(
    target='keras',
    data=DATA,
    model=MODEL,
    model_history=model_history,
    model_config=model.get_config(),
)
