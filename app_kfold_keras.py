# -*- coding: utf-8 -*-
import uuid
from tweets import Helpers, tweets_preprocessor
from models import Keras
from utils import get_glove_embeddings, log_model
from data import train_data, test_data
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from configs import get_preprocessing_algorithm, get_model_config
import numpy as np

TRAIN_UUID = str(uuid.uuid4())

SEED = 7
KFOLD = 10

USE_GLOVE = True

NETWORKS_KEYS = [
    'LSTM',
    'LSTM_DROPOUT',
    'BI_LSTM',
    'LSTM_CNN',
    'FASTTEXT',
    'RCNN',
    'CNN',
    'RNN',
    'GRU',
]

PREPROCESSING_ALGORITHM_IDS = [
    '1258a9d2',
    '60314ef9',
    '4c2e484d',
    '8b7db91c',
    '7bc816a1',
    'a85c8435',
    'b054e509',
    '2e359f0b',
    '71bd09db',
    'd3cc3c6e',
]

PAIRS_GLOVE = [
    [0, 3],
    [1, 3],
    [2, 3],
    [3, 3],
    [4, 3],
    [5, 8],
    [6, 9],
    [7, 4],
    [8, 3],
]

PAIRS_KERAS = [
    [0, 8],
    [1, 9],
    [2, 3],
    [3, 3],
    [4, 6],
    [5, 4],
    [6, 6],
    [7, 0],
    [8, 6]
]

pairs = PAIRS_GLOVE if USE_GLOVE else PAIRS_KERAS
GLOVE_EMBEDDINGS = get_glove_embeddings('./data/glove/glove.twitter.27B.200d.txt')

for pair in pairs:
    model_key = NETWORKS_KEYS[pair[0]]
    preprocessing_algorithm_id = PREPROCESSING_ALGORITHM_IDS[pair[1]]

    MODEL_CONFIG = get_model_config(model_key, USE_GLOVE)

    if USE_GLOVE:
        MODEL_CONFIG['GLOVE'] = {
            'SIZE': 200
        }
        GLOVE = f'glove.twitter.27B.{MODEL_CONFIG["GLOVE"]["SIZE"]}d.txt'
        GLOVE_FILE_PATH = f'./data/glove/{GLOVE}'

    CONFIG = MODEL_CONFIG.copy()
    CONFIG['TRAIN_UUID'] = TRAIN_UUID
    CONFIG['UUID'] = str(uuid.uuid4())
    preprocessing_algorithm = get_preprocessing_algorithm(preprocessing_algorithm_id)
    CONFIG['PREPROCESSING_ALGORITHM'] = preprocessing_algorithm
    CONFIG['PREPROCESSING_ALGORITHM_UUID'] = preprocessing_algorithm_id
    CONFIG['KFOLD_HISTORY'] = []

    kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)

    train_data['preprocessed'] = tweets_preprocessor.preprocess(
        train_data.text,
        preprocessing_algorithm,
        keywords=train_data.keyword,
        locations=train_data.location
    )

    test_data['preprocessed'] = tweets_preprocessor.preprocess(
        test_data.text,
        preprocessing_algorithm,
        keywords=test_data.keyword,
        locations=test_data.location
    )

    inputs = np.concatenate([train_data['preprocessed'], test_data.preprocessed])
    targets = np.concatenate([train_data['target'], test_data.target])

    for train, validation in kfold.split(inputs, targets):
        keras_tokenizer = Tokenizer()

        (x_train, x_val), input_dim, input_len = Helpers.get_model_inputs(
            (inputs[train], inputs[validation]),
            keras_tokenizer
        )
        y_train = targets[train]
        y_val = targets[validation]

        CONFIG['EMBEDDING_OPTIONS']['input_dim'] = input_dim
        CONFIG['EMBEDDING_OPTIONS']['input_length'] = input_len

        if USE_GLOVE:
            Helpers.with_glove_embedding_options(CONFIG, keras_tokenizer, GLOVE_EMBEDDINGS)

        model = Keras.get_model(CONFIG)

        history = Keras.fit(model, (x_train, y_train, x_val, y_val, x_val, y_val), CONFIG)

        try:
            del CONFIG['EMBEDDING_OPTIONS']['embeddings_initializer']
        except KeyError:
            pass

        try:
            del CONFIG['EMBEDDING_OPTIONS']['trainable']
        except KeyError:
            pass

        try:
            history['EMBEDDING_OPTIONS'] = CONFIG['EMBEDDING_OPTIONS'].copy()
        except KeyError:
            pass

        try:
            history['GLOVE'] = CONFIG['GLOVE'].copy()
        except KeyError:
            pass

        try:
            del CONFIG['EMBEDDING_OPTIONS']['input_dim']
        except KeyError:
            pass

        try:
            del CONFIG['EMBEDDING_OPTIONS']['input_length']
        except KeyError:
            pass

        try:
            del CONFIG['GLOVE']['VOCAB_COVERAGE']
        except KeyError:
            pass

        try:
            del CONFIG['GLOVE']['TEXT_COVERAGE']
        except KeyError:
            pass

        CONFIG['KFOLD_HISTORY'].append(history)

        log_model(CONFIG)
