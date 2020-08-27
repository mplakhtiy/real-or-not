# -*- coding: utf-8 -*-
import uuid
from tweets import Helpers, tweets_preprocessor
from models import Keras
from utils import get_glove_embeddings, log_model, ensure_path_exists
from data import train_data, test_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from configs import get_preprocessing_algorithm, get_model_config

SEED = 7
USE_GLOVE = False

NETWORKS_KEYS = ['LSTM', 'LSTM_DROPOUT', 'BI_LSTM', 'LSTM_CNN', 'FASTTEXT', 'RCNN', 'CNN', 'RNN', 'GRU']
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

pairs = [
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

for pair in pairs:

    NETWORK_KEY = NETWORKS_KEYS[pair[0]]
    PREPROCESSING_ALGORITHM_ID = PREPROCESSING_ALGORITHM_IDS[pair[1]]

    MODEL = get_model_config(NETWORK_KEY, glove=USE_GLOVE)
    PREPROCESSING_ALGORITHM = get_preprocessing_algorithm(PREPROCESSING_ALGORITHM_ID)

    if USE_GLOVE:
        MODEL['GLOVE'] = {
            'SIZE': 200
        }
        GLOVE = f'glove.twitter.27B.{MODEL["GLOVE"]["SIZE"]}d.txt'
        GLOVE_FILE_PATH = f'./data/glove/{GLOVE}'
        GLOVE_EMBEDDINGS = get_glove_embeddings(GLOVE_FILE_PATH)

    MODEL['UUID'] = str(uuid.uuid4())
    MODEL['PREPROCESSING_ALGORITHM'] = PREPROCESSING_ALGORITHM
    MODEL['PREPROCESSING_ALGORITHM_UUID'] = PREPROCESSING_ALGORITHM_ID
    MODEL['DIR'] = f'./data-saved-models/glove-false/{NETWORK_KEY}/'
    ensure_path_exists(MODEL['DIR'])
    MODEL['PREFIX'] = f'{NETWORK_KEY}-{PREPROCESSING_ALGORITHM_ID}-SEED-{SEED}'

    train_data['preprocessed'] = tweets_preprocessor.preprocess(
        train_data.text,
        PREPROCESSING_ALGORITHM,
        keywords=train_data.keyword,
        locations=train_data.location
    )

    test_data['preprocessed'] = tweets_preprocessor.preprocess(
        test_data.text,
        PREPROCESSING_ALGORITHM,
        keywords=test_data.keyword,
        locations=test_data.location
    )

    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        train_data['preprocessed'],
        train_data['target'],
        test_size=0.3,
        random_state=SEED
    )

    keras_tokenizer = Tokenizer()

    (x_train, x_val, x_test), input_dim, input_len = Helpers.get_model_inputs(
        (train_inputs, val_inputs, test_data.preprocessed),
        keras_tokenizer
    )
    y_train = train_targets
    y_val = val_targets
    y_test = test_data.target.values

    MODEL['EMBEDDING_OPTIONS']['input_dim'] = input_dim
    MODEL['EMBEDDING_OPTIONS']['input_length'] = input_len

    if USE_GLOVE:
        Helpers.with_glove_embedding_options(MODEL, keras_tokenizer, GLOVE_EMBEDDINGS)

    model = Keras.get_model(MODEL)

    history = Keras.fit(model, (x_train, y_train, x_val, y_val, x_test, y_test), MODEL)

    MODEL['HISTORY'] = history

    log_model(MODEL)
