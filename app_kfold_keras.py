# -*- coding: utf-8 -*-
import uuid
from tweets import Helpers, tweets_preprocessor
from models import Keras
from utils import get_glove_embeddings, log_model
from data import train_data, test_data
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from configs import get_preprocessing_algorithm, get_model_config

TRAIN_UUID = str(uuid.uuid4())

SEED = 7
KFOLD = 10

USE_GLOVE = True

NETWORKS_KEYS = [
    # 'LSTM', 'LSTM_DROPOUT', 'BI_LSTM',
    'LSTM_CNN',
    # 'FASTTEXT', 'RCNN', 'CNN', 'RNN', 'GRU',
]

PREFIX = NETWORKS_KEYS[0]

for key in NETWORKS_KEYS:
    MODEL_CONFIG = get_model_config(key, USE_GLOVE)
    MODEL_CONFIG['TRAIN_UUID'] = TRAIN_UUID

    if USE_GLOVE:
        MODEL_CONFIG['GLOVE'] = {
            'SIZE': 200
        }
        GLOVE = f'glove.twitter.27B.{MODEL_CONFIG["GLOVE"]["SIZE"]}d.txt'
        GLOVE_FILE_PATH = f'./data/glove/{GLOVE}'
        GLOVE_EMBEDDINGS = get_glove_embeddings(GLOVE_FILE_PATH)

    for key, preprocessing_algorithm in get_preprocessing_algorithm().items():
        CONFIG = MODEL_CONFIG.copy()
        CONFIG['UUID'] = str(uuid.uuid4())
        CONFIG['PREPROCESSING_ALGORITHM'] = preprocessing_algorithm
        CONFIG['PREPROCESSING_ALGORITHM_UUID'] = key
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

        inputs = train_data['preprocessed']
        targets = train_data['target']

        for train, validation in kfold.split(inputs, targets):
            keras_tokenizer = Tokenizer()

            (x_train, x_val, x_test), input_dim, input_len = Helpers.get_model_inputs(
                (inputs[train], inputs[validation], test_data.preprocessed),
                keras_tokenizer
            )
            y_train = targets[train]
            y_val = targets[validation]
            y_test = test_data.target.values

            CONFIG['EMBEDDING_OPTIONS']['input_dim'] = input_dim
            CONFIG['EMBEDDING_OPTIONS']['input_length'] = input_len

            if USE_GLOVE:
                Helpers.with_glove_embedding_options(CONFIG, keras_tokenizer, GLOVE_EMBEDDINGS)

            model = Keras.get_model(CONFIG)

            history = Keras.fit(model, (x_train, y_train, x_val, y_val, x_test, y_test), CONFIG)

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
