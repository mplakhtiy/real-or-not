# -*- coding: utf-8 -*-
import uuid
from tweets import Helpers, tweets_preprocessor
from models import Keras
from utils import get_glove_embeddings, log_model
from data import train_data, test_data
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer

PREPROCESSING_ALGORITHMS = {
    '1258a9d2-111e-4d4a-acda-852dd7ba3e88': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    '60314ef9-271d-4865-a7db-6889b1670f59': {
        'add_link_flag': False, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    '4c2e484d-5cb8-4e3e-ba7b-679ae7a73fca': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': True, 'add_location_flag': True, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    '8b7db91c-c8bf-40f2-986a-83a659b63ba6': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    '7bc816a1-25df-4649-8570-0012d0acd72a': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    'a85c8435-6f23-4015-9e8c-19547222d6ce': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False},
    'b054e509-4f04-44f2-bcf9-14fa8af4eeed': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False},
    '2e359f0b-bfb9-4eda-b2a4-cd839c122de6': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False},
    '71bd09db-e104-462d-887a-74389438bb49': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False},
    'd3cc3c6e-10de-4b27-8712-8017da428e41': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False}
}

'''
Example of Model Config
MODEL = {
    'ACTIVATION': 'sigmoid',
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 16,
    'EPOCHS': 12,
    'LSTM_UNITS': 128,
    'DROPOUT': 0.2,
    'DENSE_UNITS': 128,
    'CONV_FILTERS': 128,
    'CONV_KERNEL_SIZE': 5,
    'MAX_POOLING_POOL_SIZE': 4,
    'GRU_UNITS': 128,
    'EMBEDDING_OPTIONS': {
        'input_dim': 1000,
        'output_dim': 256,
        'input_length': 100
    },
    'TYPE': 'CNN'
}
'''

MODEL_CONFIG = {
    'TRAIN_UUID': str(uuid.uuid4()),
    'BATCH_SIZE': 32,
    'EPOCHS': 15,
    'OPTIMIZER': 'rmsprop',
    'LEARNING_RATE': 1e-4,
    'EMBEDDING_OPTIONS': {
        'output_dim': 256,
    },
    'LSTM_UNITS': 128,
    'TYPE': 'BI_LSTM',
}

SEED = 7
KFOLD = 10

USE_GLOVE = True

if USE_GLOVE:
    MODEL_CONFIG['GLOVE'] = {
        'SIZE': 200
    }
    GLOVE = f'glove.twitter.27B.{MODEL_CONFIG["GLOVE"]["SIZE"]}d.txt'
    GLOVE_FILE_PATH = f'./data/glove/{GLOVE}'
    GLOVE_EMBEDDINGS = get_glove_embeddings(GLOVE_FILE_PATH)

for network_type in ['LSTM', 'LSTM_DROPOUT', 'BI_LSTM']:
    MODEL_CONFIG['TYPE'] = network_type
    if network_type == 'LSTM_DROPOUT' or network_type == 'BI_LSTM':
        MODEL_CONFIG['DROPOUT'] = 0.2

    for key, preprocessing_algorithm in PREPROCESSING_ALGORITHMS.items():
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

            del CONFIG['EMBEDDING_OPTIONS']['embeddings_initializer']

            history['EMBEDDING_OPTIONS'] = CONFIG['EMBEDDING_OPTIONS'].copy()
            history['GLOVE'] = CONFIG['GLOVE'].copy()

            del CONFIG['EMBEDDING_OPTIONS']['input_dim']
            del CONFIG['EMBEDDING_OPTIONS']['input_length']
            del CONFIG['GLOVE']['VOCAB_COVERAGE']
            del CONFIG['GLOVE']['TEXT_COVERAGE']

            CONFIG['KFOLD_HISTORY'].append(history)

            log_model(CONFIG)
