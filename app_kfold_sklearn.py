# -*- coding: utf-8 -*-
from tweets import tweets_preprocessor
from models import Sklearn
from sklearn.model_selection import cross_val_score
from utils import log, save_classifier, log_classifier
from data import train_data, test_data
from sklearn.model_selection import StratifiedKFold
import uuid


########################################################################################################################

PREPROCESSING_ALGORITHMS = {
    '1258a9d2-111e-4d4a-acda-852dd7ba3e88': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    '60314ef9-271d-4865-a7db-6889b1670f59': {
        'add_link_flag': False, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    '4c2e484d-5cb8-4e3e-ba7b-679ae7a73fca': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': True, 'add_location_flag': True, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    '8b7db91c-c8bf-40f2-986a-83a659b63ba6': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    '7bc816a1-25df-4649-8570-0012d0acd72a': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    'a85c8435-6f23-4015-9e8c-19547222d6ce': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True},
    'b054e509-4f04-44f2-bcf9-14fa8af4eeed': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True},
    '2e359f0b-bfb9-4eda-b2a4-cd839c122de6': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True},
    '71bd09db-e104-462d-887a-74389438bb49': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True},
    'd3cc3c6e-10de-4b27-8712-8017da428e41': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True}
}

VECTORIZERS = [
    {
        'TYPE': 'TFIDF',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 1)
        }
    },
    {
        'TYPE': 'COUNT',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 1)
        }
    },
    {
        'TYPE': 'TFIDF',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 2)
        }
    },
    {
        'TYPE': 'COUNT',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 2)
        }
    },
    {
        'TYPE': 'TFIDF',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 3)
        }
    },
    {
        'TYPE': 'COUNT',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 3)
        }
    }
]

CLASSIFIERS = [
    # {
    #     'TYPE': 'RIDGE',
    #     'OPTIONS': {}
    # },
    # {
    #     'TYPE': 'LOGISTIC_REGRESSION',
    #     'OPTIONS': {}
    # },
    # {
    #     'TYPE': 'RANDOM_FOREST',
    #     'OPTIONS': {}
    # },
    # {
    #     'TYPE': 'DECISION_TREE',
    #     'OPTIONS': {}
    # },
    {
        'TYPE': 'SVC',
        'OPTIONS': {}
    },
    # {
    #     'TYPE': 'SGD',
    #     'OPTIONS': {}
    # }
]
SEED = 7
KFOLD = 10

for VECTORIZER in VECTORIZERS:
    for CLASSIFIER in CLASSIFIERS:
        for algorithm_id, preprocessing_algorithm in PREPROCESSING_ALGORITHMS.items():
            LOG_DICT = {
                'UUID': str(uuid.uuid4()),
                'PREPROCESSING_ALGORITHM_UUID': algorithm_id,
                'PREPROCESSING_ALGORITHM': preprocessing_algorithm,
                'VECTORIZER': VECTORIZER,
                'CLASSIFIER': CLASSIFIER,
                'KFOLD_HISTORY': []
            }

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

            k = 0

            for train, validation in kfold.split(inputs, targets):

                vectorizer = Sklearn.VECTORIZERS[VECTORIZER['TYPE']](**VECTORIZER['OPTIONS'])

                x_train = vectorizer.fit_transform(inputs[train]).todense()
                y_train = targets[train]

                x_val = vectorizer.transform(inputs[validation]).todense()
                y_val = targets[validation]

                x_test = vectorizer.transform(test_data.preprocessed).todense()
                y_test = test_data.target.values

                classifier = Sklearn.CLASSIFIERS[CLASSIFIER['TYPE']](**CLASSIFIER['OPTIONS'])
                try:
                    classifier.fit(x_train, y_train)

                    train_score = round(classifier.score(x_train, y_train), 6)
                    val_score = round(classifier.score(x_val, y_val), 6)
                    test_score = round(classifier.score(x_test, y_test), 6)

                    history = {
                        'train_score': train_score,
                        'val_score': val_score,
                        'test_score': test_score
                    }
                except:
                    history = 'error'

                LOG_DICT['KFOLD_HISTORY'].append(history)

                print(CLASSIFIER['TYPE'], VECTORIZER, history, k)

                log_classifier(LOG_DICT)

                k += 1

