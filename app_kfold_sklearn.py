# -*- coding: utf-8 -*-
from tweets import tweets_preprocessor
from models import Sklearn
from utils import log_classifier
from data import train_data, test_data
from sklearn.model_selection import StratifiedKFold
from configs import get_preprocessing_algorithm
import uuid
import numpy as np

########################################################################################################################
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
    {
        'TYPE': 'RIDGE',
        'OPTIONS': {}
    },
    {
        'TYPE': 'LOGISTIC_REGRESSION',
        'OPTIONS': {}
    },
    {
        'TYPE': 'RANDOM_FOREST',
        'OPTIONS': {}
    },
    {
        'TYPE': 'DECISION_TREE',
        'OPTIONS': {}
    },
    {
       'TYPE': 'SVC',
       'OPTIONS': {}
    },
    {
        'TYPE': 'SGD',
        'OPTIONS': {}
    }
]
SEED = 7
KFOLD = 10

PAIRS = [
    [0, 2, 7],
    [1, 3, 3],
    [2, 0, 1],
    [3, 3, 4],
    # [4, 1, 2],
    [5, 4, 7],
]
TRAIN_UUID = str(uuid.uuid4())

for pair in PAIRS:
    CLASSIFIER = CLASSIFIERS[pair[0]]
    VECTORIZER = VECTORIZERS[pair[1]]
    preprocessing_algorithm_id = PREPROCESSING_ALGORITHM_IDS[pair[2]]
    print(CLASSIFIER, VECTORIZER, preprocessing_algorithm_id)

    preprocessing_algorithm = get_preprocessing_algorithm(preprocessing_algorithm_id, join=True)

    LOG_DICT = {
        'UUID': str(uuid.uuid4()),
        'TRAIN_UUID': TRAIN_UUID,
        'PREPROCESSING_ALGORITHM_UUID': preprocessing_algorithm_id,
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

    inputs = np.concatenate([train_data['preprocessed'], test_data.preprocessed])
    targets = np.concatenate([train_data['target'], test_data.target])

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

        classifier.fit(x_train, y_train)

        train_score = round(classifier.score(x_train, y_train), 6)
        val_score = round(classifier.score(x_val, y_val), 6)
        test_score = round(classifier.score(x_test, y_test), 6)

        history = {
            'train_score': train_score,
            'val_score': val_score,
            'val_predictions': classifier.predict(x_val).tolist(),
            'test_score': test_score,
            'test_predictions': classifier.predict(x_test).tolist(),
        }

        LOG_DICT['KFOLD_HISTORY'].append(history)

        print(CLASSIFIER['TYPE'], VECTORIZER, k)

        log_classifier(LOG_DICT)

        k += 1
