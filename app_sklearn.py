# -*- coding: utf-8 -*-
from tweets import tweets_preprocessor
from models import Sklearn
from utils import save_classifier, log_classifier
from data import train_data, test_data
from configs import get_preprocessing_algorithm
from sklearn.model_selection import train_test_split
import uuid

SEED = 7

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

PREPROCESSING_ALGORITHM_ID = PREPROCESSING_ALGORITHM_IDS[7]
PREPROCESSING_ALGORITHM = get_preprocessing_algorithm(PREPROCESSING_ALGORITHM_ID, join=True)

VECTORIZER = {
    'TYPE': 'TFIDF',
    'OPTIONS': {
        'binary': True,
        'ngram_range': (1, 3)
    }
}

CLASSIFIER = {
    'TYPE': 'SGD',
    'OPTIONS': {}
}

LOG_DICT = {
    'UUID': str(uuid.uuid4()),
    'PREPROCESSING_ALGORITHM_UUID': PREPROCESSING_ALGORITHM_ID,
    'PREPROCESSING_ALGORITHM': PREPROCESSING_ALGORITHM,
    'VECTORIZER': VECTORIZER,
    'CLASSIFIER': CLASSIFIER,
}

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

vectorizer = Sklearn.VECTORIZERS[VECTORIZER['TYPE']](**VECTORIZER['OPTIONS'])

x_train = vectorizer.fit_transform(train_inputs).todense()
y_train = train_targets

x_val = vectorizer.transform(val_inputs).todense()
y_val = val_targets

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
    'test_score': test_score
}

LOG_DICT['HISTORY'] = history

log_classifier(LOG_DICT)
save_classifier(
    f'./data-saved-models/classifiers/{CLASSIFIER["TYPE"]}/{CLASSIFIER["TYPE"]}-{PREPROCESSING_ALGORITHM_ID}-{VECTORIZER["TYPE"]}-{VECTORIZER["OPTIONS"]["ngram_range"][0]}-{VECTORIZER["OPTIONS"]["ngram_range"][1]}-{train_score}-{val_score}-{test_score}.pickle',
    classifier
)
