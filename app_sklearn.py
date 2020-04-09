# -*- coding: utf-8 -*-
import numpy as np
from tweets import TweetsVectorization, tweets_preprocessor
from models import SklearnClassifiers
from data import train_data as data, test_data_with_target
from utils import log

########################################################################################################################

DATA = {
    'TRAIN_PERCENTAGE': 0.8,
    'SHUFFLE_DARA': True,
    'PREPROCESS_OPTRIONS': {
        'remove_links': True,
        'remove_users': True,
        'remove_hash': False,
        'unslang': True,
        'split_words': True,
        'stem': False,
        'remove_punctuations': True,
        'remove_numbers': True,
        'to_lower_case': True,
        'remove_stop_words': True,
        'remove_not_alpha': True,
        'join': True
    }
}

VECTORIZER = {
    'TYPE': 'TFIDF',
    'OPTIONS': {
        'ngram_range': (1, 3)
    },
    # 'TYPE': 'COUNT',
    # 'OPTIONS': {
    #     'analyzer': 'word',
    #     'binary': True
    # }
}

CLASSIFIER = 'RIDGE'

########################################################################################################################

data['preprocessed'] = tweets_preprocessor.preprocess(data.text, DATA['PREPROCESS_OPTRIONS'])
test_data_with_target['preprocessed'] = tweets_preprocessor.preprocess(
    test_data_with_target.text,
    DATA['PREPROCESS_OPTRIONS']
)

vectorizer = TweetsVectorization.get_vectorizer(VECTORIZER['TYPE'], VECTORIZER['OPTIONS'])

x_train, y_train, x_val, y_val = TweetsVectorization.get_train_test_split(
    x=data.preprocessed,
    y=data.target.values,
    train_percentage=DATA['TRAIN_PERCENTAGE'],
    shuffle_data=DATA['SHUFFLE_DARA']
)

x_train, y_train = TweetsVectorization.get_prepared_data_based_on_count_vectorizer(
    tweets=x_train,
    target=y_train,
    vectorizer=vectorizer
)

x_val = vectorizer.transform(x_val)
y_val = y_val

x_test = vectorizer.transform(test_data_with_target.preprocessed)
y_test = test_data_with_target.target.values

########################################################################################################################

classifier = SklearnClassifiers.get_classifier(CLASSIFIER)

########################################################################################################################

classifier.fit(x_train, y_train)

validation_score = classifier.score(x_val, y_val)
test_score = classifier.score(x_test, y_test)

print(f'Mean Score of Validation Set: {validation_score}')
print(f'Mean Score of Test Set: {test_score}')

log(
    classifier={
        'classifier': CLASSIFIER,
        'validation_score': "%.2f%%" % (validation_score * 100),
        'test_score': "%.2f%%" % (test_score * 100)
    },
    vectorizer=VECTORIZER,
    data=DATA
)
