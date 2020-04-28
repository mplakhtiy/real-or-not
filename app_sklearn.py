# -*- coding: utf-8 -*-
from tweets import tweets_preprocessor, Helpers
from models import Sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from utils import log
from data import train_data as data, test_data_with_target as test_data
import pickle

########################################################################################################################

DATA = {
    'VALIDATION_PERCENTAGE': 0.2,
    'PREPROCESS_OPTRIONS': {
        'add_link_flag': False,
        'add_user_flag': False,
        'add_hash_flag': False,
        'add_number_flag': False,
        'add_keyword_flag': False,
        'add_location_flag': False,
        'remove_links': True,
        'remove_users': True,
        'remove_hash': False,
        'unslang': True,
        'split_words': False,
        'stem': False,
        'remove_punctuations': True,
        'remove_numbers': True,
        'to_lower_case': True,
        'remove_stop_words': True,
        'remove_not_alpha': False,
        'join': True
    }
}

VECTORIZER = {
    'TYPE': 'TFIDF',
    'OPTIONS': {
        'binary': True,
        'ngram_range': (1, 3)
    }
}

CLASSIFIER = {
    'TYPE': 'RIDGE',
    'OPTIONS': {}
}

########################################################################################################################

data['preprocessed'] = tweets_preprocessor.preprocess(
    data.text,
    DATA['PREPROCESS_OPTRIONS'],
    keywords=data.keyword,
    locations=data.location
)

Helpers.correct_data(data)

test_data['preprocessed'] = tweets_preprocessor.preprocess(
    test_data.text,
    DATA['PREPROCESS_OPTRIONS'],
    keywords=test_data.keyword,
    locations=test_data.location
)

vectorizer = Sklearn.VECTORIZERS[VECTORIZER['TYPE']](**VECTORIZER['OPTIONS'])

x_train, x_val, y_train, y_val = train_test_split(
    vectorizer.fit_transform(data.preprocessed).todense(),
    data['target_relabeled'].values,
    test_size=DATA['VALIDATION_PERCENTAGE']
)

x_test = vectorizer.transform(test_data.preprocessed)
y_test = test_data.target.values

########################################################################################################################

classifier = Sklearn.CLASSIFIERS[CLASSIFIER['TYPE']](**CLASSIFIER['OPTIONS'])

########################################################################################################################

cross_val_scores = cross_val_score(classifier, x_train, y_train, cv=3, scoring="f1")

classifier.fit(x_train, y_train)

validation_score = classifier.score(x_val, y_val)
test_score = classifier.score(x_test, y_test)

print(f'Cross Validation Scores: {cross_val_scores}')
print(f'Mean Score of Validation Set: {validation_score}')
print(f'Mean Score of Test Set: {test_score}')

log(
    file='app_sklearn.py',
    classifier={
        'classifier': CLASSIFIER,
        'val_score': round(validation_score, 5),
        'test_score': round(test_score, 5),
        'cross_val_scores': [round(s, 5) for s in cross_val_scores]
    },
    vectorizer=VECTORIZER,
    data=DATA
)

filename = f'./data/classifiers/{CLASSIFIER["TYPE"]}-{VECTORIZER["TYPE"]}-{validation_score}-{test_score}.pickle'
pickle.dump(classifier, open(filename, 'wb'))
