# -*- coding: utf-8 -*-
from tweets import tweets_preprocessor
from models import Sklearn
from sklearn.model_selection import cross_val_score
from utils import log, save_classifier
from data import train, validation, test

########################################################################################################################

DATA = {
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

vectorizer = Sklearn.VECTORIZERS[VECTORIZER['TYPE']](**VECTORIZER['OPTIONS'])

x_train = vectorizer.fit_transform(train.preprocessed).todense()
y_train = train.target.values

x_val = vectorizer.transform(validation.preprocessed).todense()
y_val = validation.target.values

x_test = vectorizer.transform(test.preprocessed).todense()
y_test = test.target.values

########################################################################################################################

classifier = Sklearn.CLASSIFIERS[CLASSIFIER['TYPE']](**CLASSIFIER['OPTIONS'])

########################################################################################################################

cross_val_scores = [round(s, 5) for s in cross_val_score(classifier, x_train, y_train, cv=3, scoring="f1")]

classifier.fit(x_train, y_train)

train_score = round(classifier.score(x_train, y_train), 5)
val_score = round(classifier.score(x_val, y_val), 5)
test_score = round(classifier.score(x_test, y_test), 5)

print(f'Cross Validation Scores: {cross_val_scores}')
print(f'Mean Score of Train Set: {train_score}')
print(f'Mean Score of Validation Set: {val_score}')
print(f'Mean Score of Test Set: {test_score}')

log(
    target='sklearn',
    classifier={
        'classifier': CLASSIFIER,
        'train_score': train_score,
        'val_score': val_score,
        'test_score': test_score,
        'cross_val_scores': cross_val_scores
    },
    vectorizer=VECTORIZER,
    data=DATA
)

save_classifier(
    f'archive/v2/classifiers/{CLASSIFIER["TYPE"]}-{VECTORIZER["TYPE"]}-{train_score}-{val_score}-{test_score}.pickle',
    classifier
)
