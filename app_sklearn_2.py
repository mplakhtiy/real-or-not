# -*- coding: utf-8 -*-
from tweets import tweets_preprocessor, Helpers
from models import Sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from utils import log, get_from_file, save_to_file
from data import train_data as data, test_data_with_target as test_data
########################################################################################################################

DATA = {
    'VALIDATION_PERCENTAGE': 0.2,
    'PREPROCESS_OPTRIONS': [
        {
            'add_link_flag': False,
            'add_user_flag': False,
            'add_hash_flag': False,
            'add_number_flag': False,
            'remove_links': True,
            'remove_users': True,
            'remove_hash': True,
            'unslang': True,
            'split_words': True,
            'stem': False,
            'remove_punctuations': True,
            'remove_numbers': True,
            'to_lower_case': True,
            'remove_stop_words': True,
            'remove_not_alpha': True,
            'join': True
        },
        {
            'add_link_flag': True,
            'add_user_flag': True,
            'add_hash_flag': True,
            'add_number_flag': True,
            'remove_links': True,
            'remove_users': True,
            'remove_hash': True,
            'unslang': True,
            'split_words': True,
            'stem': False,
            'remove_punctuations': True,
            'remove_numbers': True,
            'to_lower_case': True,
            'remove_stop_words': True,
            'remove_not_alpha': True,
            'join': True
        },
        {
            'add_link_flag': False,
            'add_user_flag': False,
            'add_hash_flag': False,
            'add_number_flag': False,
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
        },
        {
            'add_link_flag': True,
            'add_user_flag': True,
            'add_hash_flag': True,
            'add_number_flag': True,
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
        },
        {
            'add_link_flag': True,
            'add_user_flag': True,
            'add_hash_flag': True,
            'add_number_flag': True,
            'remove_links': True,
            'remove_users': True,
            'remove_hash': False,
            'unslang': True,
            'split_words': False,
            'stem': False,
            'remove_punctuations': True,
            'remove_numbers': True,
            'to_lower_case': True,
            'remove_stop_words': False,
            'remove_not_alpha': False,
            'join': True
        },
        {
            'add_link_flag': False,
            'add_user_flag': False,
            'add_hash_flag': False,
            'add_number_flag': False,
            'remove_links': True,
            'remove_users': True,
            'remove_hash': False,
            'unslang': True,
            'split_words': False,
            'stem': False,
            'remove_punctuations': True,
            'remove_numbers': True,
            'to_lower_case': True,
            'remove_stop_words': False,
            'remove_not_alpha': False,
            'join': True
        },
        {
            'add_link_flag': True,
            'add_user_flag': True,
            'add_hash_flag': True,
            'add_number_flag': True,
            'remove_links': True,
            'remove_users': True,
            'remove_hash': True,
            'unslang': True,
            'split_words': True,
            'stem': True,
            'remove_punctuations': True,
            'remove_numbers': True,
            'to_lower_case': True,
            'remove_stop_words': True,
            'remove_not_alpha': True,
            'join': True
        },
        {
            'add_link_flag': True,
            'add_user_flag': True,
            'add_hash_flag': True,
            'add_number_flag': True,
            'remove_links': True,
            'remove_users': True,
            'remove_hash': False,
            'unslang': False,
            'split_words': False,
            'stem': False,
            'remove_punctuations': False,
            'remove_numbers': False,
            'to_lower_case': True,
            'remove_stop_words': False,
            'remove_not_alpha': False,
            'join': True
        },
        {
            'add_link_flag': False,
            'add_user_flag': False,
            'add_hash_flag': False,
            'add_number_flag': False,
            'remove_links': True,
            'remove_users': True,
            'remove_hash': False,
            'unslang': False,
            'split_words': False,
            'stem': False,
            'remove_punctuations': False,
            'remove_numbers': False,
            'to_lower_case': True,
            'remove_stop_words': False,
            'remove_not_alpha': False,
            'join': True
        },
    ]
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
            'ngram_range': (1, 3)
        },
    },
    {
        'TYPE': 'COUNT',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 3)
        }
    },
    {
        'TYPE': 'TFIDF',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 4)
        },
    },
    {
        'TYPE': 'COUNT',
        'OPTIONS': {
            'binary': True,
            'ngram_range': (1, 4)
        }
    },
]

CLASSIFIERS = [
    # 'RIDGE',
    # 'LINEAR_REGRESSION',
    # 'LOGISTIC_REGRESSION',
    'RANDOM_FOREST',
    'DECISSION_TREE',
    'SVM',
    # 'SGDClassifier'
]

########################################################################################################################

Helpers.coorrect_data(data)
v = -1
p = -1
c = -1
for vectoriz in VECTORIZERS:
    v += 1
    p = -1
    c = -1
    for preprocess_option in DATA['PREPROCESS_OPTRIONS']:
        p += 1
        c = -1

        data['preprocessed'] = tweets_preprocessor.preprocess(data.text, preprocess_option)

        test_data['preprocessed'] = tweets_preprocessor.preprocess(
            test_data.text,
            preprocess_option
        )

        vectorizer = Sklearn.get_vectorizer(vectoriz['TYPE'], vectoriz['OPTIONS'])

        x_train, x_val, y_train, y_val = train_test_split(
            vectorizer.fit_transform(data.preprocessed).todense(),
            data['target_relabeled'].values,
            test_size=DATA['VALIDATION_PERCENTAGE']
        )

        x_test = vectorizer.transform(test_data.preprocessed)
        y_test = [int(y_) for y_ in test_data.target.values]

        y_train = [int(y_) for y_ in y_train]
        y_val = [int(y_) for y_ in y_val]

        for classif in CLASSIFIERS:
            c += 1
            try:
                ########################################################################################################################

                classifier = Sklearn.get_classifier(classif)

                ########################################################################################################################

                # cross_val_scores = cross_val_score(classifier, x_train, y_train, cv=3, scoring="f1")

                classifier.fit(x_train, y_train)

                validation_score = classifier.score(x_val, y_val)
                test_score = classifier.score(x_test, y_test)

                print(f'vectorizer: {v}, preprocess: {p}, classifier {c}')
                # print(f'Cross Validation Scores: {cross_val_scores}')
                print(f'Mean Score of Validation Set: {validation_score}')
                print(f'Mean Score of Test Set: {test_score}')

                log(
                    file='app_sklearn.py',
                    classifier={
                        'classifier': classif,
                        'val_score': round(validation_score, 5),
                        'test_score': round(test_score, 5),
                        # 'cross_val_scores': [round(s, 5) for s in cross_val_scores]
                    },
                    vectorizer=vectoriz,
                    data={'PREPROCESS_OPTRIONS': preprocess_option}
                )
            except:
                print(f'Failed vectorizer: {v}, preprocess: {p}, classifier {c}')
                failed = get_from_file('failed.json')

                failed[f'{v}-{p}-{c}'] = {
                    v: vectoriz,
                    p: preprocess_option,
                    c: classif
                }

                save_to_file('failed.json', failed)
