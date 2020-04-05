# -*- coding: utf-8 -*-
from tweets import TweetsVectorization, tweets_preprocessor
from data import data

# shuffle data
# data = data.sample(frac=1).reset_index(drop=True)

TRAIN_PERCENTAGE = 0.8
PREPROCESS_OPTRIONS = {
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
}

x_train, y_train, x_val, y_val = TweetsVectorization.get_prepared_data_based_on_count_vectorizer(
    tweets_preprocessor=tweets_preprocessor,
    tweets=data.text,
    target=data.target,
    preprocess_options=PREPROCESS_OPTRIONS,
    train_percentage=TRAIN_PERCENTAGE
)

# clf = linear_model.RidgeClassifier()
# scores = model_selection.cross_val_score(clf, x_train, y_train, cv=3, scoring="f1")
# print(scores)
# clf.fit(x_train, y_train)
# print(clf.score(x_val, y_val))

# from sklearn.linear_model import LinearRegression
#
# reg = LinearRegression().fit(x_train, y_train)
# print(reg.score(x_train, y_train))
#
# print(reg.score(x_val, y_val))
