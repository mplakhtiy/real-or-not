# -*- coding: utf-8 -*-
from tweets import TweetsVectorization, tweets_preprocessor
from models import SklearnModels
from data import data

########################################################################################################################

# Shuffle Data
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

########################################################################################################################

clf = SklearnModels.get_decission_tree_classifier()
# clf = SklearnModels.get_linear_regression_classifier()
# clf = SklearnModels.get_logistic_regression_classifier()
# clf = SklearnModels.get_random_forest_classifier()
# clf = SklearnModels.get_ridge_classifier()
# clf = SklearnModels.get_svm()

########################################################################################################################

clf.fit(x_train, y_train)

print('Mean Score of Val Set:', clf.score(x_val, y_val))
print('Predictions:', clf.predict(x_val).tolist())
