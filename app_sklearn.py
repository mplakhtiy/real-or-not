# -*- coding: utf-8 -*-
from models import SklearnModels
from utils import get_prepared_data_from_file
from sklearn.model_selection import cross_val_score

PREPARED_DATA_FILE_PATH = './data/prepared_data/count_vectorizer_preprocess_all_true.json'

x_train, y_train, x_val, y_val, data = get_prepared_data_from_file(PREPARED_DATA_FILE_PATH)

clf = SklearnModels.get_decission_tree_classifier()
# clf = SklearnModels.get_linear_regression_classifier()
# clf = SklearnModels.get_logistic_regression_classifier()
# clf = SklearnModels.get_random_forest_classifier()
# clf = SklearnModels.get_ridge_classifier()
# clf = SklearnModels.get_svm()

scores = cross_val_score(clf, x_train, y_train, cv=10, scoring="f1")

print(scores)

clf.fit(x_train, y_train)

print(clf.score(x_val, y_val))
