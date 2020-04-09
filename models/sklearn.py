from sklearn.linear_model import RidgeClassifier, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class SklearnClassifiers:
    classifiers = {
        'RIDGE': RidgeClassifier,
        'LINEAR_REGRESSION': LinearRegression,
        'LOGISTIC_REGRESSION': LogisticRegression,
        'RANDOM_FOREST': RandomForestClassifier,
        'DECISSION_TREE': DecisionTreeClassifier,
        'SVM': SVC
    }

    @staticmethod
    def get_classifier(classifier_type, classifier_options=None):
        return SklearnClassifiers.classifiers[classifier_type](
            **classifier_options if classifier_options is not None else {}
        )
