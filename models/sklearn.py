from sklearn.linear_model import RidgeClassifier, LinearRegression, LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class Sklearn:
    vectorizers = {
        'TFIDF': TfidfVectorizer,
        'COUNT': CountVectorizer
    }

    classifiers = {
        'RIDGE': RidgeClassifier,
        'LINEAR_REGRESSION': LinearRegression,
        'LOGISTIC_REGRESSION': LogisticRegression,
        'RANDOM_FOREST': RandomForestClassifier,
        'DECISSION_TREE': DecisionTreeClassifier,
        'SVM': SVC,
        'SGDClassifier': SGDClassifier
    }

    @staticmethod
    def get_classifier(classifier_type, classifier_options=None):
        return Sklearn.classifiers[classifier_type](
            **classifier_options if classifier_options is not None else {}
        )

    @staticmethod
    def get_vectorizer(vectorizer_type, vectorizer_options=None):
        return Sklearn.vectorizers[vectorizer_type](
            **vectorizer_options if vectorizer_options is not None else {}
        )
