from sklearn.linear_model import RidgeClassifier, LinearRegression, LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class Sklearn:
    VECTORIZERS = {
        'TFIDF': TfidfVectorizer,
        'COUNT': CountVectorizer
    }

    CLASSIFIERS = {
        'RIDGE': RidgeClassifier,
        'LINEAR_REGRESSION': LinearRegression,
        'LOGISTIC_REGRESSION': LogisticRegression,
        'RANDOM_FOREST': RandomForestClassifier,
        'DECISION_TREE': DecisionTreeClassifier,
        'SVM': SVC,
        'SGDClassifier': SGDClassifier
    }
