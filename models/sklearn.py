from sklearn.linear_model import RidgeClassifier, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class SklearnModels:
    @staticmethod
    def get_ridge_classifier():
        return RidgeClassifier()

    @staticmethod
    def get_linear_regression_classifier():
        return LinearRegression()

    @staticmethod
    def get_logistic_regression_classifier():
        return LogisticRegression()

    @staticmethod
    def get_random_forest_classifier():
        return RandomForestClassifier()

    @staticmethod
    def get_decission_tree_classifier():
        return DecisionTreeClassifier()

    @staticmethod
    def get_svm():
        return SVC()
