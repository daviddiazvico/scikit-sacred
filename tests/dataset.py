"""
@author: David Diaz Vico
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def select_dataset(dataset, **kwargs):
    X, y = load_iris(return_X_y=True)
    if dataset == 'iris_cv':
        return X, y, None, None, None, None
    if dataset == 'iris_train_test':
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, y_train, X_test, y_test, None, None
