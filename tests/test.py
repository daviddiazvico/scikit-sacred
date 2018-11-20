"""
@author: David Diaz Vico
@license: MIT
"""

from sacred.observers import FileStorageObserver
from sklearn.datasets import load_iris
from sklearn.model_selection import (cross_validate, GridSearchCV,
                                     train_test_split)
from sklearn.tree import DecisionTreeClassifier

from sksacred import experiment


def test_experiment():
    """Tests experiment."""

    def dataset(inner_cv=None, outer_cv=None):
        data = load_iris()
        if outer_cv is None:
            X, X_test, y, y_test = train_test_split(data.data, data.target)
            data.data = X
            data.target = y
            data.data_test = X_test
            data.target_test = y_test
            data.outer_cv = None
        else:
            data.data_test = data.target_test = None
            data.outer_cv = outer_cv
        data.inner_cv = inner_cv
        return data

    def estimator(X=None, y=None, cv=None):
        return GridSearchCV(DecisionTreeClassifier(), {'max_depth': [2, 4, 8]},
                            iid=True, cv=cv)

    e = experiment(dataset, estimator, cross_validate)
    e.observers.append(FileStorageObserver.create('.results'))
    e.run(config_updates={'dataset': {'inner_cv': 3, 'outer_cv': 3}})
    e.run(config_updates={'dataset': {'inner_cv': 3, 'outer_cv': None}})
