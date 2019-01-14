"""
@author: David Diaz Vico
@license: MIT
"""

from functools import partial
from sacred.observers import FileStorageObserver
from sklearn.datasets import load_boston
from sklearn.model_selection import (cross_validate, GridSearchCV,
                                     train_test_split)
from sklearn.tree import DecisionTreeRegressor

from sksacred import experiment


def test_experiment():
    """Tests experiment."""

    def dataset(inner_cv=None, outer_cv=None):
        data = load_boston()
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
        return GridSearchCV(DecisionTreeRegressor(), {'max_depth': [2, 4, 8]},
                            iid=True, cv=cv)

    cv = partial(cross_validate, pre_dispatch=1)
    e = experiment(dataset, estimator)
    e.observers.append(FileStorageObserver.create('.results'))
    e.run(config_updates={'dataset': {'inner_cv': 3, 'outer_cv': 3}, 'cross_validate': cv})
    e.run(config_updates={'dataset': {'inner_cv': 3, 'outer_cv': None}, 'cross_validate': cv})
    data = load_boston()
    e.run(config_updates={'dataset': {'inner_cv': [[data.data[:10], data.target[:10], data.data[10:20], data.target[10:20]],
                                                   [data.data[10:20], data.target[10:20], data.data[20:30], data.target[20:30]],
                                                   [data.data[20:30], data.target[20:30], data.data[30:40], data.target[30:40]]],
                                      'outer_cv': 3},
                          'cross_validate': cv})
    e.run(config_updates={'dataset': {'inner_cv': 3,
                                      'outer_cv': [[data.data[:10], data.target[:10], data.data[10:20], data.target[10:20]],
                                                   [data.data[10:20], data.target[10:20], data.data[20:30], data.target[20:30]],
                                                   [data.data[20:30], data.target[20:30], data.data[30:40], data.target[30:40]]]},
                          'cross_validate': cv})
    e.run(config_updates={'dataset': {'inner_cv': [[data.data[:10], data.target[:10], data.data[10:20], data.target[10:20]],
                                                   [data.data[10:20], data.target[10:20], data.data[20:30], data.target[20:30]],
                                                   [data.data[20:30], data.target[20:30], data.data[30:40], data.target[30:40]]],
                                      'outer_cv': [[data.data[:10], data.target[:10], data.data[10:20], data.target[10:20]],
                                                   [data.data[10:20], data.target[10:20], data.data[20:30], data.target[20:30]],
                                                   [data.data[20:30], data.target[20:30], data.data[30:40], data.target[30:40]]]},
                          'cross_validate': cv})

