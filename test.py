"""
@author: David Diaz Vico
@license: MIT
"""

import itertools
from sklearn.datasets import load_iris
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.tree import DecisionTreeClassifier

from sksacred import sklearn_experiment


def _dataset(val=False, test=False):
    X, y = load_iris(return_X_y=True)
    X_test = y_test = None
    inner_cv = outer_cv = 3
    if val:
        inner_cv = StratifiedKFold(n_splits=3)
    if test:
        X, X_test, y, y_test = train_test_split(X, y)
    return X, y, X_test, y_test, inner_cv, outer_cv


def _estimator(X=None, y=None, cv=None):
    return GridSearchCV(DecisionTreeClassifier(), {'max_depth': [2, 4, 8]},
                        iid=True, cv=cv)


def test_sklearn_experiment():
    """Tests sklearn_experiment."""
    experiment = sklearn_experiment(_dataset, _estimator, cross_val_score)
    for val, test in itertools.product((True, False), (True, False)):
        experiment.run(config_updates={'dataset': {'val': val, 'test': test}})
