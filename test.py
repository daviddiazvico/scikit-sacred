"""
@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sksacred import sklearn_experiment


def fetch_dataset(dataset):
    X, y = load_iris(return_X_y=True)
    X_test = y_test = None
    inner_cv = outer_cv = 3
    if dataset == 'test':
        X, X_test, y, y_test = train_test_split(X, y)
    if dataset == 'validation':
        inner_cv = StratifiedKFold(n_splits=2)
    return X, y, X_test, y_test, inner_cv, outer_cv


def initialize_estimator(predictor, X=None, y=None, cv=None):
    estimator = {'lr': GridSearchCV(Pipeline([('tr', StandardScaler()),
                                              ('cls', LogisticRegression(solver='lbfgs',
                                                                         multi_class='auto'))]),
                                    {'cls__C': np.logspace(10, -30, num=8, base=2.0)},
                                    scoring=make_scorer(accuracy_score),
                                    iid=True, cv=cv, error_score=np.nan)}
    return estimator[predictor]


def test_sklearn_experiment():
    """Tests sklearn_experiment."""
    experiment = sklearn_experiment(fetch_dataset, initialize_estimator,
                                    cross_val_score)
    for dataset in ('', 'validation', 'test'):
        experiment.run(config_updates={'dataset': {'dataset': dataset},
                                       'estimator': {'predictor': 'lr'},
                                       'persist': True})
