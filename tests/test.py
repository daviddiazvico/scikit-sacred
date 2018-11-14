"""
@author: David Diaz Vico
@license: MIT
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                     train_test_split)
from sklearn.tree import DecisionTreeClassifier

from sksacred import experiment


def test_experiment():
    """Tests experiment."""

    def fetch_dataset(test=False):
        X, y = load_iris(return_X_y=True)
        X_test = y_test = None
        inner_cv = outer_cv = 3
        if test:
            X, X_test, y, y_test = train_test_split(X, y)
        return X, y, X_test, y_test, inner_cv, outer_cv

    def initialize_estimator(X=None, y=None, cv=None):
        return GridSearchCV(DecisionTreeClassifier(), {'max_depth': [2, 4, 8]},
                            iid=True, cv=cv)

    e = experiment(fetch_dataset, initialize_estimator, cross_val_score,
                   attrs=['cv_results_', 'best_score_', 'best_params_',
                          'best_index_', 'n_splits_'])
    for test in (True, False):
        e.run(config_updates={'dataset': {'test': test}})
