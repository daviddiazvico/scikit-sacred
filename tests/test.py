"""
@author: David Diaz Vico
@license: MIT
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sksacred import experiment


def test_experiment():
    """Tests experiment."""

    def dataset(inner_cv=None, outer_cv=None):
        X, y = load_iris(return_X_y=True)
        return X, y, inner_cv, outer_cv

    def estimator(X=None, y=None, cv=None):
        return GridSearchCV(DecisionTreeClassifier(), {'max_depth': [2, 4, 8]},
                            iid=True, cv=cv)

    e = experiment(dataset, estimator, cross_validate)
    e.run(config_updates={'dataset': {'inner_cv': 3, 'outer_cv': 3}})
