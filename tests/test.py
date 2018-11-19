"""
@author: David Diaz Vico
@license: MIT
"""

from sacred.observers import FileStorageObserver
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sksacred import experiment


def test_experiment():
    """Tests experiment."""

    def dataset(inner_cv=None, outer_cv=None):
        data = load_iris()
        data.inner_cv = inner_cv
        data.outer_cv = outer_cv
        return data

    def estimator(X=None, y=None, cv=None):
        return GridSearchCV(DecisionTreeClassifier(), {'max_depth': [2, 4, 8]},
                            iid=True, cv=cv)

    e = experiment(dataset, estimator, cross_validate)
    e.observers.append(FileStorageObserver.create('.results'))
    e.run(config_updates={'dataset': {'inner_cv': 3, 'outer_cv': 3}})
