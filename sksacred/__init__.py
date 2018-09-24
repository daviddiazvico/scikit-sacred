"""
@author: David Diaz Vico
"""

import pickle
from sacred import Experiment, Ingredient
from sklearn.model_selection import BaseCrossValidator, cross_val_score
import tempfile

from dataset import select_dataset
from estimator import select_estimator


dataset = Ingredient('dataset')
select_dataset = dataset.capture(select_dataset)
estimator = Ingredient('estimator')
select_estimator = estimator.capture(select_estimator)
experiment = Experiment(ingredients=[dataset, estimator])


@experiment.automain
def run(persist=False):
    """ Run an experiment. """
    X, y, X_test, y_test, inner_cv, outer_cv = select_dataset()
    estimator = select_estimator(X=X, y=y, cv=inner_cv)
    if (X_test is not None) and (y_test is not None):
        # Test split for scoring
        estimator.fit(X, y)
        score = estimator.score(X_test, y_test)
    else:
        # CV scoring
        if isinstance(inner_cv, BaseCrossValidator) or hasattr(inner_cv, '__iter__'):
            # Validation split for hyperparameter tuning
            estimator.fit(X, y)
            score = cross_val_score(estimator.best_estimator_, X, y,
                                    cv=outer_cv)
        else:
            # Random CV folds for hyperparameter tuning
            score = cross_val_score(estimator, X, y, cv=outer_cv)
        if persist:
            estimator.fit(X, y)
    # Record score and other attributes as experiment info
    experiment.info['score'] = score
    for attr in ['cv_results_', 'best_score_', 'best_params_', 'best_index_',
                 'n_splits_']:
        if hasattr(estimator, attr):
            experiment.info[attr] = estimator.__dict__[attr]
    if persist:
        # Store the model
        handler = tempfile.NamedTemporaryFile('wb')
        pickle.dump(estimator, handler)
        experiment.add_artifact(handler.name, name='estimator.pkl')
    return score
