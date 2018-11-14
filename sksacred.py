"""
@author: David Diaz Vico
@license: MIT
"""

import pickle
from sacred import Experiment, Ingredient
from sklearn.model_selection import BaseCrossValidator, cross_val_score
import tempfile


def _score(estimator, X, y=None, X_test=None, y_test=None, inner_cv=None,
           outer_cv=None, persist=False):
    """Score an estimator."""
    if (X_test is not None) and (y_test is not None):
        estimator.fit(X, y)
        score = estimator.score(X_test, y_test)
    else:
        if isinstance(inner_cv, BaseCrossValidator) or hasattr(inner_cv, '__iter__'):
            estimator.fit(X, y)
            score = cross_val_score(estimator.best_estimator_, X, y, cv=outer_cv)
        else:
            score = cross_val_score(estimator, X, y, cv=outer_cv)
        if persist:
            estimator.fit(X, y)
    info = dict()
    info['score'] = score
    for attr in ['cv_results_', 'best_score_', 'best_params_', 'best_index_', 'n_splits_']:
        if hasattr(estimator, attr):
            info[attr] = estimator.__dict__[attr]
    return estimator, info


def sklearn_experiment(fetch_dataset, initialize_estimator):
    """Prepare a Scikit-learn experiment as a Sacred experiment.

    Prepare a Scikit-learn experiment indicating a dataset and an estimator and
    return it as a Sacred experiment.

    Parameters
    ----------
    fetch_dataset : function
        Dataset fetch function. Might receive any argument. Must return X, y
        (might be None), X_test (might be None), y_test (might be None),
        inner_cv (might be None), outer_cv (might be None).
    initialize_estimator : function
        Estimator initialization function. Might receive at least X, y and cv,
        all of which can be None, and any other argument. Must return an
        initialized sklearn-compatible estimator.

    Returns
    -------
    experiment : Experiment
        Sacred experiment, ready to be run.

    """

    dataset = Ingredient('dataset')
    fetch_dataset = dataset.capture(fetch_dataset)
    estimator = Ingredient('estimator')
    initialize_estimator = estimator.capture(initialize_estimator)
    experiment = Experiment(ingredients=[dataset, estimator])

    @experiment.automain
    def run(persist=False):
        """Run the experiment.

        Run the experiment.

        Parameters
        ----------
        persist : boolean, default=False
            Flag indicating if the trained estimator should be saved to file.

        Returns
        -------
        score : float or array of float
            Experiment info.

        """
        X, y, X_test, y_test, inner_cv, outer_cv = fetch_dataset()
        estimator, info = _score(initialize_estimator(X=X, y=y, cv=inner_cv),
                                 X, y=y, X_test=X_test, y_test=y_test,
                                 inner_cv=inner_cv, outer_cv=outer_cv,
                                 persist=persist)
        experiment.info.update(info)
        if persist:
            handler = tempfile.NamedTemporaryFile('wb')
            pickle.dump(estimator, handler)
            experiment.add_artifact(handler.name, name='estimator.pkl')
        return info

    return experiment
