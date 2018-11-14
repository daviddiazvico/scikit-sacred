"""
@author: David Diaz Vico
@license: MIT
"""

import pickle
from sacred import Experiment, Ingredient
import tempfile


def experiment(fetch_dataset, initialize_estimator, cross_val_score,
               attrs=None):
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
        Estimator initialization function. Must receive at least X, y and cv,
        all of which can be None, and might receive any other argument. Must
        return an initialized sklearn-compatible estimator.
    cross_val_score : function
        Function to perform cross-validation scoring on the estimator. Must
        receive the estimator, X, y and cv (migth be None). Must return the CV
        score.
    attrs : iterable, default=None
        Tuple or list of names of estimator attributes to be saved as experiment
        info, or None.

    Returns
    -------
    experiment : Experiment
        Sacred experiment, ready to be run.

    """

    dataset = Ingredient('dataset')
    fetch_dataset = dataset.capture(fetch_dataset)
    estimator = Ingredient('estimator')
    initialize_estimator = estimator.capture(initialize_estimator)
    experiment = Experiment(ingredients=(dataset, estimator))

    @experiment.automain
    def run():
        """Run the experiment.

        Run the experiment.

        Returns
        -------
        info : dictionary
            Experiment info.

        """
        X, y, X_test, y_test, inner_cv, outer_cv = fetch_dataset()
        estimator = initialize_estimator(X=X, y=y, cv=inner_cv)
        if X_test is not None:
            estimator.fit(X, y=y)
            experiment.info['score'] = estimator.score(X_test, y=y_test)
        else:
            experiment.info['score'] = cross_val_score(estimator, X, y=y,
                                                       cv=outer_cv)
            estimator.fit(X, y=y)
        if attrs is not None:
            for attr in attrs:
                if hasattr(estimator, attr):
                    experiment.info[attr] = estimator.__dict__[attr]
        handler = tempfile.NamedTemporaryFile('wb')
        pickle.dump(estimator, handler)
        experiment.add_artifact(handler.name, name='estimator.pkl')
        return experiment.info

    return experiment
