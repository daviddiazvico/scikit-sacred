"""
@author: David Diaz Vico
@license: MIT
"""

import pickle
from sacred import Experiment, Ingredient
import tempfile


def experiment(dataset, estimator, cross_validate):
    """Prepare a Scikit-learn experiment as a Sacred experiment.

    Prepare a Scikit-learn experiment indicating a dataset and an estimator and
    return it as a Sacred experiment.

    Parameters
    ----------
    dataset : function
        Dataset fetch function. Might receive any argument. Must return X, y
        (might be None), inner_cv (might be None) and outer_cv (might be None).
    estimator : function
        Estimator initialization function. Must receive at least X, y and cv,
        all of which can be None, and might receive any other argument. Must
        return an initialized sklearn-compatible estimator.
    cross_validate : function
        Function to evaluate metrics by cross-validation. Must receive the
        estimator, X, y (migth be None) and cv (migth be None). Must return a
        dictionary with the cross-validation score and maybe other info.

    Returns
    -------
    experiment : Experiment
        Sacred experiment, ready to be run.

    """

    _dataset = Ingredient('dataset')
    dataset = _dataset.capture(dataset)
    _estimator = Ingredient('estimator')
    estimator = _estimator.capture(estimator)
    _cross_validate = Ingredient('cross_validate')
    cross_validate = _cross_validate.capture(cross_validate)
    experiment = Experiment(ingredients=(_dataset, _estimator, _cross_validate))

    @experiment.automain
    def run():
        """Run the experiment.

        Run the experiment.

        Returns
        -------
        info : dictionary
            Experiment scores and maybe other info.

        """
        X, y, inner_cv, outer_cv = dataset()
        e = estimator(X=X, y=y, cv=inner_cv)
        experiment.info.update(cross_validate(e, X, y=y, cv=outer_cv))
        with tempfile.NamedTemporaryFile('wb') as handler:
            pickle.dump(e.fit(X, y=y), handler)
            experiment.add_artifact(handler.name, name='estimator.pkl')
        return experiment.info

    return experiment
