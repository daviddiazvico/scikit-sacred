"""
@author: David Diaz Vico
@license: MIT
"""

from sacred import Experiment, Ingredient


def experiment(dataset, estimator, cross_validate):
    """Prepare a Scikit-learn experiment as a Sacred experiment.

    Prepare a Scikit-learn experiment indicating a dataset and an estimator and
    return it as a Sacred experiment.

    Parameters
    ----------
    dataset : function
        Dataset fetch function. Might receive any argument. Must return a Bunch
        with data, target (might be None), inner_cv (might be None) and outer_cv
        (might be None).
    estimator : function
        Estimator initialization function. Must receive at least X, y and cv,
        all of which can be None, and might receive any other argument. Must
        return an initialized sklearn-compatible estimator.
    cross_validate : function
        Function to evaluate metrics by cross-validation. Must receive the
        estimator, X, y (migth be None) and cv (migth be None). Must return a
        dictionary with the cross-validation score and maybe other info, like
        a list of fitted estimators.

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
        """Run the experiment."""
        data = dataset()
        e = estimator(X=data.data, y=data.target, cv=data.inner_cv)
        scores = cross_validate(e, data.data, y=data.target, cv=data.outer_cv)
        experiment.info.update(scores)

    return experiment
