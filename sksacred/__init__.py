"""
@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sacred import Experiment, Ingredient
from sklearn.model_selection import cross_validate, PredefinedSplit


def experiment(dataset, estimator):
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

    Returns
    -------
    experiment : Experiment
        Sacred experiment, ready to be run.

    """

    _dataset = Ingredient('dataset')
    dataset = _dataset.capture(dataset)
    _estimator = Ingredient('estimator')
    estimator = _estimator.capture(estimator)
    experiment = Experiment(ingredients=(_dataset, _estimator))

    @experiment.automain
    def run(cross_validate=cross_validate, return_estimator=False):
        """Run the experiment.

        Run the experiment.

        Parameters
        ----------
        cross_validate : function, default=cross_validate
            Function to evaluate metrics by cross-validation. Must receive the
            estimator, X, y (migth be None) and cv (migth be None). Must return
            a dictionary with the cross-validation score and maybe other info,
            like a list of fitted estimators.
        return_estimator : boolean, default False
            Whether to return the estimator or estimators fitted.

        """
        data = dataset()
        for a in ('target', 'data_test', 'target_test', 'inner_cv', 'outer_cv'):
            if a not in data:
                data[a] = None
        if hasattr(data.inner_cv, '__iter__'):
            X = np.array([]).reshape((0, *data.inner_cv[0][0].shape[1:]))
            y = np.array([]).reshape((0, *data.inner_cv[0][1].shape[1:]))
            cv = []
            for i, (X_, y_, X_test_, y_test_) in enumerate(data.inner_cv):
                X = np.concatenate((X, X_, X_test_))
                y = np.concatenate((y, y_, y_test_))
                cv = cv + [-1]*len(X_) + [i]*len(X_test_)
            e = estimator(X=X, y=y, cv=PredefinedSplit(cv))
            e.fit(X, y=y)
            e.fit = e.best_estimator_.fit
        else:
            e = estimator(X=data.data, y=data.target, cv=data.inner_cv)
        if data.data_test is not None:
            e.fit(data.data, y=data.target)
            scores = {'test_score': e.score(data.data_test, y=data.target_test)}
            if return_estimator:
                scores['estimator'] = e
        else:
            if hasattr(data.outer_cv, '__iter__'):
                scores = {'test_score': []}
                estimators = []
                for X, y, X_test, y_test in data.outer_cv:
                    e.fit(X, y=y)
                    if return_estimator:
                        estimators.append(e)
                    scores['test_score'].append(e.score(X_test, y=y_test))
                if return_estimator:
                    scores['estimator'] = estimators
            else:
                scores = cross_validate(e, data.data, y=data.target,
                                        cv=data.outer_cv,
                                        return_estimator=return_estimator)
                if return_estimator:
                    scores['estimator'] = e
        experiment.info.update(scores)

    return experiment
