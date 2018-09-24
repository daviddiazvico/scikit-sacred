"""
@author: David Diaz Vico
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def select_estimator(predictor, X=None, y=None, cv=None, **kwargs):
    """ Select an estimator. """
    scoring = make_scorer(accuracy_score)
    error_score = np.nan
    d = y.shape[1] if len(y.shape) >= 2 else 1
    estimator = {'LogisticRegression': GridSearchCV(Pipeline([('scaler', StandardScaler()),
                                                              ('classifier', LogisticRegression())]),
                                                    {'classifier__C': np.logspace(10, -30, num=8, base=2.0)},
                                                    scoring=scoring, cv=cv,
                                                    error_score=error_score,
                                                    **kwargs),
                 'SVC': GridSearchCV(Pipeline([('scaler', StandardScaler()),
                                               ('classifier', SVC())]),
                                     {'classifier__gamma': np.logspace(-3, 6, num=8, base=2.0) / d},
                                     scoring=scoring, cv=cv,
                                     error_score=error_score, **kwargs)}
    return estimator[predictor]
