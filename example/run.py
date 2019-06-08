#! /usr/bin/env python

"""
@author: David Diaz Vico
@license: MIT
"""

import argparse
import numpy as np
from sacred.observers import FileStorageObserver
from skdatasets import fetch as dataset
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sksacred import experiment


def estimator(predictor, X=None, y=None, cv=None, **kwargs):
    """ Initialize an estimator. """
    d = y.shape[1] if len(y.shape) >= 2 else 1
    estimator = {'LinearSVC': GridSearchCV(Pipeline([('sc', StandardScaler(with_mean=False)), ('cls', LinearSVC())]),
                                                    {'cls__C': np.logspace(20, -30, num=16, base=2.0)},
                                                    cv=cv, error_score=np.nan, **kwargs),
                 'SVC': GridSearchCV(Pipeline([('sc', StandardScaler(with_mean=False)), ('cls', SVC())]),
                                     {'cls__C': np.logspace(20, -30, num=4, base=2.0),
                                      'cls__gamma': np.logspace(-3, 6, num=4, base=2.0) / d},
                                     cv=cv, error_score=np.nan, **kwargs)}
    return estimator[predictor]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('-r', '--repository', type=str, help='repository')
    parser.add_argument('-c', '--collection', type=str, default=None, help='collection')
    parser.add_argument('-d', '--dataset', type=str, default=None, help='dataset')
    parser.add_argument('-p', '--predictor', type=str, help='predictor')
    args = parser.parse_args()
    e = experiment(dataset, estimator, cross_validate)
    e.observers.append(FileStorageObserver.create('.results'))
    e.run(config_updates={'dataset': {'repository': args.repository,
                                      'collection': args.collection,
                                      'dataset': args.dataset},
                          'estimator': {'predictor': args.predictor}})
