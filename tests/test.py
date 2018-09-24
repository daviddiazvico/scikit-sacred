"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'tests'))
from sksacred import experiment


def test_cv():
    """Tests scikit-sacred."""
    experiment.run(config_updates={'dataset': {'dataset': 'iris_cv'},
                                   'estimator': {'predictor': 'LogisticRegression'}})


def test_train_test():
    """Tests scikit-sacred."""
    experiment.run(config_updates={'dataset': {'dataset': 'iris_train_test'},
                                   'estimator': {'predictor': 'SVC'}})
