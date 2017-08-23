#!/usr/bin/env python3

from . import Observations, features

import numpy as np

import sklearn

# Suppress Future and Deprecation warning
sklearn.warnings.filterwarnings(action="ignore",category=DeprecationWarning)
sklearn.warnings.filterwarnings(action="ignore",category=FutureWarning)

from sklearn import ensemble
from sklearn.metrics import roc_curve, auc

import sys, inspect

def calc_features(sequence_data,use_flip_pattern=True,use_sliding_windows=12,
                  num_threads=1):
    """
    Calculate the features of a dataset.

    Parameters:
    sequence_data: a file with a collection of sequences with line format
        sequence value [weight]
    use_flip_pattern: bool.  whether or not to calculate vector of sign flip
                      for each feature slong the sequence
    use_sliding_windows: int. max size of sliding windows to employ.
    num_threads: number of threads to use for the calculation.

    Returns an Observations instance with calculated features.
    """

    print("Constructing feature set.")
    sys.stdout.flush()

    # Create observations object
    obs = Observations(sequence_data)

    # Append features on which to train
    simple_features = features.SimpleFeatures(use_flip_pattern=use_flip_pattern,
                                              use_sliding_windows=use_sliding_windows)
    cider_features = features.CiderFeatures(use_sliding_windows=bool(use_sliding_windows))

    obs.add_features(simple_features)
    obs.add_features(cider_features)

    # Do the calculation
    print("Calculating features on {} threads.".format(num_threads))
    sys.stdout.flush()

    obs.calc_features(num_threads)

    return obs

