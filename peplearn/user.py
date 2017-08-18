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

class MachineLearner:
    """
    """

    def __init__(self,sk_model):
        """
        sk_model is either an instance of an SKlearn model or the clas itself.
        If passed as a class, the model is used with default parameters.
        """

        self._model = sk_model   
        if inspect.isclass(self._model):
            self._model = self._model()

        # Model not yet trained
        self._is_trained = False

    def train(self,obs,weights="even"):
        """
        Train the model.

        Parameters:
        -----------
        obs: an Observations instance with features already calculated.
        weights: array that is the same length as the number of values, for 
                 weighting observations
        """

        self._obs = obs
        self._weight_type = weights

        # Parse the weights
        if self._weight_type is None:
            self._weights = np.ones(len(self._obs.training_values),dtype=int)
        elif self._weight_type == "even":
            self._weights = np.ones(len(self._obs.training_values),dtype=float)/len(self._obs.training_values)
        elif self._weight_type == "file":
            self._weights = self._obs.training_weights/np.sum(self._obs.training_weights)
        else:
            err = "'weights' should be 'even' or 'file'.\n"
            raise ValueError(err)
       
        # Standardize the training features
        self._standardization_mean = np.mean(self._obs.training_features,0)
        self._training_features = self._obs.training_features - self._standardization_mean

        self._standardization_scalar = 1/np.std(self._training_features,0)

        # If the standard deviation of the value is NaN, the value does not change
        # across the training set.  Thus, set to zero to avoid numerical problems.
        self._standardization_scalar[np.isinf(self._standardization_scalar)] = 0.0

        self._training_features = self._training_features*self._standardization_scalar
        self._test_features = (self._obs.test_features -  self._standardization_mean)*self._standardization_scalar

        # Train the model 
        try:
            self._fit_result = self._model.fit(self._training_features, self._obs.training_values,sample_weight=self._weights) 
        except TypeError:
            # not all sklearn models accept a weight term
            self._fit_result = self._model.fit(self._training_features, self._obs.training_values)
    
        self._is_trained = True

    def predict(self,obs):
        """
        Given an Observations instance with features and a trained model, 
        predict the value for all observations.

        Parameters:
        -----------
        obs: observations instance with calculated features.
        model: trained random model model.
        
        Returns a dictionary of predictions.
        """

        try:
            self._model
        except AttributeError:
            err = "You must train the model before doing a prediction.\n"
            raise ValueError(err) 

        # Standardize input features
        features = obs.features - self._standardization_mean
        features = features*self._standardization_scalar

        predictions = self._model.predict(features)
   
        out = {obs.sequences[i]:predictions[i] for i in range(len(predictions))}
 
        return out 

    @property
    def log(self):

        self._log = []
        #try:
        #
        #   y_score = self._fit_result.predict(self._test_features)

        #    a, b, c = roc_curve(self._obs.test_values,y_score)
        #    area_under_curve = auc(a,b)
        #    self._log.append("auc:      {:6.3f}".format(area_under_curve))

        #except AttributeError:
        #    pass

        try:
            r2_train = self._model.score(self._training_features,self._obs.training_values)
            r2_test = self._model.score(self._test_features,self._obs.test_values)

            self._log.append("r2_train: {:6.3f}".format(r2_train))
            self._log.append("r2_test:  {:6.3f}".format(r2_test))

            order = np.argsort(self._model.feature_importances_)
            order = order[::-1]
            for i in order:
                self._log.append("{:>25s}{:6.2f}".format(self._obs.feature_names[i],
                                                         100*self._model.feature_importances_[i]))


        except AttributeError:
            pass

        self._log = "\n".join(self._log)

        return self._log

    @property
    def model(self):

        if not self._is_trained:
            return None
        
        try:
            return self._model
        except AttributeError:
            return None

