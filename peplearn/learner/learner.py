__description__ = \
"""
This holds the main class that holds features and fits models.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-23"

from sklearn.metrics import roc_curve, auc
import numpy as np
import inspect

class MachineLearner:
    """
    Main user-accessed class.
    """

    def __init__(self,sk_model,fit_type):
        """
        sk_model is either an instance of an SKlearn model or the clas itself.
        If passed as a class, the model is used with default parameters.
        """

        self._model = sk_model   
        if inspect.isclass(self._model):
            self._model = self._model()

        if fit_type not in ["regressor","classifier"]:
            err = "fit_type {} is not recognized\n".format(fit_type)
            raise ValueError(err)
        self._fit_type = fit_type

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

        self._standardization_scalar = np.std(self._training_features,0)

        # If the standard deviation of the feature is 0, set it to inf.  This
        # will give the feature a final standard deviation of 0 --> meaning it
        # does not contribute to the final fit.  This is good because a standard
        # deviation of zero means the feature does not change across samples.
        self._standardization_scalar[self._standardization_scalar == 0] = np.inf

        self._standardization_scalar = 1.0/self._standardization_scalar

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

        if not self._is_trained:
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

        # Make sure the training has been done
        if not self._is_trained:
            self._log.append("Model has not yet been trained.\n")
            return "\n".join(self._log)

        # ROC curve
        try:
            y_score = self._fit_result.predict(self._test_features)
            a, b, c = roc_curve(self._obs.test_values,y_score)
            area_under_curve = auc(a,b)
            self._log.append("auc:      {:6.3f}".format(area_under_curve))

        except:
            self._log.append("Could not calculate ROC curve.\n")
    
        # R^2 for test and training set
        try:
            r2_train = self._model.score(self._training_features,self._obs.training_values)
            r2_test = self._model.score(self._test_features,self._obs.test_values)

            self._log.append("r2_train: {:6.3f}".format(r2_train))
            self._log.append("r2_test:  {:6.3f}".format(r2_test))
        except:
            self._log.append("Could not calculate fit statistics\n")

        # Feature importance in final model
        try:
            order = np.argsort(self._model.feature_importances_)
            order = order[::-1]
            for i in order:
                self._log.append("{:>25s}{:6.2f}".format(self._obs.feature_names[i],
                                                         100*self._model.feature_importances_[i]))
        except:
            self._log.append("Could not calculate feature importances.\n")
        
        self._log = "\n".join(self._log)

        return self._log

    @property
    def model(self):
        """
        sklearn model.
        """ 

        return self._model

    @property
    def fit_type(self):
        """
        Type of fit.
        """
        
        return self._fit_type

    @property
    def training_features(self):
        """
        Standardized features for training.
        """

        return self._training_features

    @property
    def test_features(self):
        """
        Standardized features for testing.
        """

        return self._test_features

    @property
    def training_values(self):
        """
        Values used for training.
        """

        return self._obs.training_values

    @property
    def test_values(self):
        """
        Values used for testing.
        """

        return self._obs.test_values


