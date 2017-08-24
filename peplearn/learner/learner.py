__description__ = \
"""
This holds the main class that holds features and fits models.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-23"

from . import decompose_model

from sklearn.metrics import roc_curve, auc
import numpy as np
import inspect

import sys, copy

class MachineLearner:
    """
    Main user-accessed class for controlling a peplearn calculation.
    """

    def __init__(self,sk_model,fit_type):
        """
        sk_model is either an instance of an SKlearn model or the class itself.
        If passed as a class, the model is used with default parameters.

        fit_type: "regressor" or "classifier"
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

    def train(self,obs,weights="even",kfold=False):
        """
        Train the model.

        Parameters:
        -----------
        obs: an Observations instance with features already calculated.
        weights: array that is the same length as the number of values, for 
                 weighting observations
        kfold: bool. whether or not to do k-fold training
        """

        self._obs = obs
        self._weight_type = weights
        self._kfold = kfold

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
      
        # Find standardization for features features
        self._standardization_mean = np.mean(self._obs.training_features,0)
        f = self._obs.training_features - self._standardization_mean
        self._standardization_scalar = np.std(f,0)

        # If the standard deviation of the feature is 0, set it to inf.  This
        # will give the feature a final standard deviation of 0 --> meaning it
        # does not contribute to the final fit.  This is good because a standard
        # deviation of zero means the feature does not change across samples.
        self._standardization_scalar[self._standardization_scalar == 0] = np.inf
        self._standardization_scalar = 1.0/self._standardization_scalar

        sys.stdout.write("Performing main fit.\n")
        sys.stdout.flush()

        # Create a new model with exactly the same paramters as the input model
        self._main_model = type(self._model)(**self._model.get_params())
        self._fit_result = self._do_fit(self._main_model,
                                        self.training_features,
                                        self.training_values,
                                        self._weights)

        sys.stdout.write("Performing k-fold fits.\n")
        sys.stdout.flush()
        self._k_models = []
        self._k_fit_result = []
        if self._kfold:
            for i in range(self._obs.kfold_size - 1):
              
                sys.stdout.write("     k-fold fit {} of {}.\n".format(i+1,self._obs.kfold_size-1))
                sys.stdout.flush()

                # Create a new instance of the model with exactly the same
                # parameters as the input model
                self._k_models.append(type(self._model)(**self._model.get_params()))
                features = self._standardize(self._obs.get_k_training_features(i))
                values   =                   self._obs.get_k_training_values(i)
                weights  =                   self._obs.get_k_training_weights(i)
                if self._weight_type == "even":
                    weights = self._weights[0:len(weights)] 

                self._k_fit_result.append(self._do_fit(self._k_models[-1],features,values,weights))
    
        self._is_trained = True

    def _standardize(self,some_feature_array):
        """
        Standardize features according to total dataset.
        """

        return (some_feature_array - self._standardization_mean)*self._standardization_scalar

    def _do_fit(self,model,features,values,weights):
        """
        Actually do the fit.  Should only be called internally.
        """

        # Train the model 
        try:
            result = model.fit(features,values,sample_weight=weights) 
        except TypeError:
            # not all sklearn models accept a weight term
            result = model.fit(features,values)

        return result

    def predict(self,obs):
        """
        Given an Observations instance with features and a trained model, 
        predict the value for all observations.

        Parameters:
        -----------
        obs: observations instance with calculated features.
        
        Returns a dictionary of predictions.
        """

        if not self._is_trained:
            err = "You must train the model before doing a prediction.\n"
            raise ValueError(err)

        features = self._standardize(obs.features)

        model = self._main_model
        main_predictions = self._main_model.predict(features)

        if self._kfold:

            k_preds = []
            for i in range(self._obs.kfold_size - 1):

                k_preds.append(self._k_models[i].predict(features))

            k_preds = np.array(k_preds,dtype=float)

            print(k_preds)
            print(main_predictions)

            if self._fit_type == "regressor":
                # standard deviation on prediction across k-fold replicates
                k_err = np.std(k_preds,0)
            else:
                # fraction of time the k-fold replicates got the same answer 
                # as the main fit
                k_err = np.sum(k_preds == main_predictions,0)/k_preds.shape[0] 

 
            out = {obs.sequences[i]:(main_predictions[i],k_err[i]) for i in range(len(main_predictions))}
        else:
            out = {obs.sequences[i]:(main_predictions[i],0.0) for i in range(len(main_predictions))}
            
 
        return out 

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

        return self._standardize(self._obs.training_features)

    @property
    def test_features(self):
        """
        Standardized features for testing.
        """

        return self._standardize(self._obs.test_features)

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

    def calc_kfold_roc_curve(self):
        """
        Calculate a reciever operator characterstic curve for each of the 
        k models.
        """
      
        # Only meaningful for a classifier model 
        if self._fit_type != "classifier":
            return None

        out = []

        if not self._kfold:
            err = "calc_kfold_roc_curve requires k-fold be used.\n"
            raise ValueError(err)

        kfold_pct = []
        kfold_roc = []
        kfold_auc = []
        for i in range(self._obs.kfold_size - 1):
            model = self._k_models[i]
            test_features = self._standardize(self._obs.get_k_test_features(i))
            test_values = self._obs.get_k_test_values(i)

            pct_correct, roc_results, auc_results = self._indiv_roc(model,
                                                                    test_features,
                                                                    test_values)

            kfold_pct.append(pct_correct)
            kfold_roc.append(roc_results)
            kfold_auc.append(auc_results)

        # Take mean and standard deviation of pct
        mean_pct = np.mean(kfold_pct)
        std_pct  = np.std(kfold_pct)
        final_pct = (mean_pct,std_pct)

        breaks = self._obs.breaks
        num_classes = len(breaks) + 1

        final_roc = []
        final_auc = []
        for i in range(num_classes):

            final_roc.append([[0,1],[0,1]])
            final_auc.append([])
            for j in range(self._obs.kfold_size - 1):

                break_roc = kfold_roc[j][i]

                # Skip curves where mid is at 1,1
                if not (break_roc[0][1] == 1 and break_roc[1][1] == 1):
                    final_roc[-1][0].append(break_roc[0][1])
                    final_roc[-1][1].append(break_roc[1][1])

                break_auc = kfold_auc[j][i]
                final_auc[-1].append(break_auc)

            # Sort ROC curve 
            roc_together = list(zip(final_roc[-1][0],final_roc[-1][1]))
            roc_together.sort()

            final_roc[-1] = roc_together

            # Take mean and standard deviation of AUC
            mean_auc = np.mean(final_auc[-1])
            std_auc  = np.std(final_auc[-1])
            final_auc[-1] = (mean_auc,std_auc)

        out.append("# {}\n".format(44*"-"))
        out.append("# k-fold classifier statistics\n")
        out.append("# {}\n".format(44*"-"))

        out.append("Percent correct: {:8.3f} +/- {:8.3f}\n".format(100*final_pct[0],
                                                                   100*final_pct[1]))
        out.append("\n")

        # Now deal with area under curve stats
        out.append("Area under ROC curve:\n")
        breaks = self._obs.breaks
        num_classes = len(breaks) + 1
        for i in range(num_classes):

            if i == 0:
                label = "            E <={:8.3f}".format(breaks[0])
            elif i == (num_classes - 1):
                label = "            E > {:8.3f}".format(breaks[-1])
            else:
                label = "{:8.3f} <= E < {:8.3f}".format(breaks[i-1],breaks[i])

            result = final_auc[i][0]
            result_std = final_auc[i][1]
            result_key = "{:.1f}".format(np.floor(10*result))
            result_dict = {"5.0":"failed",
                           "6.0":"poor",
                           "7.0":"fair",
                           "8.0":"good",
                           "9.0":"excellent"}

            result_call = result_dict[result_key]

            out.append("{:30s}: {:8.3f} +/- {:8.3f}-> {}\n".format(label,
                                                                   100*result,
                                                                   100*result_std,
                                                                   result_call))
        out.append("\n")

        # Spit out ROC curve proper
        out.append("Reciever Operating Characteristic curve\n")
        out.append("{:20s}{:>14s}{:>14s}\n".format("class","false_rate","pos_rate"))
        for i in range(num_classes):

            if i == 0:
                label = "E<={:.3f}".format(breaks[0])
            elif i == (num_classes - 1):
                label = "E>{:.3f}".format(breaks[-1])
            else:
                label = "{:.3f}<=E<{:.3f}".format(breaks[i-1],breaks[i])

            for j in range(len(final_roc[i])):
                out.append("{:20s}{:14.5f}{:14.5f}\n".format(label,final_roc[i][j][0],final_roc[i][j][1]))
                 
        out.append("\n")

        return "".join(out)

    def calc_main_roc_curve(self):
        """
        Calculate ROC curve and statistics for the main model.
        """

        out = []

        model = self._main_model 
        pct_correct, roc_results, auc_results = self._indiv_roc(model,
                                                                self.test_features, 
                                                                self.test_values)

        out.append("# {}\n".format(44*"-"))
        out.append("# final classifier statistics\n")
        out.append("# {}\n".format(44*"-"))

        out.append("Percent correct: {:8.3f}\n".format(100*pct_correct))

        out.append("\n")

        # Now deal with area under curve stats
        out.append("Area under ROC curve:\n")
        breaks = self._obs.breaks
        num_classes = len(breaks) + 1
        for i in range(num_classes):

            if i == 0:
                label = "            E <={:8.3f}".format(breaks[0])
            elif i == (num_classes - 1):
                label = "            E > {:8.3f}".format(breaks[-1])
            else:
                label = "{:8.3f} <= E < {:8.3f}".format(breaks[i-1],breaks[i])

            result = auc_results[i]
            result_key = "{:.1f}".format(np.floor(10*result))
            result_dict = {"5.0":"failed",
                           "6.0":"poor",
                           "7.0":"fair",
                           "8.0":"good",
                           "9.0":"excellent"}

            result_call = result_dict[result_key]

            out.append("{:30s}: {:8.3f} -> {}\n".format(label,
                                                        100*result,
                                                        result_call))
        out.append("\n")

        # Spit out ROC curve proper
        out.append("Reciever Operating Characteristic curve\n")
        out.append("{:20s}{:>14s}{:>14s}\n".format("class","false_rate","pos_rate"))
        for i in range(num_classes):

            if i == 0:
                label = "E<={:.3f}".format(breaks[0])
            elif i == (num_classes - 1):
                label = "E>{:.3f}".format(breaks[-1])
            else:
                label = "{:.3f}<=E<{:.3f}".format(breaks[i-1],breaks[i])

            for j in range(len(roc_results[i][0])):
                out.append("{:20s}{:14.5f}{:14.5f}\n".format(label,roc_results[i][0][j],roc_results[i][1][j]))
                 
        out.append("\n")
 
        return "".join(out)

    def _indiv_roc(self,model,test_features,test_values):

        # Calculate the predicted versus real values 
        y_calc = model.predict(test_features)
        y_obs = test_values

        # Determine the total percent correct
        pct_correct = sum(y_calc == y_obs)/len(y_obs)

        # Go through each class and calculate an AUC curve
        breaks = self._obs.breaks
        num_classes = len(breaks) + 1
        roc_results = []
        auc_results = []
        for i in range(num_classes):

            true_calc =  y_obs == i
            pred_calc = y_calc == i

            a, b, c = roc_curve(true_calc,pred_calc)

            roc_results.append((a,b))
            auc_results.append(auc(a,b))

        return pct_correct, roc_results, auc_results

    def calc_kfold_r2(self):
        """
        Calculate R^2 for k-fold models.
        """

        out = []

        out.append("# {}\n".format(44*"-"))
        out.append("# k-fold summary statistics\n")
        out.append("# {}\n".format(44*"-"))

        if not self._kfold:
            err = "calc_kfold_r2 requires kfold model\n"
            raise ValueError(err)

        r2_train = []
        r2_test = []            
        for i in range(self._obs.kfold_size - 1):

            model = self._k_models[i]

            train_features = self._standardize(self._obs.get_k_training_features(i))    
            train_values   =                   self._obs.get_k_training_values(i)    

            test_features = self._standardize(self._obs.get_k_test_features(i))    
            test_values   =                   self._obs.get_k_test_values(i)    

            r2_train.append(100*model.score(train_features,train_values))
            r2_test.append(100*model.score(test_features,test_values))


        out.append("r2_train: {:8.3f} +/- {:8.3f}\n".format(np.mean(r2_train),
                                                            np.std(r2_train)))
        out.append("r2_test:  {:8.3f} +/- {:8.3f}\n".format(np.mean(r2_test),
                                                            np.std(r2_test)))
        out.append("\n")
    

        return "".join(out)

    def calc_main_r2(self):
        """
        Calculate R^2 for main model.
        """

        out = []

        out.append("# {}\n".format(44*"-"))
        out.append("# final summary statistics\n")
        out.append("# {}\n".format(44*"-"))

        # R^2 for test and training set
        r2_train = self._main_model.score(self.training_features,self.training_values)
        r2_test = self._main_model.score(self.test_features,self.test_values)

        out.append("r2_train: {:8.3f}\n".format(100*r2_train))
        out.append("r2_test:  {:8.3f}\n".format(100*r2_test))
        out.append("\n")

        return "".join(out)

    def calc_feature_importance(self):
        """
        Return a report describing the importance of the features of the model
        for discrimniating peptides.
        """

        out = []

        # Feature importance in final model.  copy prevents the @property method
        # from running over and over and slowing script down
        importance = np.copy(self._main_model.feature_importances_)

        feature_dict = {}
        for i in range(len(importance)):
            feature_dict[self._obs.feature_names[i]] = importance[i] 

        out.append(decompose_model.summary(feature_dict))

        return "".join(out)

    def calc_summary_stats(self,show_main_fit=False):
        """
        Return a report with summary statistics for the model.
        """

        out = []

        # Make sure the training has been done
        if not self._is_trained:
            out.append("Model has not yet been trained.\n\n")
            return "".join(out)

        # Write main fit statistics if requested
        if show_main_fit:
            out.append(self.calc_main_r2())
            if self._fit_type == "classifier":
                out.append(self.calc_main_roc_curve())
    
        # Write kfold fit statistics if relevant
        if self._kfold:
            out.append(self.calc_kfold_r2())
            if self._fit_type == "classifier":
                out.append(self.calc_kfold_roc_curve())

        return "".join(out)
