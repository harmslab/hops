#!/usr/bin/env python3
__description__ = \
"""
Use peplearn and sklearn to train a random forest model against a
pre-calculated feature file.  If the user specifies specifies category
breaks, use a classifier.  Otherwise, use a continuous regression 
model.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-23"

import peplearn as pl

import sklearn

# Suppress Future and Deprecation warning
sklearn.warnings.filterwarnings(action="ignore",category=DeprecationWarning)
sklearn.warnings.filterwarnings(action="ignore",category=FutureWarning)
from sklearn import ensemble

#In the future, can be modified to use basically any sklearn model  
#from sklearn import neural_network, svm, naive_bayes

import os, sys, pickle, argparse

def main(argv=None):

    parser = argparse.ArgumentParser(description=__description__)
    
    # Positionals
    parser.add_argument("feature_pickle",help="pickled feature file used to train model")

    # Options 
    parser.add_argument("-o","--outdir",help="output directory",action="store",type=str,default=".")
    parser.add_argument("-n","--numcpu",help="number of cpus to use.  if -1, use all cpus",action="store",type=int,default=-1)
    parser.add_argument("-w","--weight",help="weight regression according to weights in the feature file",action="store_true")
    parser.add_argument("-e","--estimators",help="number of estimators to use",action="store",type=int,default=40)
    parser.add_argument("-b","--break", 
                        help="add category break. specify multiple --break to train for multiple categories. if no breaks are specified, model will be continuous",
                        type=float,
                        action="append",
                        dest="breaks",
                        default=[])

    args = parser.parse_args(argv)

    if len(args.breaks) == 0:
        fit_type = "regressor"
    else:
        fit_type = "classifier"

    # Figure out number of threads
    if args.numcpu == -1:
        num_threads = os.cpu_count()
    else:
        num_threads = args.numcpu

    # Figure out what kinds of weights to use
    weight_type = "even"
    if args.weight:
        weight_type = "file"

    # Create output directory (if necessary)
    try:
        os.mkdir(args.outdir)
    except FileExistsError:
        pass

    # Figure out output files
    training_file = os.path.split(args.feature_pickle)[1]
    model_out_file = os.path.join(args.outdir,"{}_model.pickle".format(training_file))

    if os.path.isfile(model_out_file):
        err = "out file '{}' already exists.\n".format(model_out_file)
        raise FileExistsError(err)

    features = pickle.load(open(args.feature_pickle,'rb'))

    # Choose model type.  In the future, could include a bunch of possible models
    if fit_type == "classifier":
        features.add_classes(args.breaks)
        model = ensemble.RandomForestClassifier(n_estimators=args.estimators,
                                                n_jobs=num_threads) 
        #model = neural_network.MLPClassifier()
        #model = svm.SVC() #gamma=2, C=1)
        #model = naive_bayes.GaussianNB()
        #model = ensemble.AdaBoostClassifier()
        #model = svm.SVC(gamma=2, C=1)
    else:
        model = ensemble.RandomForestRegressor(n_estimators=args.estimators,
                                               n_jobs=num_threads) 
        #model = ensemble.GradientBoostingRegressor(n_estimators=100) 
        #model = svm.SVR(kernel='rbf',C=1e-3,gamma=0.1) 
        
    # Train models
    forest = pl.MachineLearner(model,fit_type)
    forest.train(features,weights=weight_type,kfold=True)

    # Write out model pickle
    f = open(model_out_file,"wb")
    pickle.dump(forest,f)
    f.close()

if __name__ == "__main__":
    main() 
