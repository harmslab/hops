#!/usr/bin/env python3
__description__ = \
"""
Predict binding from a pre-calculated feature file and trained peplearn model.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-23"

import sklearn

# Suppress Future and Deprecation warning
sklearn.warnings.filterwarnings(action="ignore",category=DeprecationWarning)
sklearn.warnings.filterwarnings(action="ignore",category=FutureWarning)

import os, sys, pickle, arparse

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=__description__)
    
    # Positionals
    parser.add_argument("feature_pickle",help="pickled feature file used to train model")
    parser.add_argument("trained_pickle",help="pickled trained model")

    # Options 
    parser.add_argument("-o","--outdir",help="output directory",action="store",type=str,default=".")
    parser.add_argument("-n","--numcpu",help="number of cpus to use.  if -1, use all cpus",action="store",type=int,default=-1)

    args = parser.parse_args(argv)

    # Figure out number of threads
    if args.numcpu == -1:
        num_threads = os.cpu_count()
    else:
        num_threads = args.numcpu

    # Create output directory (if necessary)
    try:
        os.mkdir(args.outdir)
    except FileExistsError:
        pass

    # Figure out output files
    feature_base = os.path.split(feature_pickle)[1]
    trained_model_base = os.path.split(trained_pickle)[1]

    predictions_file = os.path.join(out_dir,"{}_{}_predictions.txt".format(feature_base,trained_model_base))

    if os.path.isfile(predictions_file):
        err = "out file '{}' already exists.\n".format(predictions_file)
        raise FileExistsError(err)

    forest = pickle.load(open(trained_pickle,"rb"))

    # Calculate features
    features = pickle.load(open(feature_pickle,"rb"))
    features.add_classes([-2,0])
    
    # Make predictions
    predictions = forest.predict(features)
    seqs = list(predictions.keys())
    seqs.sort()
    
    f = open(predictions_file,'w')
    for s in seqs:
        f.write("{} {}\n".format(s,predictions[s]))
    f.close()

if __name__ == "__main__":
    main() 
