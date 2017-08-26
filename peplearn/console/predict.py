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

import os, sys, pickle, argparse

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=__description__)
    
    # Positionals
    parser.add_argument("feature_pickle",help="pickled feature file against which to perform predictions")
    parser.add_argument("trained_pickle",help="pickled trained model")

    # Options 
    parser.add_argument("-o","--outfile",help="output file",action="store",type=str,default=None)
    parser.add_argument("-n","--numcpu",help="number of cpus to use.  if -1, use all cpus",action="store",type=int,default=-1)

    args = parser.parse_args(argv)

    # Figure out number of threads
    if args.numcpu == -1:
        num_threads = os.cpu_count()
    else:
        num_threads = args.numcpu

    # Figure out output files
    if args.outfile is None:
        feature_base = os.path.split(args.feature_pickle)[1]
        trained_model_base = os.path.split(args.trained_pickle)[1]
        predictions_file = os.path.join(args.outdir,"{}_{}_predictions.txt".format(feature_base,trained_model_base))
    else:
        predictions_file = args.outfile

    if os.path.isfile(predictions_file):
        err = "out file '{}' already exists.\n".format(predictions_file)
        raise FileExistsError(err)

    forest = pickle.load(open(args.trained_pickle,"rb"))

    # Calculate features
    features = pickle.load(open(args.feature_pickle,"rb"))
    
    # Make predictions
    predictions = forest.predict(features)
    seqs = list(predictions.keys())
    seqs.sort()
    
    f = open(predictions_file,'w')

    if forest.fit_type == "classifier":

        for i in range(len(forest._obs.breaks)+1):
        
            if i == 0:
                f.write("# class {:2d}:           E <= {:7.3f}\n".format(i,forest._obs.breaks[i]))
            elif i == len(forest._obs.breaks):
                f.write("# class {:2d}: {:7.3f} < E\n".format(i,forest._obs.breaks[i-1]))
            else:
                f.write("# class {:2d}: {:7.3f} < E <= {:7.3f}\n".format(i,
                                                             forest._obs.breaks[i-1],
                                                             forest._obs.breaks[i]))
    

    for s in seqs:

        if predictions[s][0].is_integer():
            fmt_string = "{} {:12d}{:12.3f}\n"
            f.write(fmt_string.format(s,int(predictions[s][0]),predictions[s][1]))
        else:
            fmt_string = "{} {:12.3f}{:12.3f}\n"
            f.write(fmt_string.format(s,predictions[s][0],predictions[s][1]))
    
    f.close()

if __name__ == "__main__":
    main() 
