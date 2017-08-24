#!/usr/bin/env python3
__description__ = \
"""
Get statistics from a fit model. 
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
    parser.add_argument("trained_pickle",help="pickled trained model")

    # Options 
    parser.add_argument("-f","--final",help="Include final test statistics.  WARNING: if you look at this during fit optimization, you can overfit your data",
                        action="store_true")

    args = parser.parse_args(argv)

    forest = pickle.load(open(args.trained_pickle,"rb"))
   
    out = [] 
    out.append(forest.calc_summary_stats(show_main_fit=args.final))
    out.append(forest.calc_feature_importance())

    sys.stdout.write("".join(out))

if __name__ == "__main__":
    main() 
