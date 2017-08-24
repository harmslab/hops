#!/usr/bin/env python3
__description__ = \
"""
Use peplearn to calculate a collection of features for all sequences in a file.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-01"
__usage__ = "calc_features.py sequence_data out_dir [num_threads]"

import peplearn as pl
import os, sys, pickle, argparse

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=__description__)
    
    # Positionals
    parser.add_argument("enrich_file",help="enrichment file from which features will be extracted")

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
        base_file = os.path.split(args.enrich_file)[1]
        data_out_file = "{}_features.pickle".format(base_file)
    else:
        data_out_file = args.outfile

    if os.path.isfile(data_out_file):
        err = "out file '{}' already exists.\n".format(data_out_file)
        raise FileExistsError(err)

    # calculate features
    features = pl.calc_features(args.enrich_file,num_threads=num_threads)

    f = open(data_out_file,"wb")
    pickle.dump(features,f)
    f.close()

if __name__ == "__main__":
    main() 
