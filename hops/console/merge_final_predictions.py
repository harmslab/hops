#!/usr/bin/env python3
__description__ = \
"""
Takes predicted binding for all models on a single proteome and assembles into
a single ascii output file.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-11"
__usage__ = "merge_final_predictions.py proteome_name kmer_directory predictions_directory"

import sys, os

def read_prediction_file(filename):
    """
    Read a prediction output file.
    """
   
    out_dict = {} 
    with open(filename,'r') as infile:
        for line in infile:
            if line.strip() == "": 
                continue
            col = line.split()
            out_dict[col[0]] = float(col[1])

    return out_dict

def read_proteome_kmer_file(filename):
    """
    Read kmer file used to generate proteome features for machine learning.
    """
   
    out_dict = {} 
    with open(filename,'r') as infile:
        for line in infile:
            if line.strip() == "": 
                continue
            col = line.split()
            out_dict[col[0]] = int(col[1])

    return out_dict

def aggregate_predictions(prediction_directory,proteome_name):
    """
    Aggregate all predicted binding into a single dictionary.  Dictionary has
    the form: 

    prediction_out = {model1:{seq1:pred1,seq2:pred2...seqM,predM},
                      model2:{seq1:pred1,seq2:pred2...seqM,predM},
                      modelN:{seq1:pred1,seq2:pred2...seqM,predM}}
    """
   
    prediction_out = {}

    files = os.listdir(prediction_directory)
    for filename in files:

        # Make sure this prediction file came fromt he proteome we're aggregating
        this_proteome_name = filename.split(".fasta_kmers")[0]
        if this_proteome_name != proteome_name:
            continue

        # Figure out what model was used to generate kmers
        parts = filename.split("features.pickle") 
        model = parts[1].split("_")[1]
 
        # Read predictions 
        predictions = read_prediction_file(os.path.join(prediction_directory,
                                                        filename))
        
        try:
            prediction_out[model].update(predictions)
        except KeyError:
            prediction_out[model] = {}
            prediction_out[model].update(predictions)

    return prediction_out 
        
def aggregate_kmer_counts(kmer_directory,proteome_name):
    """
    Aggregate counts for each kmer in proteome from a kmer directory.
    """
   
    kmer_counts_out = {}

    files = os.listdir(kmer_directory)
    for filename in files:

        # Make sure this prediction file came fromt he proteome we're aggregating
        this_proteome_name = filename.split(".fasta_kmers")[0]
        if this_proteome_name != proteome_name:
            continue

        # Read predictions 
        kmer_counts = read_proteome_kmer_file(os.path.join(kmer_directory,
                                                           filename))
       
        kmer_counts_out.update(kmer_counts) 

    return kmer_counts_out
    
def assemble_file(proteome_name,kmer_directory,predictions_directory):
    """
    Create an output file called proteome_name.final that has all sequences as 
    rows, with columns for counts in proteome and for predictions with each
    model.
    """

    predictions = aggregate_predictions(predictions_directory,proteome_name)
    kmer_counts = aggregate_kmer_counts(kmer_directory,proteome_name)

    kmer_list = [(kmer_counts[k],k) for k in kmer_counts.keys()]
    kmer_list.sort(reverse=True)
    kmer_list = [k[1] for k in kmer_list]

    model_columns = list(predictions.keys())
    model_columns.sort()

    out = ["{:>14s}{:>10s}".format("seq","counts")]
    for m in model_columns:
        out.append("{:>10s}".format(m))
    out.append("\n")

    for k in kmer_list:
        out.append("{:>14s}{:10d}".format(k,kmer_counts[k]))
        for m in model_columns:
            out.append("{:10.3f}".format(predictions[m][k]))
        out.append("\n")

    f = open("{}.final".format(proteome_name),"w")
    f.write("".join(out))
    f.close()
 
def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    try:
        proteome_name = argv[0]
        kmer_directory = argv[1]
        predictions_directory = argv[2]
    except IndexError:
        err = "Incorrect arguments. Usage:\n\n{}\n\n".format(__usage__)
        raise IndexError(err)

    assemble_file(proteome_name,kmer_directory,predictions_directory)

if __name__ == "__main__":
    main()
