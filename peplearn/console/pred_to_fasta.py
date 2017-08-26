#!/usr/bin/env python3
__description__ = \
"""
Take a prediction file and map it back to a fasta file.  For each sequence 
in the fasta file, return whether or not it is predicted to bind as well as 
a sequence profile indicating where binding may occur.  
"""
__author__ = "Michael J. Harms"
__date__ = ""
__usage__ = ""

import numpy as np

import sys, argparse

def read_predictions(pred_file):
    """
    Read a predictions file with format:

    SEQUENCE1  pred1
    SEQUENCE2  pred2
    ...        ...
    
    return a dictionary mapping sequence to prediction
    """

    out_dict = {} 
    with open(pred_file) as lines:
        for l in lines:
            if l.strip() == "" or l[0] == "#":
                continue

            col = l.split()

            key = col[0]
            pred = float(col[1])
    
            out_dict[key] = pred

    return out_dict

def predict_binding(sequence,pred_dict,cutoff,kmer_size):
    """
    Predict whether (and where) a sequence should bind.
    """
     
    kmer_list = [sequence[i:(i+kmer_size)] for i in range(len(sequence)-kmer_size+1)]
   
    bind_array = np.zeros((len(kmer_list),len(sequence)),dtype=int)
    bind_array = bind_array + 10

    # Figure out if any of these kmers are predicted to bind
    num_pred_kmers = 0
    best_score = 1000000000000
    for i, k in enumerate(kmer_list):
        try:
            prediction = pred_dict[k]
            if prediction < best_score:
                best_score = prediction
            if prediction <= cutoff:
                prediction = 1  
                num_pred_kmers += 1
            else:
                prediction = 0                             
        except KeyError:
            prediction = -1  

        bind_array[i,i:(i + kmer_size)] = prediction

    # Look for sequence positions that have at least one "no-bind"
    # call from a kmer, then sequence positions that have at least
    # one "bind" from a kmer
    nobind_pos =  np.array(np.sum(bind_array ==  0,0) > 0,dtype=bool)
    binding_pos = np.array(np.sum(bind_array ==  1,0) > 0,dtype=bool)

    # Start with all sites as ambiguous.  Then set non-binding
    # positions.  Then overwrite both ambiguous and non-binding
    # calls with binding calls 
    seq_array = np.array(list(len(sequence)*"?"))

    seq_array[nobind_pos] = "-"
    seq_array[binding_pos] = "+" 
   
    # Count contiguous patches of "+" 
    num_patches = 0 
    current = "?" 
    for s in seq_array:
        if s != current:
            if s == "+":
                num_patches += 1
            current = s

    binds = num_pred_kmers > 0

    meta_info = "META: binds: {}, best_score: {}, num_kmers: {}, num_patches: {}".format(binds,
                                                                                         best_score,
                                                                                         num_pred_kmers,
                                                                                         num_patches)
    pred_string = "".join(seq_array)
    
    out = "{}\n{}\n{}\n".format(meta_info,sequence,pred_string)
    
    return out, binds     


def fasta_pred(fasta_file,pred_file,cutoff=-3,hits_only=False):
    """
    """

    pred_dict = read_predictions(pred_file)

    kmer_size = len(list(pred_dict.keys())[0])

    all_kmers = {}
    seq_name = None 
    current_sequence = []

    out = []

    # Parse fasta file, splitting into kmers and doing preditions as we go
    with open(fasta_file) as lines:
        for l in lines:
            
            if l.startswith(">"):
                if seq_name is not None:
                    sequence = "".join(current_sequence)
                    out_lines, binds = predict_binding(sequence,pred_dict,cutoff,kmer_size)
                    if (hits_only and binds) or (not hits_only):
                        out.append(">{}\n".format(seq_name))
                        out.append(out_lines)
 
                current_sequence = []
                seq_name = l[1:].strip()
            else:
                if seq_name is None or l.strip() == "":
                    continue
                current_sequence.append(l.strip())
    
    if seq_name is not None:
        sequence = "".join(current_sequence)
        out_lines, binds = predict_binding(sequence,pred_dict,cutoff,kmer_size)
        if (hits_only and binds) or (not hits_only):
            out.append(">{}\n".format(seq_name))
            out.append(out_lines)

    return "".join(out)

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=__description__)
    
    # Positionals
    parser.add_argument("fasta_file",help="fasta file on which to apply predictions")
    parser.add_argument("prediction_file",help="prediction file (output of pep_predict")

    # Options 
    parser.add_argument("-o","--outfile",help="output file name",action="store",type=str,default=None)
    parser.add_argument("-g","--hits",help="only spit out hits",action="store_true")
    parser.add_argument("-c","--cutoff",help="cutoff for calling a site as binding",action="store",
                        type=float,default=-3)

    args = parser.parse_args(argv)

    if args.outfile is None:
        out_file = "{}_{}.assembled".format(args.fasta_file,args.prediction_file)
    else:
        out_file = args.outfile

    out = fasta_pred(args.fasta_file,args.prediction_file,args.cutoff,hits_only=args.hits)

    f = open(out_file,'w')
    f.write("# binding model: {}\n".format(args.prediction_file))
    f.write("# binding cutoff: {}\n".format(args.cutoff))
    f.write(out)
    f.close() 


if __name__ == "__main__":
    main()
