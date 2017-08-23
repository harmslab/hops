#!/usr/bin/env python3
__description__ = \
"""
Summarize results for model that describe specificity.
"""
__author__ = "Michael J. Harms"
__date__ = ""
__usage__ = ""

import numpy as np

import sys, json

def read_log_file(log_file):
    """
    Read a log file that has feature importances. 
    """
    
    f = open(log_file,'r')
    lines = f.readlines()
    f.close()

    out_dict = {}
    for l in lines:
        if ":" in l or l.strip() == "":
            continue
   
        col = l.split()
        key = col[0]
        value = float(col[1])

        out_dict[key] = value 
        
    return out_dict

def load_categories_json(json_file):
    """
    Load json file describing the categories of each parameter.
    """

    categories = json.load(open(json_file))["categories"]
    
    out_dict = {} 
    for k in categories.keys():
        for p in categories[k]:
            out_dict[p] = k

    return out_dict

def merge_windows(feature_dict):
    """
    Merge sliding windows to calculate total feature importance across the 
    entire sequence.
    """

    total = 0

    out_dict = {}
    for k in feature_dict.keys():

        # Merge flips and _pos sliding windows
        key = k.split("_pos")
        if len(key) == 1:
            if "_flip" in key[0]:
                key = key[0].split("_flip")[0]
            else:
                key = k 
        else:
            key = key[0]

        try:
            out_dict[key] += feature_dict[k]
        except KeyError:
            out_dict[key] = feature_dict[k]

        total += feature_dict[k]

    for k in out_dict.keys():
        out_dict[k] = out_dict[k]/total
    
    return out_dict

def merge_positions(feature_dict):
    """
    Calculate total importance of all features in feature_dict across all
    positions in the sequence.
    """
   
    # Figure out sequence length 
    max_size = 0 
    for k in feature_dict.keys():

        key = k.split("_length")
        if len(key) == 2:
            size = int(key[1])
            if size > max_size:
                max_size = size
           
    # Calculate total contributions of weight at each position along the
    # sequence 
    positions = np.zeros(max_size,dtype=np.float)
    degen = np.zeros(max_size,dtype=np.float)
    for k in feature_dict.keys():
        
        key = k.split("_pos")
        if len(key) == 1:
            positions[:] += feature_dict[k] 
            degen[:] += 1.0
        else:
            indexes = key[1].split("_length")
            bottom = int(indexes[0])
            top = int(indexes[1]) + bottom

            positions[bottom:top] += feature_dict[k]
            degen[bottom:top] += 1.0

    positions = positions/degen

    positions = positions/np.sum(positions)
   
    return positions 

def merge_categories(feature_dict,json_file=None):
    """
    Merge features into categories.  Keep track of positional information
    if this is still part of the features.
    """
   
    categories = load_categories_json("x.json")
 
    out_dict = {} 
    for k in feature_dict.keys():

        if "_pos" in k:
            to_merge = k.split("_pos")
            category = categories[to_merge[0]]
            out_key = "{}_pos{}".format(category,to_merge[1])

            try:
                out_dict[out_key] += feature_dict[k]
            except KeyError:
                out_dict[out_key] =  feature_dict[k]

            continue

        if "_flip" in k:
            to_merge = k.split("_flip")
            out_key = categories[to_merge[0]]

            try:
                out_dict[out_key] += feature_dict[k]
            except KeyError:
                out_dict[out_key] =  feature_dict[k]
           
            continue 

        out_key = categories[k]
        try:
            out_dict[out_key] += feature_dict[k]
        except KeyError:
            out_dict[out_key]  = feature_dict[k]
      
   
    return out_dict 

def print_report(features,title=""):
    """
    """

    # Construct header
    out = []
    out.append("# {}\n".format((76*"-")))
    if title != "":
        out.append("# {}\n".format(title))
    else:
        out.append("# Model report\n")
    out.append("# {}\n".format((76*"-")))

    # Construct table
    out.append("{:30s}{:10s}\n".format("feature","importance"))

    # Sort by importance (if dict)
    if type(features) == dict:
   
        out_dict = features
 
        ordered = []
        for k in out_dict.keys():
            ordered.append((out_dict[k],k))

        ordered.sort(reverse=True)
        key_list = [o[1] for o in ordered]

    # Or keep sorting (if not dict)
    else:
        out_dict = dict([("{}".format(i),v) for i,v in enumerate(features)])
        key_list = ["{}".format(i) for i in range(len(features))]
        
    for k in key_list:
        out.append("{:30s}{:10.5f}\n".format(k,100*out_dict[k]))
        
    print("".join(out),end="")


def summary(log_file,json_file):
    """
    """

    feature_dict = read_log_file(log_file)
   
    total_features = merge_windows(feature_dict)
    print_report(total_features,"total feature importance")

    total_categories = merge_categories(total_features,json_file)
    print_report(total_categories,"total category importance")

    pos_importance = merge_positions(feature_dict) 
    print_report(pos_importance,"total position importance") 

    cat_pos_summary = {}
    all_categories = list(total_categories.keys())
    for categ in all_categories:
        local_categories = merge_categories(feature_dict,json_file)
        local_categ = [k for k in local_categories.keys() if categ in k]
        local_categ = {k:local_categories[k] for k in local_categ}
        categ_pos_importance = merge_positions(local_categ)
        cat_pos_summary[categ] = categ_pos_importance

        print_report(categ_pos_importance,"{} position importance".format(categ))


    categ = list(cat_pos_summary.keys())
    categ.sort()
   
    out = ["{:>13s}".format("position")]
    out.extend(["{:>13s}".format(c) for c in categ])
    out.append("\n")

    size = len(cat_pos_summary[categ[0]])
    for i in range(size):
        out.append("{:13d}".format(i))
        for c in categ:
            out.append("{:13.5}".format(100*total_categories[c]*cat_pos_summary[c][i]))
        out.append("\n") 
        
    print("".join(out),end="")

 
def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    try:
        log_file = argv[0]
    except IndexError:
        err = "Incorrect arguments. Usage:\n\n{}\n\n".format(__usage__)
        raise IndexError(err)

    summary(log_file,"x.json")

if __name__ == "__main__":
    main()
