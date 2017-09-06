#!/usr/bin/env python3
__description__ = \
"""
Summarize results for model that describe specificity.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-23"

import numpy as np
import os

def _load_categories(cat_file=None):
    """
    cat_file: string or None.  If string, file containing categories. If None,
    load the built in categories.  Category file should have the form:

    feature     cat1    cat2    cat3 ...
    a_feature   1.0     0.0     0.0  ...
    another_f   0.2     0.2     0.6  ...
    ...         ...     ...     ...  ...

    Returns a dictionary in which features are mapped to categories.  Each 
    category is given a float value between 0 and 1, representing the relative
    contribution of that feature to that category.  
    """

    # Grab default file
    if cat_file is None:
        data_dir = os.path.dirname(os.path.realpath(__file__))
        cat_file = os.path.join(data_dir,"..","features","data",
                                "feature-to-category.txt")
   
    # Read file 
    f = open(cat_file,'r')
    lines = f.readlines()
    f.close()

    lines = [l for l in lines if l.strip() != "" and l[0] != "#"]

    # Grab category names
    header = lines[0]
    col = header.split() 
    categories = col[1:]

    # Now load feature -> value maps
    out_dict = {}
    for l in lines[1:]:
        col = l.split()

        feature = col[0]
        values = [float(c) for c in col[1:]]

        out_dict[feature] = {categories[i]:values[i] for i in range(len(values))}
        
    return out_dict

def _merge_windows(feature_dict):
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

def _merge_positions(feature_dict):
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

def _merge_categories(feature_dict):
    """
    Merge features into categories.  Keep track of positional information
    if this is still part of the features.
    """
  
    category_dict = _load_categories()
 
    out_dict = {} 
    for k in feature_dict.keys():

        # Deal with positionally oriented features correctly
        if "_pos" in k:
            to_merge = k.split("_pos")
            categories = category_dict[to_merge[0]]
    
            for c in categories.keys():

                if categories[c] == 0.0:
                    continue

                out_key = "{}_pos{}".format(c,to_merge[1])

                try:
                    out_dict[out_key] += feature_dict[k]*categories[c]
                except KeyError:
                    out_dict[out_key] =  feature_dict[k]*categories[c]

            continue

        # Flatten flipped features
        if "_flip" in k:
            to_merge = k.split("_flip")
            categories = category_dict[to_merge[0]]

            for c in categories.keys():

                if categories[c] == 0.0:
                    continue

                out_key = c

                try:
                    out_dict[out_key] += feature_dict[k]*categories[c]
                except KeyError:
                    out_dict[out_key] =  feature_dict[k]*categories[c]

            continue 

       
        # Deal with stand-alone features 
        categories = category_dict[k]
        for c in categories.keys():

            if categories[c] == 0.0:
                continue

            out_key = c

            try:
                out_dict[out_key] += feature_dict[k]*categories[c]
            except KeyError:
                out_dict[out_key] =  feature_dict[k]*categories[c]

    return out_dict 

def _compile_report(features,title="",xlabel=""):
    """
    Compile a pretty report given a set of features.
    """

    if title == "":
        title = "Model report"

    if xlabel == "":
        xlabel = "feature"

    # Construct header
    out = []
    out.append("# {}\n".format(44*"-"))
    out.append("# {}\n".format(title))
    out.append("# {}\n".format(44*"-"))

    # Construct table
    out.append("{:36s}{:10s}\n".format(xlabel,"importance"))

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
        out.append("{:36s}{:10.5f}\n".format(k,100*out_dict[k]))

    out.append("\n")
    return "".join(out)

def _category_by_position_table(category_position_importance,category_importance):

    categories = list(category_position_importance.keys())
    categories.sort()
    
    out = []
    out.append("# {}\n".format(44*"-"))
    out.append("# {}\n".format("Category vs. position breakdown"))
    out.append("# {}\n".format(44*"-"))
    
    out.append("{:>13s}".format("position"))
    out.extend(["{:>13s}".format(c) for c in categories])
    out.append("\n")

    size = len(category_position_importance[categories[0]])
    for i in range(size):
        out.append("{:13d}".format(i))
        for c in categories:
            out.append("{:13.5f}".format(100*category_importance[c]*category_position_importance[c][i]))
        out.append("\n") 
    
    out.append("\n")
    return "".join(out) 

def summary(feature_dict):
    """
    Summarize the importance of the features and categories for the model.
    """

    out = []

    # Calcualte the importance of each position (integrated over features) 
    position_importance = _merge_positions(feature_dict)

    # Calculate the importance of each feature (integrated over positions)
    feature_importance  = _merge_windows(feature_dict)

    # Calculate the importance of each feature category (integrated over 
    # positions)
    category_importance = _merge_categories(feature_importance)

    # Calculate the importance of each feature category at each position
    category_position_importance = {}

    # Collapse features into categories, keeping position information
    local_categories = _merge_categories(feature_dict)
    for category in category_importance.keys():

        # Grab only features matching this category
        this_category_at_positions = {}
        for c in local_categories.keys():  
            if c.find(category) != -1:
                this_category_at_positions[c] = local_categories[c]

        # Calculate the importance of each position 
        category_position_importance[category] = _merge_positions(this_category_at_positions)

    # Category by position breakdown
    out.append(_category_by_position_table(category_position_importance,
                                           category_importance))


    # Category, position, feature importance
    out.append(_compile_report(category_importance,"total category importance",
                               xlabel="category"))

    out.append(_compile_report(position_importance,"total position importance",
                               xlabel="position"))

    out.append(_compile_report(feature_importance,"total feature importance"))

    # For each category, write report.  Sort categories from most to least 
    # important
    #all_categories = [(category_importance[k],k) for k in category_importance.keys()]
    #all_categories.sort(reverse=True)
    #all_categories = [c[1] for c in all_categories]

    #for category in all_categories:
    #    out.append(_compile_report(category_position_importance[category],
    #                               "{} position importance".format(category),
    #                               xlabel="position"))

    return "".join(out)
