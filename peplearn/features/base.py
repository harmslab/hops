__description__ = \
"""
Classes for manipulating strings of amino acids to return arrays of float features
for machine learning of peptide classes.
"""
__author__ = "Michael J. Harms"
__date__ = "2016-04-23"

import numpy as np

import json, os

class Features:
    """
    Base class for holding peptide sequence feature(s).
    """
    
    def __init__(self,data_files=None,
                      seq_length=12,
                      use_sliding_windows=0,
                      use_flip_pattern=True,
                      features_to_ignore=None):
        """
        Initialize the class

        Parameters:
        -----------
    
        data_files: list of json and/or csv files with information about amino
                    acids.  If None, the json files in features/data/*.json are
                    used.
        seq_length: (int) length of the sequence you'll use for calculations
        use_sliding_windows: (int) max size of sliding windows to use
        use_flip_pattern: (bool) measure pattern flipping 
        features_to_ignore: list of features to ignore
        """
        
        self._data_files = data_files

        self._seq_length = seq_length
        self._use_sliding_windows = use_sliding_windows
        self._use_flip_pattern = use_flip_pattern
        self._features_to_ignore = np.array(features_to_ignore)

        self._compiled = False             
        self._ref_loaded = False
    
        self._base_features = []
        self._base_feature_dict = {}

        # If the user specifies data files...
        if self._data_files is not None:
            if type(self._data_files) == str:
                data_files = [data_files]

            for d in self._data_files:
                if d[-4:] == ".csv":
                    self._read_aa_data_file_csv(d)
                elif d[-5:] == ".json":
                    self._read_aa_data_file_json(d)
                else:
                    err = "Data file type for {} not recongized. should be csv or json\n".format(d)
                    raise ValueError(err) 

        # Otherwise, use the built in data files
        else:
            data_dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(data_dir,"data")
            json_files = [f for f in os.listdir(data_dir) if f[-5:] == ".json"]
            self._data_files = [os.path.join(data_dir,f) for f in json_files]
            for d in self._data_files:
                self._read_aa_data_file_json(d)

        self._compile_features()

    def _read_aa_data_file_json(self,data_file):
        """
        Parse a json file that has the form:

        {
            "feature_name_1": {
                "values": {"A":1.0,
                           "C":0.0,
                            ...}
                },
            "feature_name_2": {
                "values": {"A":0.0,
                           "C":1.0,
                            ...}
                },
        }
    
        Missing values are assigned the average of the non-missing values.
        """

        if self._compiled:
            err = "You cannot add more features after compiling\n"
            raise ValueError(err) 

        data = json.load(open(data_file,'r'))

        for k in data.keys():

            # Make sure we haven't already seen this feature    
            try:
                self._base_feature_dict[k]
                err = "Feature name {} duplicated".format(k)
                raise ValueError(err)
            except:
                pass

            # Populate base_feature_dict
            self._base_feature_dict[k] = {}
            for aa in data[k]["values"].keys():

                if data[k]["values"][aa] == "NA":
                    val = list(data[k]["values"].values())
                    val = [v for v in val if v != "NA"]
                    v = np.mean(val)
                else:
                    v = data[k]["values"][aa] 

                self._base_feature_dict[k][aa] = v


    def _read_aa_data_file_csv(self,data_file):
        """
        Read in a data file in whitespace-delimited format.  The top row is assumed to be
        the name of the feature.  Empty lines and lines beginning with # are ignored.  
        """
       
        if self._compiled:
            err = "You cannot add more features after compiling\n"
            raise ValueError(err) 
 
        # Read file
        f = open(data_file,"r")
        lines = f.readlines()
        f.close()

        lines = [l for l in lines if l.strip() != "" and not l.startswith("#")]
        
        # Grab top line for each feature
        base_features = lines[0].split()

        for k in base_features:

            # Make sure we haven't already seen this feature    
            try:
                self._base_feature_dict[k]
                err = "Feature name {} duplicated".format(k)
                raise ValueError(err)
            except:
                pass

            self._base_feature_dict[k] = {a:0.0 for a in "ACDEFGHIKLMNPQRSTVWY"}

        # Go through lines, populating features for each amino acid
        for l in lines[1:]:
            col = l.split()

            aa = col[0]
            for i, k in enumerate(base_features):

                try:
                    v = float(col[i+1])
                except ValueError:
                    v = np.NaN

                self._base_feature_dict[k][aa] = v

    def _compile_features(self):
        """
        Compile all of the features into a useful calculation.
        """

        # You can only compile once
        if self._compiled:
            return
        self._compiled = True       
 
        # Get base features
        self._base_features = list(self._base_feature_dict.keys())
        self._base_features.sort()
        self._base_features = np.array(self._base_features)

        # Get rid of features we're supposed to ignore
        if self._features_to_ignore is not None:
            keep = np.logical_not(np.in1d(self._base_features,self._features_to_ignore))
            self._base_features = self._base_features[keep]

        # Record the number of base features
        self._num_base_features = len(self._base_features)

        # Deal with sliding windows
        if self._use_sliding_windows > 0:

            # Create masks for running sliding window calculation
            self._window_masks = []
            self._window_addresses = []
            for length in range(1,1 + self._use_sliding_windows):
                for i in range(self._seq_length - length + 1):
                    window = np.zeros(self._seq_length,dtype=bool)
                    window[i:(i+length)] = True

                    self._window_masks.append(window)
                    self._window_addresses.append("pos{}_length{}".format(i,length))

            # Create list of names of sliding window features
            self._window_features = []
            for k in self._base_features:
                for w in self._window_addresses:
                    self._window_features.append("{}_{}".format(k,w))

            self._num_windows = len(self._window_masks)
            self._window_masks = np.array(self._window_masks)
            self._window_addresses = np.array(self._window_addresses)
            self._window_features = np.array(self._window_features)
        else:
            self._num_windows = 0
            self._window_features = np.array([],dtype=str)

        # Whether or not to do flip patterns
        if self._use_flip_pattern:
            self._pattern_features = np.array(["{}_flip".format(f) for f in self._base_features])
        else:
            self._pattern_features = np.array([],dtype=str)

    def _calc_score(self,seq,**kwargs):
        """
        Dummy method. Should be defined for each scoring function.
        """
        
        return np.array(1.0,dtype=float)
    
    def load_ref(self,ref_seq_list):
        """
        Load a reference set of sequences.  Scores will be calculated relative to the average
        for this set. 
        """
        
        scores = np.zeros(len(ref_seq_list),dtype=float)
        for i, s in enumerate(ref_seq_list):
            scores[i] = self._calc_score(s)
    
        self._ref_score = scores/len(scores)
        self._ref_loaded = True
    
    def score(self,seq,**kwargs):
        """
        Calculate the score for a given sequence. 
        """
        
        score = self._calc_score(seq,**kwargs)
        
        if self._ref_loaded:
            score = score - self._ref_score
            
        return score
   
    @property
    def num_features(self):

        # No features if not yet compiled
        if not self._compiled:
            return 0

        return len(self._base_features) + len(self._window_features) + len(self._pattern_features)

    @property
    def features(self):

        # No features if not yet compiled
        if not self._compiled:
            return None

        return np.concatenate((self._base_features,
                               self._window_features,
                               self._pattern_features))


