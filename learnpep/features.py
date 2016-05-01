__description__ = \
"""
Classes for manipulating strings of amino acids to return arrays of float features
for machine learning of peptide classes.
"""
__author__ = "Michael J. Harms"
__date__ = "2016-04-23"

import numpy as np

class SequenceFeature:
    """
    Base class for holding peptide sequence feature(s).
    """
    
    def __init__(self,data_file,seq_length,normalize=True,window_size=1,use_flip_pattern=True,features_to_ignore=None):
        """
        Initialize the class
        """
        
        self._data_file = data_file
        self._seq_length = seq_length
        
        self._normalize = normalize
        self._window_size = window_size
        self._use_flip_pattern = use_flip_pattern
        self._features_to_ignore = np.array(features_to_ignore)

        self._read_aa_data_file()
        
        self._ref_loaded = False


    def _read_aa_data_file(self):
        """
        Read in a data file in whitespace-delimited format.  The top row is assumed to be
        the name of the feature.  Empty lines and lines beginning with # are ignored.  If
        normalize=True, each feature is normalized such that the maximum magnitude value
        is set to one.  For example, a feature ranging from -100 to 10 will be rescaled 
        from -1 to 0.1; a feature ranging from 0 to 100 will be rescaled from 0 to 1.0.
        """
        
        # Read file
        f = open(self._data_file,"r")
        lines = f.readlines()
        f.close()

        lines = [l for l in lines if l.strip() != "" and not l.startswith("#")]
        
        # Grab top line for each feature
        self._base_features = np.array(lines[0].split())

        # Go through lines, populating features for each amino acid
        self._base_feature_dict = {}
        for l in lines[1:]:
            col = l.split()

            aa = col[0]
            self._base__feature_dict[aa] = {}

            for i, p in enumerate(self._base_features):

                if self._features_to_ignore != None:
                    if p in self._features_to_ignore:
                        continue

                try:
                    v = float(col[i+1])
                except ValueError:
                    v = np.NaN

                self._base_feature_dict[aa][p] = v

        # Get rid of features we're supposed to ignore
        if self._features_to_ignore != None:
            keep = np.logical_not(np.in1d(self._base_features,self._features_to_ignore))
            self._base_features = self._base_features[keep]
       
        self._num_base_features = len(self._features)

        if self._normalize:

            # Grab current feature values
            for p in self._base_features:

                feature_values = []
                for aa in self._base_feature_dict.keys():
                    feature_values.append(self._base_feature_dict[aa][p])
                    
                # Normalize to -1 to 1.  
                feature_values = np.array(feature_values)
                feature_values = feature_values/np.nanmax(np.abs(feature_values))
                    
                for i, aa in enumerate(self._base_feature_dict.keys()):
                    self._base_feature_dict[aa][p] = feature_values[i]

        self._window_features = np.array(dtype=str)
        self._use_sliding_windows = False 
        if self._window_size > 0:

            self._use_sliding_windows = True

            self._window_features = []
            num_windows = self._seq_length - self._window_size

            for i in range(len(self._base_features)):
                for j in range(num_windows):
                    feature_names.apppend("{}_w{}".format(self._base_features[i],j))

            self._window_features = np.array(window_features)      
            
        self._pattern_features = np.array(dtype=str)
        if self._use_flip_pattern:
            self._pattern_features = np.array(["{}_flip".format(f) for f in self._base_features])

            
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
        return len(self._base_features) + len(self._window_features) + len(self._pattern_features)
 
    @property
    def features(self):

        return np.concatenate((self._base_features,self._window_features,self._pattern_features)


class SequenceCharge(SequenceFeature):
    """
    Calculate the total charge on a sequence at a given pH.
    """
    
    def __init__(self,data_file,seq_length,window_size=None,pH=7.4,n_term=True,c_term=True):
        """
        Load in the data file and then determine various pH-dependent charge info. 
        """
        
        # Call the parent class
        super(self.__class__,self).__init__(data_file,seq_length,window_size,normalize=False)

        self._n_term = n_term
        self._c_term = c_term
        self._pH = pH

        # Determine the charge on each of these amino acids at this pH
        self.charge_dict = {}
        for aa in self._feature_dict.keys():
            
            pKa = self._feature_dict[aa]["pKa"]
            q = self._feature_dict[aa]["charge"]
            if np.isnan(pKa):
                self.charge_dict[aa] = 0.0
            else:
                X = 10**(q*(self._pH-pKa))
                self.charge_dict[aa] = q*1/(1 + X)

        # Determine whether to treat terminii
        self._term_offset = 0.0
        if self._n_term:
            self._term_offset += self.charge_dict["Nterm"]
            
        if self._c_term:
            self._term_offset += self.charge_dict["Cterm"]
            
        self._features = np.array(["charge"])
    
    def _calc_score(self,seq):
        """
        Return the total charge.
        """
        
        if self._window_size != None:
            
            L = len(seq)
            if L > self._window_size:
                
                charge_window = []
                for i in range(L-self._window_size):
                    charge_window.append(sum(self.charge_dict[s] for s in seq[i:(i+self._window_size)]))
                     
                return np.array(charge_window)
        
        
        return np.array(sum(self.charge_dict[s] for s in seq) + self._term_offset)
        
class SequenceMain(SequenceFeature):
    """
    Sequence features that are a simple sum across amino acids.
    """
    
    def __init__(self,data_file,seq_length,window_size=None,normalize=True,features_to_ignore=("pKa","charge")):
        """
        Load in each feature.
        """
        
        # Call the parent class
        super(self.__class__,self).__init__(data_file,seq_length,window_size,normalize,features_to_ignore)
        
    def _calc_score(self,seq):
        """
        Each feature is simply the sum of the features.
        """
        
        if self._window_size != None:

            L = len(seq)
            if L > self._window_size:
                
                score_window = []
                for i, p in enumerate(self._features):
                    for j in range(L - self._window_size):
                        score_window.append(0)
                        for k in range(self._window_size):
                            score_window[-1] += self._feature_dict[seq[j + k]][p]
        
            self._scores = np.array(score_window)
            return self._scores
        
        self._scores = np.zeros(len(self._features),dtype=float)
        for s in seq:
            for i, p in enumerate(self._features):
                self._scores[i] += self._feature_dict[s][p]
        
        return self._scores
