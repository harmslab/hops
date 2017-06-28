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
    
    def __init__(self,data_file,seq_length,normalize=True,use_sliding_windows=True,use_flip_pattern=True,features_to_ignore=None):
        """
        Initialize the class
        """
        
        self._data_file = data_file
        self._seq_length = seq_length
        
        self._normalize = normalize
        self._use_sliding_windows = use_sliding_windows
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
            self._base_feature_dict[aa] = {}

            for i, p in enumerate(self._base_features):

                if self._features_to_ignore is not None:
                    if p in self._features_to_ignore:
                        continue

                try:
                    v = float(col[i+1])
                except ValueError:
                    v = np.NaN

                self._base_feature_dict[aa][p] = v

        # Get rid of features we're supposed to ignore
        if self._features_to_ignore is not None:
            keep = np.logical_not(np.in1d(self._base_features,self._features_to_ignore))
            self._base_features = self._base_features[keep]
 
        self._num_base_features = len(self._base_features)

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

        if self._use_sliding_windows:

            self._num_windows = np.sum(np.arange(self._seq_length,0,-1))

            # Loop overall features
            window_features = []
            for i in range(self._num_base_features):
        
                # Loop over all possible window sizes
                window_features.append([])
                for j in range(self._seq_length):

                    # Loop over all possible window start positions
                    for k in range(self._seq_length - j):
                        window_features[-1].append("{}_size{}_pos{}".format(self._base_features[i],(j+1),k))

            self._window_features = np.array(window_features)

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

        if self._use_sliding_windows:
            if self._use_flip_pattern:
                return len(self._base_features) + len(np.ravel(self._window_features)) + len(self._pattern_features)
            else:
                return len(self._base_features) + len(np.ravel(self._window_features))
        else:
            if self._use_flip_pattern:            
                return len(self._base_features) + len(self._pattern_features)
 
        return len(self._base_features)

 
    @property
    def features(self):

        if self._use_sliding_windows:
            if self._use_flip_pattern:
                return np.concatenate((self._base_features,
                                       np.ravel(self._window_features),
                                       self._pattern_features))
            else:
                return np.concatenate((self._base_features,
                                       np.ravel(self._window_features)))
        else:
            if self._use_flip_pattern:            
                return np.concatenate((self._base_features,
                                       self._pattern_features))
 
        return self._base_features
 

class SequenceCharge(SequenceFeature):
    """
    Calculate the total charge on a sequence at a given pH.
    """
    
    def __init__(self,data_file,seq_length,
                 normalize=True,use_sliding_windows=True,use_flip_pattern=True,
                 pH=7.4,n_term=True,c_term=True):
        """
        Load in the data file and then determine various pH-dependent charge info. 
        """
        
        # Call the parent class
        super(self.__class__,self).__init__(data_file,seq_length,
                                            normalize=normalize,
                                            use_sliding_windows=use_sliding_windows,
                                            use_flip_pattern=use_flip_pattern)

        self._n_term = n_term
        self._c_term = c_term
        self._pH = pH

        # Determine the charge on each of these amino acids at this pH
        self.charge_dict = {}
        for aa in self._base_feature_dict.keys():
            
            pKa = self._base_feature_dict[aa]["pKa"]
            q = self._base_feature_dict[aa]["charge"]
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
        
        total = np.array([sum(self.charge_dict[s] for s in seq) + self._term_offset])

        if self._use_sliding_windows:

            windows = []
            # loop over possible sliding window lengths
            for i in range(self._seq_length):
         
                # loop over possible start points for window 
                for j in range(self._seq_length - i):
                    windows.append(sum(self.charge_dict[s] for s in seq[j:(j+i)]))
        
             
            total = np.concatenate((total,np.array(windows)))
        
        if self._use_flip_pattern:
          
            flips = np.array([0.0])
            for i in range(1,self._seq_length):
                if abs(self.charge_dict[seq[i]]) != abs(self.charge_dict[seq[i-1]]):
                    flips[0] += 1

            total = np.concatenate((total,flips))
      
        return total
     
        
class SequenceMain(SequenceFeature):
    """
    Sequence features that are a simple sum across amino acids.
    """
    
    def __init__(self,data_file,seq_length,normalize=True,use_sliding_windows=True,
                 use_flip_pattern=True,features_to_ignore=("pKa","charge")):
        """
        Load in each feature.
        """
        
        # Call the parent class
        super(self.__class__,self).__init__(data_file,seq_length,
                                            normalize=normalize,
                                            use_sliding_windows=use_sliding_windows,
                                            use_flip_pattern=use_flip_pattern,
                                            features_to_ignore=features_to_ignore)

    def _calc_score(self,seq):
        """
        Each feature is simply the sum of the features.
        """

        if len(seq) != self._seq_length:
            err = "Sequence length {} does not match length used to initialize class ({})\n".format(len(seq),self._seq_length)
            raise ValueError(err)

        totals = np.zeros(self._num_base_features,dtype=float)
        for s in seq:
            for i, f in enumerate(self._base_features):
                totals[i] += self._base_feature_dict[s][f]

        if self._use_sliding_windows:

            windows = np.zeros((self._num_base_features,self._num_windows),dtype=float)
            for i, f in enumerate(self._base_features):

                # loop over possible sliding window lengths
                window_counter = 0
                for j in range(self._seq_length):

                    # loop over possible start points for window 
                    for k in range(self._seq_length - j):
                        windows[i,window_counter] = sum(self._base_feature_dict[s][f] for s in seq[k:(k+j)])
                        window_counter += 1

            totals = np.concatenate((totals,np.ravel(windows)))

        if self._use_flip_pattern:
       
            flip_scores = np.zeros(self._num_base_features,dtype=float) 
            for i, f in enumerate(self._base_features):

                f_over_seq = np.array([self._base_feature_dict[s][f] for s in seq])
                f_flips = f_over_seq <= np.mean(f_over_seq)

                # See if the element before each element in the vector is the same as the previous 
                flip_scores[i] = np.sum(f_flips[1:self._seq_length] == f_flips[0:(self._seq_length-1)])/(self._seq_length/2)

            totals = np.concatenate((totals,flip_scores))

        return totals
 
