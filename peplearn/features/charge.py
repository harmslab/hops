__description__ = \
"""
Classes for manipulating strings of amino acids to return arrays of float features
for machine learning of peptide classes.
"""
__author__ = "Michael J. Harms"
__date__ = "2016-04-23"

from .base import SequenceFeature
import numpy as np


class SequenceCharge(SequenceFeature):
    """
    Calculate the total charge on a sequence at a given pH.
    """
    
    def __init__(self,data_file,seq_length,
                 normalize=True,use_sliding_windows=True,use_flip_pattern=True,
                 pH=7.4,n_term=False,c_term=False):
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
           

        self._base_features = np.array(["charge"])
        self._num_base_features = len(self._base_features)
        if self._use_flip_pattern:
            self._pattern_features = np.array(["{}_flip".format(f) for f in self._base_features])

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
    
    def _calc_score(self,seq):
        """
        Return the total charge.
        """
      
        total = np.array([sum([self.charge_dict[s] for s in seq]) + self._term_offset])

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
     
        
