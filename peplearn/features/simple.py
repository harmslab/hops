
from .base import Features

import numpy as np

class SimpleFeatures(Features):
    """
    Sequence features that are a simple sum across amino acids.
    """
    
    def _calc_score(self,seq):
        """
        Each feature is simply the sum of the features across sites.
        """

        if len(seq) != self._seq_length:
            err = "Sequence length {} does not match length used to initialize class ({})\n".format(len(seq),self._seq_length)
            raise ValueError(err)

        value_vec = np.zeros(self._seq_length,dtype=float)
        totals = np.zeros(self._num_base_features,dtype=float)

        # Sliding windows
        if self._use_sliding_windows > 0:
            window_features = np.zeros((self._num_base_features,self._num_windows),dtype=float)
        else:
            window_features = np.zeros(0,dtype=float)
      
        # Flip pattern 
        if self._use_flip_pattern: 
            flip_features = np.zeros(self._num_base_features,dtype=float)
        else:
            flip_features = np.zeros(0,dtype=float)

        for i, f in enumerate(self._base_features):

            # Vector of values for each amino acid in the sequence for this 
            # feature
            for j, s in enumerate(seq):
                value_vec[j] = self._base_feature_dict[f][s]

            # Sum of the values
            totals[i] = np.sum(value_vec)
            
            # Create vector of sliding windows of this feature
            if self._use_sliding_windows > 0:
                for j in range(len(self._window_masks)):
                    window_features[i,j] = np.sum(value_vec[self._window_masks[j,:]])

            # See if the element before each element in the vector has the same
            # sign as the previous element
            if self._use_flip_pattern:
                f_flips = value_vec <= np.mean(value_vec)
                flip_features[i] = np.sum(f_flips[1:self._seq_length] == f_flips[0:(self._seq_length-1)])/(self._seq_length/2)
       
        return np.concatenate((totals,np.ravel(window_features),flip_features))
        
 
