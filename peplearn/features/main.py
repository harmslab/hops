
from .base import SequenceFeature

import numpy as np

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
 
