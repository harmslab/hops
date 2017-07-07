__description__ = \
"""
Classes for manipulating strings of amino acids to return arrays of float features
for machine learning of peptide classes.
"""
__author__ = "Michael J. Harms"
__date__ = "2016-04-23"

from .base import Features
import localcider
import numpy as np

class CiderFeatures(Features):
    """
    Calculate a bunch of CIDER parameters on the sequence.
    """

    def __init__(self,seq_length=12,
                      use_sliding_windows=True,
                      features_to_ignore=None):
        """
        Initialize the class

        Parameters:
        -----------

        seq_length: length of peptide sequences being studied
        """
        
        self._seq_length = seq_length
        self._normalize = False
        self._use_sliding_windows = (use_sliding_windows != 0)
        self._use_flip_pattern = False
        self._features_to_ignore = np.array(features_to_ignore)

        self._compiled = False             
        self._ref_loaded = False
    
        self._base_features = ["cider_FCR",
                               "cider_FER",
                               "cider_FPPII_chain",
                               "cider_Fminus",
                               "cider_Fplus",
                               "cider_NCPR",
                               "cider_Omega",
                               "cider_charge_at_pH4",
                               "cider_charge_at_pH5",
                               "cider_charge_at_pH6",
                               "cider_charge_at_pH7",
                               "cider_charge_at_pH8",
                               "cider_charge_at_pH9",
                               "cider_countNeg",
                               "cider_countNeut",
                               "cider_countPos",
                               "cider_cumMeanHydropathy",
                               "cider_delta",
                               "cider_deltaMax",
                               "cider_dmax",
                               "cider_fraction_disorder_promoting",
                               "cider_isoelectric_point",
                               "cider_kappa",
                               "cider_meanHydropathy",
                               "cider_mean_net_charge",
                               "cider_molecular_weight",
                               "cider_sigma",
                               "cider_uverskyHydropathy"]

        self._base_feature_dict = {f:0.0 for f in self._base_features}
        self._compile_features()
    
    def _calc_score(self,seq):
        """
        Return the total charge.
        """

        total = np.array(self._single_seq_score(seq))

        if self._use_sliding_windows:

            windows = []
            # loop over possible sliding window lengths
            for i in range(self._seq_length):
         
                # loop over possible start points for window 
                for j in range(self._seq_length - i):
                    windows.extend([self._single_seq_score(s) for s in seq[j:(j+i)]])
        
            total = np.concatenate((total,np.array(windows)))
        
        if self._use_flip_pattern:
            pass
          
        return total
    
    def _calc_score(self,seq):
        """
        Each feature is simply the sum of the features across sites.
        """

        seq = np.array(list(seq))

        if len(seq) != self._seq_length:
            err = "Sequence length {} does not match length used to initialize class ({})\n".format(len(seq),self._seq_length)
            raise ValueError(err)

        # Sliding windows
        if self._use_sliding_windows:
            window_features = np.zeros((self._num_windows,self._num_base_features),dtype=float)
        else:
            window_features = np.zeros(0,dtype=float)
      
        # Flip pattern 
        if self._use_flip_pattern: 
            flip_features = np.zeros(self._num_base_features,dtype=float)
        else:
            flip_features = np.zeros(0,dtype=float)

        # Grab parametres for total sequence
        totals = self._single_seq_score("".join(seq)) 

        # Do sliding window calculation
        for i in range(self._num_windows):
            new_seq = "".join(seq[self._window_masks[i]])
            window_features[i,:] = self._single_seq_score(new_seq)
    
        # Transpose matrix so its in the same basic form as other calcs
        if self._use_sliding_windows:
            window_features = window_features.T 
           
        return np.concatenate((totals,np.ravel(window_features),flip_features))
        
    def _single_seq_score(self,seq):        
        """
        Return a list of cider calculated values for seq.
        """

        seq = str(seq)
        s = localcider.sequenceParameters.Sequence(seq)
        
        out = [s.FCR(),
               s.FER(),
               s.FPPII_chain(),
               s.Fminus(),
               s.Fplus(),
               s.NCPR(),
               s.Omega(),
               s.charge_at_pH(4.0),
               s.charge_at_pH(5.0),
               s.charge_at_pH(6.0),
               s.charge_at_pH(7.0),
               s.charge_at_pH(8.0),
               s.charge_at_pH(9.0),
               s.countNeg(),
               s.countNeut(),
               s.countPos(),
               s.cumMeanHydropathy()[-1],
               s.delta(),
               s.deltaMax(),
               s.dmax,
               s.fraction_disorder_promoting(),
               s.isoelectric_point(),
               s.kappa(),
               s.meanHydropathy(),
               s.mean_net_charge(),
               s.molecular_weight(),
               s.sigma(),
               s.uverskyHydropathy()]

        return out
