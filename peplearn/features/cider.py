__description__ = \
"""
Classes for manipulating strings of amino acids to return arrays of float features
for machine learning of peptide classes.
"""
__author__ = "Michael J. Harms"
__date__ = "2016-04-23"

from .base import SequenceFeature
import localcider
import numpy as np

class SequenceCider(SequenceFeature):
    """
    Calculate the total charge on a sequence at a given pH.
    """
    
    def __init__(self,data_file,seq_length,
                 normalize=True,use_sliding_windows=True,use_flip_pattern=True):
        """
        Load in the data file and then determine various pH-dependent charge info. 
        """
        
        # Call the parent class
        super(self.__class__,self).__init__(data_file,seq_length,
                                            normalize=normalize,
                                            use_sliding_windows=use_sliding_windows,
                                            use_flip_pattern=use_flip_pattern)

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

        self._num_base_features = len(self._base_features)

        if self._use_flip_pattern:
            self._pattern_features = np.array()

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
