__description__ = \
"""
Class storing observations and allowing calculation of features. 
"""
__author__ = "Michael J. Harms"
__usage__ = "2017-08-23"

from . import features

import numpy as np
import operator

from multiprocessing import Process, Queue
import queue as queue_module
import copy, sys

class Observations:
    """
    Class for creating and holding machine learning training/test sets.
    """
    
    def __init__(self,observation_file,kfold_size=10,value_type=None):
        """
        Initialize the class.

        observation_file: text file with observations. 
        kfold_size: split the data into k_fold sets for cross validation.  
                    1/kfold_size observations are set aside as a true test set.
                    The remaining observations are used for (kfold_size-1)
                     cross validation.
        value_type: type of value in the dataset.  If None, this is determined
                    from the file itself.
        """
       
        self._observation_file = observation_file
        self._kfold_size = kfold_size
        self._value_type = value_type 
 
        # load in all of the observations
        self._load_observations()
        
        # initial features.  everyone gets a 0
        self._features = np.zeros((len(self._raw_values),1),dtype=float)
        self._feature_names = np.array(["dummy"])
        self._features_engines = []

        # set initial breaks to None
        self._breaks = None
        
    def _new_test_set(self):
        """
        Create a test set by randomizing indexes and selecting a subset to be used as test versus
        training.
        """
        
        self._k_fraction = 1.0/self._kfold_size
        self._k_length = int(np.floor(len(self._indexes)*self._k_fraction))
               
        np.random.shuffle(self._indexes)
        
        
    def add_features(self,features_instance):
        """
        Calculate features using a features_instance and append to the total feature set.
        """
      
        self._features_engines.append(features_instance)


    def _calc_features_on_thread(self,first_seq,last_seq,queue):
        """
        Calculate features on a set of sequences on its own thread.

        Create a 1D features array for each sequence.  Put those into a list
        and put outo the queue.
        """

        out = []
        for i in range(first_seq,last_seq):
            tmp = []
            for f in self._feature_functions:
                tmp.append(f(self._sequences[i]))
            out.append(np.concatenate(tmp))

        queue.put((first_seq,last_seq,out))


    def calc_features(self,num_threads=1,block_size=3000):
        """
        Calculate the features for every observation using the appended Features
        instances. 

        num_threads: number of threads to run on
        block_size: number of seqeuences to put on a single process
        """
    
        # Create a list of feature functions... 
        feature_names = []
        self._feature_functions = [] 
        num_features = 0
        for e in self._features_engines:
            num_features += e.num_features
            feature_names.append(e.features)
            self._feature_functions.append(e.score)

        # Create a compiled list of feature names
        self._feature_names = np.concatenate(feature_names)
       
        # If enough threads are specified that only a few threads would start, 
        # make the block size smaller 
        if len(self._sequences)//num_threads < block_size:
            block_size = len(self._sequences)//num_threads + 20
        
        # Split squences in to blocks of block_size
        block_edges = []
        for i in range(0,len(self._sequences),block_size):
            block_edges.append(i)
        block_edges.append(len(self._sequences) - 1)

        # Start a process for each thread
        proc_list = []
        queue_list = []
        out = []

        # Go through each sequence
        for i in range(len(block_edges)-1):
            
            first_seq = block_edges[i]
            last_seq =  block_edges[i+1]

            queue_list.append(Queue())
            proc_list.append(Process(target=self._calc_features_on_thread,
                                     args=(first_seq,last_seq,queue_list[-1])))
            proc_list[-1].start()
           
            # If we've capped our number of threads, wait until one of the
            # processes finishes to move on 
            if (len(queue_list) == num_threads) or (i == len(block_edges) - 2):

                waiting = True
                while waiting:

                    # Go through queues
                    for j, q in enumerate(queue_list):
                      
                        # Try to get output on queue.  If output is there, get 
                        # the output and then remove the associated process and
                        # queue 
                        try:
                            out.append(q.get(block=True,timeout=0.1))
                            p = proc_list.pop(j)
                            queue_list.pop(j)

                            waiting = False
                            break

                        except queue_module.Empty:
                            pass

                    # If we're on the last block, wait until the queue is
                    # completely empty before proceeding
                    if len(queue_list) != 0 and i == (len(block_edges) - 2):
                        waiting = True
 
        # Load results into self._features
        self._features = np.zeros((len(self._sequences),num_features),dtype=float)
        for o in out:
            self._features[o[0]:o[1],:] = o[2]

 
    def _load_observations(self):
        """
        Load in a file of observations.  Expected to have format:
        
        SEQ VALUE [VALUE_WEIGHT]
        
        Blank lines and lines starting with # are ignored.
        """

        seq_list = []
        raw_value_list = []
        weight_list = []
        with open(self._observation_file) as f:
            for l in f:
            
                if l.strip() == "" or l.startswith("#"):
                    continue
            
                col = l.split()

                sequence = col[0].strip()

                # Figure out whether this data is type float, integer, or string
                if self._value_type is None:
                    try:
                        v = float(col[1].strip())
                        if v.is_integer():
                            self._value_type = int  
                        else:
                            self._value_type = float
                    except ValueError:
                        self._value_type = str

                raw_value = self._value_type(col[1].strip())

                try:
                    weight = float(col[2])
                except (IndexError,ValueError):
                    weight = 1.0

                seq_list.append(sequence)
                raw_value_list.append(raw_value)
                weight_list.append(weight)

        self._sequences = np.array(seq_list)
       
        self._raw_values = np.array(raw_value_list,dtype=self._value_type)
        self._weights = np.array(weight_list,dtype=float)
        self._indexes = np.arange(len(self._raw_values))
       
        self._values = np.copy(self._raw_values)
     
        self._new_test_set()
        
    def add_cutoff_filter(self,logic=">=",cutoff=None):
        """
        Add a cutoff, removing a subset of the data.  This is nondestructive.  Filters can 
        be added one on top of the other sequentially.  
        
        logic is a string version of a logical operator (>=, <=, >, <, ==, !=).
            
        cutoff is the value against which the data are compared. 
        """
        
        filter_ops = {">=":operator.ge,
                      "<=":operator.le,
                      ">" :operator.gt,
                      "<" :operator.lt,
                      "==":operator.eq,
                      "!=":operator.ne}
        
        if cutoff is not None:
            f = filter_ops[logic](self.values,cutoff)
            self._indexes = self._indexes[f]
                                              
            self._new_test_set()

    def remove_filters(self):
        """
        Remove any filters that have been applied.
        """          
            
        self._indexes = np.arange(len(self._raw_values))
        self._new_test_set()
    
    def add_classes(self,breaks):
        """
        Convert into classes defined by breaks.  self.values (and test_values, training_values)
        will now return discrete classes rather than float values.  These can be removed by
        remove_classes.  
        """

        self._breaks = copy.copy(breaks)

        classes = np.zeros(len(self._raw_values))
        
        # Values below the first break        
        class_number = 0
        first_class = self._raw_values < self._breaks[0]
        if np.sum(first_class) > 0:
            classes[first_class] = class_number
            class_number += 1
            
        # Values between each break
        for i in range(len(self._breaks)-1):
            c1 = self._raw_values >= self._breaks[i]
            c2 = self._raw_values < self._breaks[i+1]
            
            classes[c1*c2] = class_number
            class_number += 1
        
        # Values above the top break
        last_class = self._raw_values >= self._breaks[-1]
        classes[last_class] = class_number
        
        self._values = classes
        
    def remove_classes(self):
        """
        Remove discrete class calls so values returns floats.
        """
        self._values = np.copy(self._raw_values)
        self._breaks = None
   
    @property
    def breaks(self):
        """
        Return breaks between classes.
        """
       
        if not self._breaks is None:
            return self._breaks
    
        return None

    @property
    def feature_names(self):
        """
        Return names of all features.
        """
        
        return self._feature_names
 
    @property
    def sequences(self):
        """
        Return all sequences, with appropriate cutoffs applied.
        """
    
        return self._sequences[self._indexes]
    
    @property
    def features(self):
        """
        Return all sequence features, with appropriate cutoffs applied.
        """
            
        return self._features[self._indexes,:]
            
    @property
    def values(self):
        """
        Return all values, with appropriate cutoffs or class filters applied.
        """
        
        return self._values[self._indexes]
    
    @property
    def weights(self):
        """
        Return all value weights, with appropriate cutoffs or class filters applied.
        """
        
        return self._weights[self._indexes]
            
    @property
    def test_sequences(self):
        """
        Return sequences for test set.
        """
        
        return self.sequences[0:self._k_length]

    @property
    def test_features(self):
        """
        Return features for test set.
        """
        
        return self.features[0:self._k_length,:]

    @property
    def test_values(self):
        """
        Return values for test set.
        """
        
        return self.values[0:self._k_length]    

    @property
    def test_weights(self):
        """
        Return weights for test set.
        """
        
        return self.weights[0:self._k_length]

    @property
    def training_sequences(self):
        """
        Return sequences for training set.
        """
        
        return self.sequences[self._k_length:]
          
    @property
    def training_features(self):
        """
        Return features for training set.
        """
        
        return self.features[self._k_length:,:]
        
    @property
    def training_values(self):
        """
        Return values for training set.
        """

        return self.values[self._k_length:]

    @property
    def training_weights(self):
        """
        Return weights for training set. 
        """
        
        return self.weights[self._k_length:]

    @property
    def kfold_size(self):
        """
        Number of categories for k-fold training and validation.
        """
    
        return self._kfold_size

    def _meta_k_training(self,array_to_slice,k):
        """
        Meta function that slices an array that returns a k-fold 
        training set.  This should only be called by the public methods
        in this class (things like get_k_training_sequences). 
        """

        # Only use the real training set
        real_training = array_to_slice[self._k_length:]

        if k > self._kfold_size -2 or k < 0:
            err = "k must be between 0 and {}.\n".format(0,self._kfold_size-2)
            raise ValueError(err)

        start =  real_training[:(k*self._k_length)]        
        finish = real_training[((k+1)*self._k_length):]
        
        return np.concatenate((start,finish))
     
    def _meta_k_test(self,array_to_slice,k):
        """
        Meta function that slices an array that returns a k-fold 
        test set.  This should only be called by the public methods
        in this class (things like get_k_test_sequences). 
        """

        # Only use the real training set
        real_training = array_to_slice[self._k_length:]

        if k > self._kfold_size -2 or k < 0:
            err = "k must be between 0 and {}.\n".format(0,self._kfold_size-2)
            raise ValueError(err)

        return real_training[(k*self._k_length):((k+1)*self._k_length)]        

    def get_k_training_sequences(self,k):
        """
        Get the kth set of training sequences for k-fold cross training and
        validation.
        
        k must be between 0 and total k -1.
        """

        return self._meta_k_training(self.sequences,k)

    def get_k_training_features(self,k):
        """
        Get the kth set of training features for k-fold cross training and
        validation.
        
        k must be between 0 and total k -1.
        """

        return self._meta_k_training(self.features,k)

    def get_k_training_values(self,k):
        """
        Get the kth set of training values for k-fold cross training and
        validation.
        
        k must be between 0 and total k -1.
        """

        return self._meta_k_training(self.values,k)

    def get_k_training_weights(self,k):
        """
        Get the kth set of training weights for k-fold cross training and
        validation.
        
        k must be between 0 and total k -1.
        """

        return self._meta_k_training(self.weights,k)

    def get_k_test_sequences(self,k):
        """
        Get the kth set of test sequences for k-fold cross test and
        validation.
        
        k must be between 0 and total k -1.
        """

        return self._meta_k_test(self.sequences,k)

    def get_k_test_features(self,k):
        """
        Get the kth set of test features for k-fold cross test and
        validation.
        
        k must be between 0 and total k -1.
        """

        return self._meta_k_test(self.features,k)

    def get_k_test_values(self,k):
        """
        Get the kth set of test values for k-fold cross test and
        validation.
        
        k must be between 0 and total k -1.
        """

        return self._meta_k_test(self.values,k)

    def get_k_test_weights(self,k):
        """
        Get the kth set of test weights for k-fold cross test and
        validation.
        
        k must be between 0 and total k -1.
        """

        return self._meta_k_test(self.weights,k)


def calc_features(sequence_data,use_flip_pattern=True,use_sliding_windows=12,
                  num_threads=1):
    """
    Calculate the features of a dataset.

    Parameters:
    sequence_data: a file with a collection of sequences with line format
        sequence value [weight]
    use_flip_pattern: bool.  whether or not to calculate vector of sign flip
                      for each feature slong the sequence
    use_sliding_windows: int. max size of sliding windows to employ.
    num_threads: number of threads to use for the calculation.

    Returns an Observations instance with calculated features.
    """

    print("Constructing feature set.")
    sys.stdout.flush()

    # Create observations object
    obs = Observations(sequence_data)

    # Append features on which to train
    simple_features = features.SimpleFeatures(use_flip_pattern=use_flip_pattern,
                                              use_sliding_windows=use_sliding_windows)
    cider_features = features.CiderFeatures(use_sliding_windows=bool(use_sliding_windows))

    obs.add_features(simple_features)
    obs.add_features(cider_features)

    # Do the calculation
    print("Calculating features on {} threads.".format(num_threads))
    sys.stdout.flush()

    obs.calc_features(num_threads)

    return obs

