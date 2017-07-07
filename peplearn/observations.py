import numpy as np
import operator

from multiprocessing import Process, Queue
import queue as queue_module
import time

import sys

class Observations:
    """
    Class for creating and holding machine learning training/test sets.
    """
    
    def __init__(self,observation_file,test_size=0.1):
        """
        Initialize the class.
        """
       
        self._observation_file = observation_file
        self._test_size = test_size
        
        # load in all of the observations
        self._load_observations()
        
        # initial features.  everyone gets a 0
        self._features = np.zeros((len(self._raw_values),1),dtype=float)
        self._feature_names = np.array(["dummy"])
        self._features_engines = []
        
    def new_test_set(self,test_size=None):
        """
        Create a test set by randomizing indexes and selecting a subset to be used as test versus
        training.
        """
        
        if test_size is not None:
            self._test_size = test_size
        
        np.random.shuffle(self._indexes)
        self._test_length = int(np.floor(len(self._indexes)*self._test_size))
        
        
    def add_features(self,features_instance):
        """
        Calculate features using a features_instance and append to the total feature set.
        """
      
        self._features_engines.append(features_instance)


    def _calc_features_on_thread(self,first_seq,last_seq,queue):

        # Create a 1D features array for each sequence.  Put those into a list
        # and put outo the queue.
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
        if len(self._raw_values)//num_threads < block_size:
            block_size = len(self._raw_values)//num_threads + 20
        

        # Split up the observations across threads
        block_edges = []
        for i in range(0,len(self._raw_values),block_size):
            block_edges.append(i)
        block_edges.append(len(self._raw_values) - 1)

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
 


        print(len(proc_list),len(queue_list))
        # Make sure the processes are all done before proceeding
        #for p in proc_list:
        #    p.join()
                
        # Get any remaining job results
        #for q in queue_list:
        #    out.append(q.get())

        # Load results into self._features
        self._features = np.zeros((len(self._raw_values),num_features),dtype=float)
        for o in out:
            self._features[o[0]:o[1],:] = o[2]
 
    def _load_observations(self):
        """
        Load in a file of observations.  Expected to have format:
        
        SEQ VALUE [VALUE_ERROR]
        
        Blank lines and lines starting with # are ignored.
        """

        seq_list = []
        raw_value_list = []
        raw_err_list = []
        with open(self._observation_file) as f:
            for l in f:
            
                if l.strip() == "" or l.startswith("#"):
                    continue
            
                col = l.split()

                sequence = col[0].strip()
                value = float(col[1]) 

                try:
                    value_err = float(col[2])
                except (IndexError,ValueError):
                    value_err = 1.0

                seq_list.append(sequence)
                raw_value_list.append(value)
                raw_err_list.append(value_err)

        self._sequences = np.array(seq_list)
        
        self._raw_values = np.array((raw_value_list,raw_err_list),dtype=float)
        self._raw_values = self._raw_values.T

        # Add noise to the measured values to prevent numerical errors downstream
        # if two values happen to be identical
        noise = np.random.normal(0,np.std(self._raw_values[:,0])/1000,len(self._raw_values)) 
        self._raw_values[:,0] = self._raw_values[:,0] + noise
     
        self._values = np.copy(self._raw_values)
        self._indexes = np.arange(len(self._raw_values))
        
        self.new_test_set()
        
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
                                              
            self.new_test_set()
    
    def add_classes(self,breaks):
        """
        Convert into classes defined by breaks.  self.values (and test_values, training_values)
        will now return discrete classes rather than float values.  These can be removed by
        remove_classes.  
        """

        classes = np.zeros(len(self._raw_values))
        
        # Values below the first break        
        class_number = 0
        first_class = self._raw_values[:,0] < breaks[0]
        if np.sum(first_class) > 0:
            classes[first_class] = class_number
            class_number += 1
            
        # Values between each break
        for i in range(len(breaks)-1):
            c1 = self._raw_values[:,0] >= breaks[i]
            c2 = self._raw_values[:,0] < breaks[i+1]
            
            classes[c1*c2] = class_number
            class_number += 1
        
        # Values above the top break
        last_class = self._raw_values[:,0] >= breaks[-1]
        classes[last_class] = class_number
        
        self._values[:,0] = classes
        self._values[:,1] = 1.0
            
    def remove_filters(self):
        """
        Remove any filters that have been applied.
        """          
            
        self._indexes = np.arange(len(self._raw_values))
        self.new_test_set()
        
    def remove_classes(self):
        """
        Remove discrete class calls so values returns floats.
        """
        self._values = np.copy(self._raw_values)
    
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
        
        return self._values[self._indexes,0]
    
    @property
    def errors(self):
        """
        Return all value errors, with appropriate cutoffs or class filters applied.
        """
        
        return self._values[self._indexes,1]
            
    @property
    def test_sequences(self):
        """
        Return sequences for test set.
        """
        
        return self.sequences[0:self._test_length]

    @property
    def test_features(self):
        """
        Return features for test set.
        """
        
        return self.features[0:self._test_length,:]

    @property
    def test_values(self):
        """
        Return values for test set.
        """
        
        return self.values[0:self._test_length]    
    
    @property
    def training_errors(self):
        """
        Return errors for training set. 
        """
        
        return self.errors[self._test_length:]
    
    @property
    def training_sequences(self):
        """
        Return sequences for training set.
        """
        
        return self.sequences[self._test_length:]
          
    @property
    def training_features(self):
        """
        Return features for training set.
        """
        
        return self.features[self._test_length:,:]
        
    @property
    def training_values(self):
        """
        Return values for training set.
        """

        return self.values[self._test_length:]
    
    @property
    def test_errors(self):
        """
        Return errors for test set.
        """
        
        return self.errors[0:self._test_length]
    
    @property
    def feature_names(self):
        
        return self._feature_names


