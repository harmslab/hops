import numpy as np
import operator

from multiprocessing import Process, Queue

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


    def _calc_stuff(self,thread_number,queue):

        j = self._per_thread_sets[thread_number]
        k = self._per_thread_sets[thread_number + 1]

        out = []
        for i in range(j,k):
            tmp = []
            for f in self._feature_functions:
                tmp.append(f(self._sequences[i]))
            out.append(np.concatenate(tmp))

        queue.put((thread_number,out))

    def calc_features(self,num_threads=1):
        """
        Calculate the features for every observation using the appended Features
        instances. 
        """

        # Create a list of feature functions... 
        feature_names = []
        self._feature_functions = [] 
        num_features = 0
        for e in self._features_engines:
            num_features += e.num_features
            feature_names.append(e.features)
            self._feature_functions.append(e.score)
        
        self._feature_names = np.concatenate(feature_names)
        self._features = np.zeros((len(self._raw_values),num_features),dtype=float)
 
        per_thread = len(self._raw_values)//num_threads
        self._per_thread_sets = [(i+1)*per_thread for i in range(num_threads)]
        self._per_thread_sets.insert(0,0)
        self._per_thread_sets[-1] = len(self._raw_values)

        queue = Queue()

        proc_list = []
        for i in range(num_threads):
            proc_list.append(Process(target=self._calc_stuff,args=(i,queue)))
            proc_list[-1].start()

        out = []
        for p in proc_list:
            out.append(queue.get())
            #p.join()

        #out.sort()
        for o in out:
            self._features[self._per_thread_sets[o[0]]:self._per_thread_sets[o[0]+1],:] = o[1]
 
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


