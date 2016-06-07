import numpy as np
import operator

class MachineLearnerData:
    """
    Class for creating and holding machine learning training/test sets.
    """
    
    def __init__(self,input_file,test_size=0.1):
        """
        Initialize the class.
        """
        
        self._input_file = input_file
        self._test_size = test_size
        
        # soad in all of the observations
        self._load_observations()
        
        # initial features.  everyone gets a 0
        self._features = np.zeros((len(self._raw_values),1),dtype=float)
        self._feature_names = np.array(["dummy"])
        
    def new_test_set(self,test_size=None):
        """
        Create a test set by randomizing indexes and selecting a subset to be used as test versus
        training.
        """
        
        if test_size != None:
            self._test_size = test_size
        
        np.random.shuffle(self._indexes)
        self._test_length = np.floor(len(self._indexes)*self._test_size)
        
        
    def add_feature(self,seq_feature_instance):
        """
        Calculate features using a seq_feature_instance and append to the total feature set.
        """
       
        new_features = np.zeros((len(self._raw_values),seq_feature_instance.num_features),
                                dtype=float)
       
        for i, s in enumerate(self._sequences):
            new_features[i,:] = seq_feature_instance.score(s)

        self._feature_names = np.concatenate((self._feature_names,seq_feature_instance.features))
        self._features = np.hstack((self._features,new_features))
                
    
    def _load_observations(self):
        """
        Load in a file of observations.  Expected to have format:
        
        SEQ VALUE [VALUE_ERROR]
        
        Blank lines and lines starting with # are ignored.
        """

        seq_list = []
        raw_value_list = []
        raw_err_list = []
        with open(self._input_file) as f:
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
        
        if cutoff != None:
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
