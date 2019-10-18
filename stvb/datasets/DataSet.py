import numpy as np

# This is creating the object for the dataset

class DataSet():

    def __init__(self, X, shuffle=False):
        
        # In this continous formualation X is a list of events locations

        # Get the number of training obs - this is task specific cause each task has a different number of events' locations
        self._num_examples = X.shape[0]
        self._X = X

        # Set the initial value of iterations to zero
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        # Gives the dimension for each obs - the input dimension is the same for all tasks
        self._Din = X.shape[0]

        # Gives the number of tasks
        self._Dout = len(X)

        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        # This is allowing for batch optimisation. Only part of the data is used for each batch. 
        # In the discrete case all tasks had the same number of training points 
        # Now the _num_examples is specific for each task
        # Batch size need to be a vector of size num_tasks

        # When training, this functions pass the batch of data to use
        # If only one batch is used that it returns the overall dataset
        # It keeps track of the epochs completed with the variable _epochs_completed 
        # which is incremented by 1 everytime we have a complete pass over the data
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        # This is the case of batch opt in which we consider the overall dataset
        if (self._index_in_epoch > self._num_examples) and (start != self._num_examples):
            self._index_in_epoch = self._num_examples
        
        if self._index_in_epoch > self._num_examples:   # Finished epoch
            self._epochs_completed += 1
            # This is shuffling each element of the vector
            perm = np.arange(self._num_examples)
            #np.random.shuffle(perm)                  # Shuffle the data
            # each element in the list needs to be shuffled 
            self._X = self._X[perm,:]

            start = 0                               # Start next epoch
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        # get start to end for each element in a list
        return self._X[start:end,:]


    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def Din(self):
        return self._Din

    @property
    def X(self):
        return self._X



