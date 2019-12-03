# -*- coding: utf-8 -*-
"""
Created on Fri November 8 16:55:04 2019

@author: Carlos
"""

import numpy as np
from tensorflow.contrib import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, X, y, vocab_size, batch_size = 32):
        'Initialization'
        
        self.X = X
        self.y = y
        self.vocab_size = vocab_size        
        self.batch_size = batch_size
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        batch_start = index * self.batch_size
        batch_end   = batch_start + self.batch_size
        indexes = np.arange(batch_start, batch_end)

        # Get this batch of integer encoded sequences
        batch_sequns = self.X[indexes,:]
        batch_labels = self.y[indexes]
        
        # Perform one-hot encoding to generate data ready for input to LSTM
        X = [keras.utils.to_categorical(x, num_classes = self.vocab_size) for x in batch_sequns]
        X = np.array(X)
        y = keras.utils.to_categorical(batch_labels, num_classes = self.vocab_size)
        
        # Return the one-hot encoded data and labels
        return X, y
    
    
    def get_seed_text(self):
        '''
        Get random seed text from the full set of data
        '''
        seed_index = np.random.randint(0, len(self.X))
        return(self.X[seed_index,:])