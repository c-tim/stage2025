#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:09:07 2025

@author: tim
"""

from __future__ import print_function

import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


class TensorNet():
    
    def __init__(self, layers = [6,32,32,2]):
        self.net = tflearn.input_data(shape=[None, layers[0]])
        for n in range(len(1,layers-1)):
            self.net = tflearn.fully_connected(self.net, layers[n])
        self.net = tflearn.fully_connected(self.net, layers[-1], activation='softmax')
        self.net = tflearn.regression(self.net)
      
        # Define model
        self.model = tflearn.DNN(self.net)
        
    
    def train(self, data, n_epoch = 10, batch_size = 16):
        # Start training (apply gradient descent algorithm)
        self.model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

    def get_output(self, data):
        self.model.predict(data)