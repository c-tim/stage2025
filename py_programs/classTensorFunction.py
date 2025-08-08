#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:09:07 2025

@author: tim
"""

from __future__ import print_function

import numpy as np
import tflearn
from classTfDatasets import tfDatasets
from classTensorModule import *
# Download the Titanic dataset
from tflearn.datasets import titanic
#titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
#data, labels = load_csv('titanic_dataset.csv', target_column=0,
#                       categorical_labels=True, n_classes=2)


class TensorNet():
    

    def __init__(self, layers = [6,32,32,2]):
        self.is_tensor_built : bool = False
        
        self.list_module = []
        
        if type(layers[0])==int:
            self.list_module.append(tfInputData(shape=[None, layers[0]]))

        elif type(layers[0])== tfInputData :
            self.list_module.append(layers[0])
        else :
            return TypeError("the first element ", layers[0], " in the layers is not a int or InputData")

 
            
        for n in range(1,len(layers)):
            if type(layers[n]) == int :
                self.list_module.append(tfFullyConnected(layers[n]))
            else :
                self.list_module.append(layers[n])
                    
            
        
        '''self.net = tflearn.input_data(shape=[None, layers[0]])
        for n in range(1,len(layers)-1):
            if type(layers[n])==int:
                self.net = tflearn.fully_connected(self.net, layers[n])
            elif len(layers[n])>1:
                current_layer=layers[n]
                if current_layer[0] == "dropout":
                    self.net = tflearn.dropout(self.net, layers[n][1])
                
        self.net = tflearn.fully_connected(self.net, layers[-1], activation='softmax')
        self.net = tflearn.regression(self.net)
      
                    '''

            
    def print_modules(self):
        for module in self.list_module:
            print(module.toString())
    
    def build_model(self):
        self.net = None
        for module in self.list_module:
            self.net = module.addLayer(self.net)
        self.net = tflearn.regression(self.net)

        
        # Define model
        self.model = tflearn.DNN(self.net)
        
        self.is_tensor_built = True
            
            
        
    
    def train(self, data, labels, n_epoch = 10, batch_size = 16):
        if not self.is_tensor_built:
            print("Warning : tensor was not built. Building the net")
            self.build_model()
        #data, labels = dataset
        # Start training (apply gradient descent algorithm)
        self.model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

    def get_output(self, data):
        self.model.predict(data)