#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 14:39:31 2025

@author: tim
"""

import tflearn

class TensorModule():
    
    def toString(self) ->str:
        return "Blanck Tensor Module"
    
    
    def addLayer(self, net):
        print("Blanck Tensor Module : return net")
        return net


class tfInputData(TensorModule):
    
    def __init__(self, **kwargs):
        self.args = kwargs
    
    def toString(self) -> str:
        return "Input layer with args" + str(self.args)
        
    def addLayer(self, net=None):
        return tflearn.input_data(**self.args)



class tfFullyConnected(TensorModule):
    
    def __init__(self, n_nodes):
        self.nodes = n_nodes
    
    def toString(self) -> str:
        return "Fully connected layer of " + str(self.nodes)+ " nodes"
        
    def addLayer(self, net):
        return tflearn.fully_connected(net, self.nodes)


class tfDropout(TensorModule):
    
    def __init__(self, keep_propbability):
        self.keep_propbability = keep_propbability
    
    def toString(self) -> str:
        return "Dropout layer of keep_probability = "+str(self.keep_propbability)
        
    def addLayer(self, net):
        return tflearn.dropout(net, self.keep_propbability)
