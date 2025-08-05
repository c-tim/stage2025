#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 18:03:53 2025

@author: tim
"""

from Model import Model as m
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch
import torchvision

## This class will be used as an object contained by pytorchModel
class pyTorchNet(nn.Module):
        """
        A model of Pytorch used by the class pyTorchModel.
        """
    def __init__(self, n_layer_conv2d):

        super().__init__()
        self.conv1 = nn.Conv2d(n_layer_conv2d, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #self.useCuda = onCUDA
        
        
    #TODO maybe add a way to personnalize the layer and the forward propagation function
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    #TODO finish here
