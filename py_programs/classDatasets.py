#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 10:16:34 2025

@author: tim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models



import matplotlib.pyplot as plt
import numpy as np



class Dataset():
    
    
    USUAL_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __init__(self):
        """
        Create a Dataset for the model, including set and loader for training and test.

        Parameters
        ----------
        given_dataset
        transform, optional (default is USUAL_TRANSFORM)
        batch_size : int, optionnal (default is 4)

        Returns
        -------
        None.

        """
        self.train_inputs = None
        self.test_inputs = None
        
