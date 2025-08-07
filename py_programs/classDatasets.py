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



class pytorchDataset():
    
    
    USUAL_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __init__(self, given_dataset, transform = USUAL_TRANSFORM, batch_size=4):
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
        self.dataset = given_dataset
        self.trainset = given_dataset(root='../data', train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        self.testset = given_dataset(root='../data', train=False,
                                               download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        self.test_sample = next(iter(self.trainloader))
        
