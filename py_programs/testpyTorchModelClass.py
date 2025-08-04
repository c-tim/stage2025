#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:38:46 2025

@author: tim
"""


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from codecarbon import track_emissions


import DataTools
from pyTorchModel import pyTorchModel


'''print("oui")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
'''
#testModel = torchM(criterion)

#the commented methods below work
#testModel.train(trainloader, "./testsave.pth")
#testModel.print_performances_categories("./testsave.pth", testloader, classes)
'''
param_model = {"conv1_out":6}

#testModel.print_performances_global("./testsave.pth", testloader, classes)
testCustomNet = pyTorchModel(DataTools.usual_criterion,param_net_model=param_model)
testCustomNet.train(DataTools.CIFAR10.trainloader, "./testsave.pth")'''