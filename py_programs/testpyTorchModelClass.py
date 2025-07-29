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



from pyTorchModel import pyTorchModel as torchM


print("oui")

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

testModel = torchM(criterion)

#the commented methods below work
#testModel.train(trainloader, "./testsave.pth")
#testModel.print_performances_categories("./testsave.pth", testloader, classes)
testModel.print_performances_global("./testsave.pth", testloader, classes)
