#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:02:41 2025

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

from tuto_remaster import Net

#project variables
PATH = './cifar_net.pth'

def separator(count = 1):
    for i in range(1,count):
        print("-------------------------------")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)



criterion = nn.CrossEntropyLoss()

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




for scale in range(1,6):
    #Network Training
    print("Using a network scaled by ", scale)
    net = Net(scale)

    net.to(device)

    
    # analyze performance and consumption based on the times the data set is reused for the model
    for i in range(1, 5):
        separator(2)
        print("Train with dataset used ", i, " times")
        net.train(trainloader, PATH, i)
        net.analyse_performance_global(trainloader, classes)
    separator(3)
'''

#net.train(trainloader, PATH, 2)


# tests increasing the scale of the network
net = Net(2)

net.to(device)
net.train(trainloader, PATH, 2)
net.analyse_performance_global(trainloader, classes)

separator()
separator()
separator()

net = Net(10)

net.to(device)
net.train(trainloader, PATH, 2)
net.analyse_performance_global(trainloader, classes)

'''





