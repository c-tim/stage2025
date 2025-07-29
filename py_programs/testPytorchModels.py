#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:04:21 2025

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

import sys
print(sys.getrecursionlimit())

from Model import Model as mod
from pyTorchModel import pyTorchModel
from classPytorchNet import pyTorchNet

#Initialisation before implementng pytorchNet
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


PATH = './saveTestModel.pth'
testModel = pyTorchModel(criterion)
#testModel.train(trainloader, PATH)
#testModel.print_performances_categories(PATH, testloader, classes)
#testModel.print_performances_global(PATH, testloader, classes)


all_models = models.list_models()
classification_models = models.list_models(module=models)
#testmodel = models.resnet18(pretrained=True)
PATH_EXAMPLE = './saveTestExampleModel.pth'
testModelExample = pyTorchModel(criterion, models.resnet18)
testModelExample.save_model(PATH_EXAMPLE)
testModelExample.train(trainloader, PATH_EXAMPLE)
testModelExample.print_performances_categories(PATH, testloader, classes)
testModelExample.print_performances_global(PATH, testloader, classes)

