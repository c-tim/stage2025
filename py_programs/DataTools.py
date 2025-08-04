#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:32:55 2025

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

from classDatasets import Dataset

def imshow(img):
    """
    Display image
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_transform(dimension : int):
    vec = [0.5 for i in range(dimension)]
    print(vec)
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(vec, vec)])

def scientificNotation_to_double(t):
    part = t.split('e')
    print(part)
    if (len(part)>1):
        return float(part[0])*10 ** (int(part[1]))
    return float(part[0])

def str_to_singleton(obj):
    """
    

    Parameters
    ----------
    obj : str or list

    Returns list
    """
    if type(obj)==str:
        obj = [obj]
    return obj
usual_transform = get_transform(3)

usual_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

usual_criterion = nn.CrossEntropyLoss()

testData =  transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#Datasets
CIFAR10 = Dataset(torchvision.datasets.CIFAR10)
MNIST = Dataset(torchvision.datasets.MNIST, transform=testData)
#Flowers = Dataset(torchvision.datasets.Flowers102) not working