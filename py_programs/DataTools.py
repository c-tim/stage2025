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

from classpyDatasets import pytorchDataset

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

def get_transform_resized(dimension: int):
    vec = [0.5 for _ in range(dimension)]
    print(vec)
    return transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to something reasonable for CNN
        transforms.ToTensor(),
        transforms.Normalize(vec, vec)
    ])

def get_transform_resized2(dimension: int):
    vec = [0.5 for _ in range(dimension)]
    print(vec)
    return transforms.Compose([
        transforms.Resize((224, 224)),   # Resize to what AlexNet expects
        transforms.ToTensor(),
        transforms.Normalize(vec, vec)
    ])


def scientificNotation_to_double(t):
    """
    Exemple : "1e-10" --> 0.1
    """
    part = t.split('e')
    if (len(part)>1):
        return float(part[0])*10 ** (int(part[1]))
    return float(part[0])

def str_to_singleton(obj):
    """
    If obj is a string, returns it as a singleton of a List.
    Returns obj otherwise.
    """
    if type(obj)==str:
        obj = [obj]
    return obj

#Usual components for the traininf and test of models
usual_transform = get_transform(3)

adapted_transform = get_transform_resized(3)

usual_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform_resize = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

usual_criterion = nn.CrossEntropyLoss()

testData =  transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#Datasets declaration
CIFAR10 = pytorchDataset(torchvision.datasets.CIFAR10, transform=usual_transform)
MNIST = pytorchDataset(torchvision.datasets.MNIST, transform=testData)

#Flowers = Dataset(torchvision.datasets.Flowers102) not working