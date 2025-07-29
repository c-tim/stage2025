#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 11:38:26 2025

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


from Model import Model as mod
from pyTorchModel import pyTorchModel

from codecarbon import track_emissions

from CSVReader import CSVfile

from classEnergyAnalyzer import EnergyAnalyzer

import DataTools

#  @track_emissions() must be set on trained 

# First Test :
# does track_emissions works for a method calling the desired method
test1_label = "Test1_track_method_within_method"

#@track_emissions(project_name=test1_label)
def test_calling_tracked_emissions(model : pyTorchModel, trainData, path):
    model.train(trainData, path)

def aaaargh(a,b,c):
    print(a,b,c)

#@track_emissions(project_name="test")
def test_track(func , *args):
    print(args, "aaaa")
    
    func(args)

def encaplusalte_with_decorators(f):
    tracked = track_emissions(project_name="test")(f)
    return tracked


def encapsulate(f):
    @track_emissions(project_name="test")
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapped 
testModel = pyTorchModel(DataTools.usual_criterion, None, 3)

testEnergyAnalyzer = EnergyAnalyzer("testClassEnergy")
testa = testEnergyAnalyzer.track_function(testModel.train)
testa(DataTools.CIFAR10.trainloader, "save_model.pth")

#test_calling_tracked_emissions(testModel, DataTools.CIFAR10.trainloader, "save_model.pth")

#encaplusalte_with_decorators(testModel.train(DataTools.CIFAR10.trainloader, "save_model.pth")) 
'''testa = encapsulate(testModel.train)
testa(DataTools.CIFAR10.trainloader, "save_model.pth")'''
#test_track(testModel.train2, testModel, DataTools.CIFAR10.trainloader, "save_model.pth")
#test_track(aaaargh,testModel, DataTools.CIFAR10.trainloader, "save_model.pth")
'''result_file = CSVfile()
result_file.print_columns(["project_name","run_id","duration", "emissions","cpu_energy", "gpu_energy", "ram_energy", "energy_consumed"])

'''