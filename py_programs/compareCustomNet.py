#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:46:13 2025

@author: tim
"""


from classEnergyAnalyzer import EnergyAnalyzer
import DataTools
from pyTorchModel import pyTorchModel

#%%
label_prefix_project_name = "CustomNet"
analyse= EnergyAnalyzer(label_prefix_project_name, name_output_file="collab_compCustomModel_emissions.csv")

#initialized with default value
param_model = {"in_channels":3,        
             "conv1_out":6,
             "conv2_out":16,
             "kernel_size_conv":5,
             "pool_type":'max',       
             "pool_kernel":2,
             "fc1_out":120,
             "fc2_out":84,
             "size_input":32,
             "num_classes":10}

#%%
#Tests on Avg vs Max
'''analyse.set_new_project(label_prefix_project_name+"(AvgVsMax)")

for pool_type in ['max', 'avg']:
    param_model["pool_type"]=pool_type
    test_model = pyTorchModel(DataTools.usual_criterion, param_net_model=param_model)
    tracked_function = analyse.track_function(test_model.train, "with_max")
    tracked_function(DataTools.CIFAR10.trainloader, "trained_test_model.pth")
'''
#%%fc1 and fc2 
#analyse.display_data_axis("emissions", condFilter={"project_name":label_prefix_project_name+"(AvgVsMax)"})
analyse.display_data_axis("emissions", condFilter={"project_name":label_prefix_project_name+"(size fc1&2)"})
