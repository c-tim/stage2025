#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 11:12:34 2025

@author: tim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 18:03:53 2025

@author: tim
"""

from Model import Model as m
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch
import torchvision

## This class will be used as an object contained by pytorchModel
## Parts of the initialisation have been generated 
class pyTorchCNet(nn.Module):
    """
    A customizable model of Pytorch used by the class pyTorchModel.
    """
    def __init__(self,
             in_channels=3,        
             conv1_out=6,
             conv2_out=16,
             kernel_size_conv=5,
             pool_type='max',       
             pool_kernel=2,
             fc1_out=120,
             fc2_out=84,
             size_input = 32,
             num_classes=10):
        super().__init__()    
        self.calculate_net_model(in_channels=3,        
                     conv1_out=6,
                     conv2_out=16,
                     kernel_size_conv=5,
                     pool_type='max',       
                     pool_kernel=2,
                     fc1_out=120,
                     fc2_out=84,
                     size_input = 32,
                     num_classes=10)

        
    def calculate_net_model (self,
             in_channels=3,        
             conv1_out=6,
             conv2_out=16,
             kernel_size_conv=5,
             pool_type='max',       
             pool_kernel=2,
             fc1_out=120,
             fc2_out=84,
             size_input = 32,
             num_classes=10):
        
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size_conv)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size_conv)

        if pool_type == 'max':
            self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(pool_kernel, pool_kernel)
        else:
            print("la pool doit être 'max' ou 'avg'")

        conv_output_size = self.calculate_conv_output_size(size_input,in_channels, kernel_size_conv, pool_kernel)

        self.fc1 = nn.Linear(conv2_out * conv_output_size * conv_output_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, num_classes)
        
        self.forward_function_used = self.reLu_forward

    def calculate_conv_output_size(self, size_input,in_channels, kernel_size, pool_kernel):
        size_input = (size_input - kernel_size + 1) // pool_kernel  
        size_input = (size_input - kernel_size + 1) // pool_kernel 
        return size_input
    
    def reLu_forward(self, x ):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_function_used(x)

