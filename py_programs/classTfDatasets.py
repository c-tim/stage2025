#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 10:12:54 2025

@author: tim
"""
import tflearn
from tflearn.data_utils import load_csv


class tfDatasets():
    
    def __init__(self, path_file, module):
        
        module.download_dataset(path_file)
        self.inputs = load_csv(path_file, target_column=0,
                                categorical_labels=True, n_classes=2)
        