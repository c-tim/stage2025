#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 10:55:18 2025

@author: tim
"""
import sys
sys.path.append('../py_programs/')

import unittest
import DataTools
from pyTorchModel import pyTorchModel
from classEnergyAnalyzer import EnergyAnalyzer

class testPyTorchModel(unittest.TestCase):
    
    def test_noErrorPytorchNet(self):
        self.test_model = pyTorchModel(DataTools.usual_criterion)
        #self.assertEqual(1, 1)

    def test_noErrorPytorchCNetChangeConfig(self):
        original_config ={"in_channels":3,        
             "conv1_out":6,
             "conv2_out":16,
             "kernel_size_conv":5,
             "pool_type":'max',       
             "pool_kernel":2,
             "fc1_out":120,
             "fc2_out":84,
             "size_input" : 32,
             "num_classes":10}
        config = original_config.copy()
        for item in config:
            config = original_config.copy()
            if item == "pool_type":
                combinaisons = ['avg', 'max']
            else :
                original_value = original_config[item]
                combinaisons = [int(original_value/5 * i) for i in range(1,10)]
            for test_combinaisons in combinaisons :
                config[item] = test_combinaisons
                #print("test ", item, " with ", test_combinaisons)
                test_model = pyTorchModel(DataTools.usual_criterion, param_net_model=config)

class testDataTools(unittest.TestCase):
    
    # small test, overkilled but add layer of security
    def test_scientific_to_double(self):
        f = DataTools.scientificNotation_to_double
        self.assertEqual(f("1e-1"), 0.1)
        self.assertEqual(f("2e-10"), 0.0000000002)
        self.assertEqual(f("1e0"), 1)
        self.assertEqual(f("15.7e-5"), 0.000157)

class testEnergyAnalyzer(unittest.TestCase):
    
    def test_noErrorCreation(self):
        test = EnergyAnalyzer("test")
        
    #TODO add test to check suffixe for file name inputs
    



                
                
        
unittest.main()