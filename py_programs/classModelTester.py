#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:20:08 2025

@author: tim
"""


from classEnergyAnalyzer import EnergyAnalyzer

def testGpu(analyse : EnergyAnalyzer, power_ten_iterations : int):
    analyse.set_new_project("evaluate_performance_biais")
    def consuming_function(loop_count :int):
      i=0
      for loop in range(loop_count):
        for a in range(100):
          for c in range(100):
            i+=1
    test_iterations = [10** i for i in range(power_ten_iterations)]

    for n_loop in test_iterations:
      print("Test : ", n_loop*10000, " iterations")
      tracked_function = analyse.track_function(consuming_function, "number iterations = "+str(n_loop*10000))
      tracked_function(n_loop)
    analyse.display_data_axis("emissions", x_axis="Number iterations", x_col=test_iterations, condFilter={"project_name":"evaluate_performance_biais6"})
    
#TODO complete the series of tests
def train_and_tracke_model_emissions(analyser:EnergyAnalyzer, model):
    tracked_function = analyse.track_function(test_model.train, "(60/42)x"+ str(i))
    tracked_function(DataTools.CIFAR10.trainloader, "trained_test_model.pth")
