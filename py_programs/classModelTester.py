#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:20:08 2025

@author: tim
"""


from classEnergyAnalyzer import EnergyAnalyzer
import DataTools
import torchvision.models as ExampleModels
from pyTorchModel import pyTorchModel


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
    analyse.display_data_axis("emissions", x_axis="Number iterations", x_col=test_iterations, condFilter={"project_name":"evaluate_performance_biais"})


def series_train_and_track_model_emissions(analyser:EnergyAnalyzer, model, dataset, path_save_model,list_info_tracking):
    for i in range(len(list_info_tracking)):
        _train_and_track_model_emissions(analyser, model, dataset, path_save_model, list_info_tracking[i])

def pyTorch_series_train_and_track_emissions(analyser:EnergyAnalyzer, list_id_model,dataloader,list_info_tracking, number_epoch):
    tested_model = {}
    sucessfully_tested_model={}
    unsucessfully_tested_model={}
    list_model = DataTools.models.list_models(module=ExampleModels)
    for id_tested in list_id_model:
        print(list_model[id_tested], " added")
        label_model = list_model[id_tested]
        ref_model = ExampleModels.get_model(list_info_tracking+":"+list_model[id_tested])
        tested_model[list_model[id_tested]] =  ref_model
        _train_and_track_model_emissions(analyser, )
        try :
            model = pyTorchModel(DataTools.usual_criterion, given_Model= ref_model)
            tracked_function = analyser.track_function(model.train, label_model)
            tracked_function(dataloader, str(label_model)+".pth", number_epoch = number_epoch)         
            sucessfully_tested_model[id_tested] = label_model
        except:
            print(label_model," is incompatible with this configuration")
            unsucessfully_tested_model[id_tested] = label_model


#TODO complete the series of tests
def _train_and_track_model_emissions(analyser:EnergyAnalyzer, model, dataset, path_save_model,info_tracking):
    tracked_function = analyser.track_function(model.train, info_tracking)
    tracked_function(dataset, path_save_model)
