#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:08:19 2025

@author: tim
"""


# Modules python a importer :
# Classes
from classEnergyAnalyzer import EnergyAnalyzer
from classPyTorchModel import pyTorchModel
from classDatasets import Dataset
from CSVReader import CSVfile

# Modules
import classModelTester as Tester
import DataTools

# Librairies externes
import torchvision
import torchvision.models as ExampleModels
import matplotlib.pyplot as plt

analyzer = EnergyAnalyzer("testPytorch", name_output_file="./testgeneral.csv")
analyzer.recup_file_result()
file_analysis = analyzer.csvResult
list_model = DataTools.models.list_models(module=ExampleModels)
print(list_model)


#get id from the column project_name
worked_models_parameters = []
list_id_worked_models = []
list_total_layer = []
all_labels_models = []


col_project_name = file_analysis.get_column("project_name")
for line in col_project_name:
  label_model = line.split("CollabLinux:train:CollabLinux(")[1]
  for i in range(len(list_model)):
    if list_model[i]==label_model:
      list_id_worked_models.append(i)
      all_labels_models.append(str(list_model[i]))

      break
  
resultFile =  CSVfile.create_file("./result_perf.csv")

resultFile.add_column("name_machine", all_labels_models)


indices_to_skip = []
list_ratio_performances = []
for i in range(len(all_labels_models)) :
  try :
    m = pyTorchModel.importPyTorchExampleNet(DataTools.usual_criterion, DataTools.models.get_model(str(all_labels_models[i]))) # on ilporte un autre modele apres donc pas impportant (TODO corriger pr ne plus faire ca)
    path = "./data_gathered/all_pth/"+str(all_labels_models[i])+".pth"
    print("test output on ", path)
    correct_predictions, total_predictions = m.analyse_performance(path, DataTools.CIFAR10.test_inputs, DataTools.usual_classes)
    correct,total= m.get_global_performance(correct_predictions, total_predictions)

  except :
    print("failed")
    indices_to_skip.append(i)
    list_ratio_performances.append(-1)

    continue
  list_ratio_performances.append(correct/total)
print("Modele not found for ", indices_to_skip)

resultFile.add_column("ratio_perf", list_ratio_performances)
