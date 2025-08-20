#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 10:55:18 2025

@author: tim
"""
import sys
sys.path.append('../py_programs/')
import os

import unittest
import DataTools
import DataValidation
from pyTorchModel import pyTorchModel

from classEnergyAnalyzer import EnergyAnalyzer
from classTensorFunction import TensorNet
from classTensorModule import *
import classModelTester as Tester
from CSVReader import CSVfile

import numpy as np

from tflearn.datasets import titanic
from tflearn.data_utils import load_csv

import torchvision.models as ExampleModels

from torchsummary import summary


class testPyTorchModel():
    
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
    def test_noErrorTraining(self):
        self.test_model = pyTorchModel(DataTools.usual_criterion)
        self.test_model.train(DataTools.CIFAR10.train_inputs , "temp.pth", number_epoch=1)

    def test_noErrorTrainingExampleModel(self):
        list_model = DataTools.models.list_models(module=ExampleModels)
        self.test_model = pyTorchModel(DataTools.usual_criterion, given_Model=ExampleModels.get_model(list_model[0]))
        self.test_model.train(DataTools.CIFAR10.trainloader , "temp.pth", number_epoch=1)
    
    def test_noErrorSummary(self):
        #not my module but I am testing it to see if it can changes sometimes and if i use it correctly
        summary(DataTools.models.get_model("wide_resnet50_2"),input_size=DataTools.CIFAR10.datasets.size_input)
    
    def test_ErrorSummaryWringDatasetForModel(self):
        summary(DataTools.models.get_model("densenet121"),input_size=DataTools.CIFAR10.datasets.size_input)

    


class testDataTools(unittest.TestCase):
    
    # small test, overkilled but add layer of security
    def test_scientific_to_double(self):
        f = DataTools.scientificNotation_to_double
        self.assertEqual(f("1e-1"), 0.1)
        self.assertEqual(f("2e-10"), 0.0000000002)
        self.assertEqual(f("1e0"), 1)
        self.assertEqual(f("15.7e-5"), 0.000157)

class testEnergyAnalyzer(unittest.TestCase):
    
    def function_emissions(self):
        a = 1
        for i in range(10):
            a+=1
    
    def remove_temp_file(self, path):
        try :
            os.remove(path)
        except :
            print("file ", path," not found")
    
    def start_analyser_and_track_function(self):
        self.remove_temp_file("testEnergyAnalyzer.csv")
        test = EnergyAnalyzer("test", name_output_file="testEnergyAnalyzer.csv")
        f = test.track_function(self.function_emissions)
        for i in range(5):
            f()
        return test
    
    
    
    def test_noErrorCreation(self):
        test_CSVreader.remove_temp_file("testEnergyAnalyzer.csv")
        test = EnergyAnalyzer("test", name_output_file="testEnergyAnalyzer.csv")
    
    
    
    def test_noErrorTrackFunctionAndDisplay(self):
        test = self.start_analyser_and_track_function()
        test.display_data_axis("duration")

    '''def test_displayGraphKnownXAxis(self):
        test = self.start_analyser_and_track_function()
        test.display_data_axis("duration", x_axis="x_axis")
     '''
    def test_displayGraphKnownXcol(self):
        test = self.start_analyser_and_track_function()
        test.display_data_axis("duration",x_col = ['1','2','3','4','5'])        
                  
    def test_displayGraphKnownXAxisAndXcol(self):
        test = self.start_analyser_and_track_function()
        test.display_data_axis("duration", x_axis="x_axis", x_col = ['1','2','3','4','5'])
        
            
class testTensorFlow():
    
    def test_noErrorModel(self):
        test_model = TensorNet([6,32,32,2])
    
    def test_noErrorMultilayer(self):
        l = []
        for i in range(10):
            l.append(5)
            test_model = TensorNet(l)
    
    def create_sample_tfModules(self):
        return [tfInputData(shape=[None, 8]), tfFullyConnected(4), tfFullyConnected(40), tfDropout(0.4), tfFullyConnected(50)]
    
    def test_noErrorModule(self):
        test_net = TensorNet(layers=self.create_sample_tfModules())
        test_net.print_modules()
    
    def test_tensorTraining(self):
        test_net = TensorNet(layers=[tfInputData(shape=[None, 6]), tfFullyConnected(32), tfFullyConnected(32), tfFullyConnected(2)])

        titanic.download_dataset('titanic_dataset.csv')
        data, labels = load_csv('titanic_dataset.csv', target_column=0,
                       categorical_labels=True, n_classes=2)
        
        
        def preprocess(passengers, columns_to_delete):
            # Sort by descending id and delete columns
            for column_to_delete in sorted(columns_to_delete, reverse=True):
                [passenger.pop(column_to_delete) for passenger in passengers]
            for i in range(len(passengers)):
                # Converting 'sex' field to float (id is 1 after removing labels column)
                passengers[i][1] = 1. if data[i][1] == 'female' else 0.
            return np.array(passengers, dtype=np.float32)
        
        data = preprocess(data, [1,2,3,4,5,6])

        test_net.train(data, labels)
    
    

class testTensorModules():
    
    def test_noErrorInputData(self):
        #tfInputData(45)
        tfInputData(shape=[None, 78])
    
    def test_noErrorFullyConnected(self):
        tfFullyConnected(7)
        tfFullyConnected(30)
    
    def test_noErrorDropout(self):
        tfDropout(0.5)
    
    def test_WrongRangeDropout(self):
        t=tfDropout(4)
        self.assertEqual(t.isValid(), False)
    
    
class test_DataValidation(unittest.TestCase):
    
    def test_isCorrectType(self):
        f = DataValidation.isCorrectType
        self.assertTrue(f(5, int))
        self.assertTrue(f("ds", str))
        self.assertTrue(f(5.1, float))
        self.assertFalse(f(5.1, int))
        self.assertFalse(f("eee", int))
        self.assertFalse(f("eee", float))
        self.assertFalse(f(1, bool))

    
    def test_addSuffixIfNecessary(self):
        f = DataValidation.addSuffixIfNecessary
        self.assertEqual(f("test.txt", "txt"), "test.txt")
        self.assertEqual(f("test.txt", ".txt"), "test.txt")
        self.assertEqual(f("test", ".txt"), "test.txt")
        self.assertEqual(f("test", "txt"), "test.txt")
        
class test_ModelTester():
    
    def getAnalyze(self):
        return EnergyAnalyzer("temp", name_output_file="temp.csv")
    
    def test_testGpu(self):
        Tester.testGpu(self.getAnalyze(), 2)
        
    def test_pyTorch_series_train_and_track_emissions(self):
        Tester.pyTorch_series_train_and_track_emissions(self.getAnalyze(), [0], DataTools.CIFAR10.train_inputs, "uniTest",1, False)

class test_CSVreader(unittest.TestCase):
    
    
    temp_file = "temp.csv"
    
    def remove_temp_file(self, path = temp_file):
        try :
            os.remove(path)
        except :
            print("file ", path," not found")
    
    def test_noErrorWriting(self):
        self.remove_temp_file()
        f = CSVfile.create_file(self.temp_file)
        f.add_column("test", [1,2,3])
        self.assertEqual(f.get_categories(), ['id_col', 'test'])
        self.assertEqual(f.get_columns_value("test"), ['1','2','3'])
    
    def test_noErrorPrintingLabel(self):
        self.remove_temp_file()

        f = CSVfile.create_file(self.temp_file)
        f.add_column("test_label", ['a','b','a','b','a', 'a', 'b'])
        f.add_column("test_value", [1,2,3,4,5,6,7])
        self.assertEqual(f.get_categories(), ['id_col', 'test_label', 'test_value'])
        self.assertEqual(f.get_columns_value("test_label"), ['a','b','a','b','a', 'a', 'b'])
        self.assertEqual(f.get_columns_value("test_value"), ['1','2','3','4','5','6','7'])
        self.assertEqual(f.get_columns_value("test_value", {"test_label":'a'}), ['1','3','5','6'])
    
    
    def test_noErrorPrintingLabelAddingColumnsAllAtOnce(self):
        self.remove_temp_file()

        f = CSVfile.create_file(self.temp_file)
        f.add_columns(["test_label","test_value"], [['a','b','a','b','a', 'a', 'b'], [1,2,3,4,5,6,7]])
        self.assertEqual(f.get_categories(), ['id_col', 'test_label', 'test_value'])
        self.assertEqual(f.get_columns_value("test_label"), ['a','b','a','b','a', 'a', 'b'])
        self.assertEqual(f.get_columns_value("test_value"), ['1','2','3','4','5','6','7'])
        self.assertEqual(f.get_columns_value("test_value", {"test_label":'a'}), ['1','3','5','6'])
    
    
    def test_noErrorWritingAndREtrievingData(self):
        self.remove_temp_file()


        f = CSVfile.create_file(self.temp_file)
        f.add_column("test_label", ['a','b','a','b','a', 'a', 'b'])
        f.add_column("test_value", [1,2,3,4,5,6,7])
        f.write_data_to_file()
        
        f2 = CSVfile.create_file(self.temp_file)
        self.assertEqual(f2.get_categories(), ['id_col', 'test_label', 'test_value'])
        self.assertEqual(f2.get_columns_value("test_label"), ['a','b','a','b','a', 'a', 'b'])
        self.assertEqual(f2.get_columns_value("test_value"), ['1','2','3','4','5','6','7'])
        self.assertEqual(f2.get_columns_value("test_value", {"test_label":'a'}), ['1','3','5','6'])
 

unittest.main(verbosity=20)
