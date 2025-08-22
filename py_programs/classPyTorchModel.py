#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:28:39 2025

@author: tim
"""

from Model import Model as mod
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch
import torchvision
from classPytorchNet import pyTorchNet
from classCustomCNET import pyTorchCNet

from codecarbon import track_emissions
 
import DataTools
import DataValidation

class pyTorchModel(mod):
    
    def createPyTorchNet(given_criterion, n_layers_output):
        pyModel = pyTorchNet(n_layers_output)
        return pyTorchModel(given_criterion, pyModel)
    
    def createPyTorchCNet(given_criterion, param_net_model):
        pyModel = pyTorchCNet(**param_net_model)   
        return pyTorchModel(given_criterion, pyModel)
    
    def createPyTorchExampleNet(given_criterion, given_Model):
        return pyTorchModel(given_criterion, given_Model)


    def __init__(self, given_criterion, given_Model):
        super().__init__()        
        #initialisation of pyModel
        self.pyModel = given_Model
        
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cuda")
        self.pyModel = self.pyModel.to(self.device)
        # optimizer here because it use net properties
        self.optimizer = optim.SGD(self.pyModel.parameters(), lr=0.001, momentum=0.9)
        self.criterion = given_criterion
     
    def train(self, training_data, path_save_model,number_epoch = 2, verbose = True):
        """
        Train the pyTorch model

        Parameters
        ----------
        training_data : dataset
        path_save_model : str
            DESCRIPTION.
        number_epoch : TYPE, optional
            DESCRIPTION. The default is 2.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """      
        DataValidation.addSuffixIfNecessary(path_save_model, ".pth")
        
        for epoch in range(number_epoch):  # loop over the dataset multiple times
        
            running_loss = 0.0
            for i, data in enumerate(training_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.pyModel(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999 and verbose:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        
        print('Finished Training')
        self.save_model(path_save_model)

    def get_outputs(self, path,dataloader):
        #self.pyModel.load_state_dict(torch.load(path,weights_only=True))


        self.pyModel.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        self.pyModel = self.pyModel.to(self.device)
        self.pyModel.eval()
        # again no gradients needed -> this method< remove backward call and reduce memory
        outputs_group =  []
        labels_group = []
        with torch.no_grad():
            for data in dataloader:
                #inputs, label = data
                inputs, label = data[0].to(self.device), data[1].to(self.device)

                #images, labels = data

                #inputs, labels = data[0].to(device), data[1].to(device)
                outputs_group.append(self.pyModel(inputs))
            for data in dataloader : 
                #inputs, label = data
                inputs, label = data[0].to(self.device), data[1].to(self.device)

                labels_group.append(label)
            return labels_group, outputs_group
        
    def get_predictions(self, outputs):
        predictions = []
        for output in outputs:
            _, prediction = torch.max(output, 1)
            predictions.append(prediction)
        return predictions
    
    def analyse_performance(self, path, dataloader, classes):
        """
        Returns the rate of the correc predictions based on the dataloader and the classes provided

        """
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        labels_grouped, outputs_grouped = self.get_outputs(path, dataloader)
        predictions_grouped = self.get_predictions(outputs_grouped)
        
        for label, prediction in zip(labels_grouped,predictions_grouped):
            # collect the correct predictions for each class
            for label_sample, prediction_sample in zip(label, prediction):
                if label_sample == prediction_sample:
                    correct_pred[classes[label_sample]] += 1
                total_pred[classes[label_sample]] += 1
        return correct_pred, total_pred
    
    def get_global_performance(self, correct_predictions,total_predictions ):
        total = 0
        correct = 0

        for classname, correct_count in correct_predictions.items():
            correct += correct_predictions[classname]
            total += total_predictions[classname]
        return correct, total


    def print_performances_categories(self, path, dataloader, classes):
        correct_predictions, total_predictions = self.analyse_performance(path, dataloader, classes)
        for classname, correct_count in correct_predictions.items():
            print("total pred ", classname, " / ", total_predictions[classname])
            if total_predictions[classname] <= 0 :
                continue
            accuracy = 100 * float(correct_count) / total_predictions[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    def print_performances_global(self, path, dataloader, classes):

        correct_predictions, total_predictions = self.analyse_performance(path, dataloader, classes)
        correct, total = self.get_global_performance(correct_predictions, total_predictions)
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def save_model(self, path):
        self.savePathModel = path
        torch.save(self.pyModel.state_dict(), path)
        print("Model saved in ", path)
    


