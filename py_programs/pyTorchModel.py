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

from codecarbon import track_emissions
 

class pyTorchModel(mod):
    
    
    def __init__(self, given_criterion, given_Model = None, n_layers_outpout=3):
        super().__init__()
        print("end super")
        
        #initialisation of pyModel
        self.pyModel = None
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if given_Model is None :
            self.pyModel = pyTorchNet(n_layers_outpout)

        else :
            self.pyModel = given_Model()
        self.pyModel.to(self.device)
        # optimizer here because it use net properties
        self.optimizer = optim.SGD(self.pyModel.parameters(), lr=0.001, momentum=0.9)
        self.criterion = given_criterion
    
    #TODO delete after
    def train2(training_data, path_save_model,number_epoch = 2):
            print(training_data, path_save_model,number_epoch, " bbb")
    
    #@track_emissions(project_name="Test1_track_method_within_method")
    def train(self, training_data, path_save_model,number_epoch = 2):
        for epoch in range(number_epoch):  # loop over the dataset multiple times
        
            running_loss = 0.0
            for i, data in enumerate(training_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                #inputs, labels = data
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
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        
        print('Finished Training')
        self.save_model(path_save_model)

    def get_outputs(self, path,dataloader):
        self.pyModel.load_state_dict(torch.load(path,weights_only=True))
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

    def print_performances_categories(self, path, dataloader, classes):
        correct_predictions, total_predictions = self.analyse_performance(path, dataloader, classes)
        for classname, correct_count in correct_predictions.items():
            print("total pred ", classname, " / ", total_predictions[classname])
            if total_predictions[classname] <= 0 :
                continue
            accuracy = 100 * float(correct_count) / total_predictions[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    def print_performances_global(self, path, dataloader, classes):
        total = 0
        correct = 0

        correct_predictions, total_predictions = self.analyse_performance(path, dataloader, classes)
        for classname, correct_count in correct_predictions.items():
            correct += correct_predictions[classname]
            total += total_predictions[classname]

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def save_model(self, path):
        self.savePathModel = path
        torch.save(self.pyModel.state_dict(), path)
        print("Model saved in ", path)

    
