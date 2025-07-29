#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 09:47:08 2025

@author: tim
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from codecarbon import track_emissions
#import ABC
import DataTools

#project variables
isTrained = False
PATH = './cifar_net.pth'

class Net(nn.Module):
    
    #basic modifier to test net scale performan
    def __init__(self, modifier=1, onCUDA = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3*modifier, 6*modifier, 5*modifier)
        self.pool = nn.MaxPool2d(2*modifier, 2*modifier)
        self.conv2 = nn.Conv2d(6*modifier, 16*modifier, 5*modifier)
        self.fc1 = nn.Linear(16 * 5 * 5*modifier, 120*modifier)
        self.fc2 = nn.Linear(120*modifier, 84*modifier)
        self.fc3 = nn.Linear(84*modifier, 10*modifier)
        self.useCuda = onCUDA
        
        # optimizer here because it use net properties
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @track_emissions()
    def train(self, training_data, path_save_model,number_epoch = 2):
        for epoch in range(number_epoch):  # loop over the dataset multiple times
        
            running_loss = 0.0
            for i, data in enumerate(training_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
        
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        
        print('Finished Training')
        torch.save(net.state_dict(), path_save_model)
    
    def analyse_performance_categories(self, dataloader, classes):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    


    def analyse_performance_global(self, dataloader, classes):
        ## Analyse performance on the dataset
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_data:
                inputs, labels = data[0].to(device), data[1].to(device)

                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')



    
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


#First tests
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
'''trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)'''
'''testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)'''

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#Test with another model
ROOT = '.data'

train_data = torchvision.datasets.MNIST(root=ROOT,
                            train=True,
                            download=True)

mean = train_data.data.float().mean() / 255
std = train_data.data.float().std() / 255
train_transforms = transforms.Compose([
                            transforms.RandomRotation(5, fill=(0,)),
                            transforms.RandomCrop(28, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[mean], std=[std])
                                      ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[mean], std=[std])
                                     ])



train_data = torchvision.datasets.MNIST(root=ROOT,
                            train=True,
                            download=True,
                            transform=train_transforms)

test_data = torchvision.datasets.MNIST(root=ROOT,
                           train=False,
                           download=True,
                           transform=test_transforms)




# get some random training images
dataiter = iter(train_data)
images, labels = next(dataiter)

# show images
DataTools.imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(1)))



#Network Training
net = Net()

net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if not isTrained :
        net.train(train_data, PATH, 2)
        print("\n\n\n----------")

dataiter = iter(test_data)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
net.load_state_dict(torch.load(PATH, weights_only=True))


##Prediction Trained Model
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

net.analyse_performance_categories(train_data, classes)
net.analyse_performance_global(train_data, classes)


