#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:34:25 2025

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

from torchvision.models import resnet50, ResNet50_Weights, ResNet18_Weights
import torchvision.prototype.models as p
import torch
import DataTools
#incorrect import
#from torchvision.prototype.models import ResNet18_Weights

#works
model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")


#another test with torchvision
from torchvision.io import decode_image
#from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as models

from DataTools import imshow
#import tuto_remaster

PATH = './cifar_net.pth'
isTrained = True


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


#First tests
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# tests with other trained data
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()





# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
DataTools.imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))





all_models = models.list_models()
classification_models = models.list_models(module=models)
testmodel = models.resnet18(pretrained=True)

optimizer = optim.SGD(testmodel.parameters(), lr=0.001, momentum=0.9)


if not isTrained:
    print(all_models)
    print("/")
    print(classification_models)
    
    for epoch in range(2):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = testmodel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(i)
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    torch.save(testmodel.state_dict(), PATH)

testmodel.load_state_dict(torch.load(PATH, weights_only=True))
##Prediction Trained Model
outputs = testmodel(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

## Analyse performance on the dataset
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        images, labels = data
        # calculate outputs by running images through the network
        outputs = testmodel(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


#models that dont work
'''weights = ResNet18_Weights.DEFAULT
#weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
testmodel = models.ResNet(5, [5,5,5])

weights =  models.mobilenetv3.Weights(url, transforms, meta)  #ResNet50_Weights.IMAGENET1K_V2
testmodel = p.resnet50(weights=weights)
testmodel.eval()
#optimizer = optim.SGD(testmodel., lr=0.001, momentum=0.9)
'''

'''#img = decode_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

'''
'''
#doesnt work
m1 = get_model("mobilenet_v3_large", weights=None)
all_models = list_models()
classification_models = list_models(module=torchvision.models)


print(all_models)
print(classification_models)'''