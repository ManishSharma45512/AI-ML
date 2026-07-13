#Import neccessary Library

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Tranform and load composed data

data_transform=torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])#composing data

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=data_transform)#downloading the composed data set using torchvision
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

trainset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=data_transform) # Changed train=True to train=False for test set and download=True to ensure it's available.
testloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=False,num_workers=2) # Changed trainloader to testloader


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Creating a Convolutional Neural Network(CNN) using PyTorch's Neural Network(NN) library

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1=nn.Conv2d(3,6,5)
    self.pool=nn.MaxPool2d(2,2)
    self.conv2=nn.Conv2d(6,16,5) # Added conv2 layer
    self.fc1=nn.Linear(16*5*5,120)
    self.fc2=nn.Linear(120,84)
    self.fc3=nn.Linear(84,10)


  def forward(self,x):
    x=self.pool(F.relu(self.conv1(x)))
    x=self.pool(F.relu(self.conv2(x)))
    x=x.view(-1,16*5*5) # Flatten for fully connected layers
    x=F.relu(self.fc1(x)) # Connect to fc layers
    x=F.relu(self.fc2(x))
    x=self.fc3(x)
    return x




Net=Net() #constructor


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Finding Loss of the Model and optimizing to minimum loss


criterion=nn.CrossEntropyLoss()#Finding Loss
optimizer=optim.SGD(Net.parameters(),lr=0.001,momentum=0.9)#optimizing to minimum loss

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Training Of Model Looping twice on the dataset

for epoch in range(2):
  running_loss=0.0
  for i,data in enumerate(trainloader,0):
    #get the inputs; data is list of [inputs,labels]
    inputs,labels=data

    #zero the parameter gradients
    optimizer.zero_grad()

    #forward + backward + optimizer
    outputs=Net(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    #printing our statistics
    running_loss+=loss.item()
    if i%2000==1999:
      print(f'[{epoch + 1},{i + 1:5d}] loss: {running_loss / 2000 :.3f}')
      running_loss=0.0



print("Training Finished")

'''
Output:


[1, 2000] loss: 2.185
[1, 4000] loss: 1.835
[1, 6000] loss: 1.634
[1, 8000] loss: 1.567
[1,10000] loss: 1.514
[1,12000] loss: 1.456
[2, 2000] loss: 1.403
[2, 4000] loss: 1.359
[2, 6000] loss: 1.319
[2, 8000] loss: 1.351
[2,10000] loss: 1.294
[2,12000] loss: 1.271
Training Finished

'''

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Testing data on unseen data

correct=0
total=0
with torch.no_grad():
  for data in testloader:
    images,labels = data
    outputs=Net(images)
    _,predicted=torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()


print(f'Accuracy of the network on the label 10000 test images : {100*correct/total}%')


#Output: Accuracy of the network on the label 10000 test images : 55.16%

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Saving the Model

PATH='./cifar_net.pth'
torch.save(Net.state_dict(), PATH)

#--------------------------------------------------XXXXXXXXXXXXXXXXXXXXXXX---------------------------------------XXXXXXXXXXXXXXXXXXXXXX------------------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXX-------------------------------------------------------------------------------
