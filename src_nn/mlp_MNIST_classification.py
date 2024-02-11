import torch
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import transforms
import inputImageAnalysis
import mlp_network
import reusingTrainedModel

# Note:: STEP 1
# number of subprocesses to use for data loading.
num_workers = 0
# how many samples per batch to load.
batch_size = 20

# convert data to torch.FloatSensor.
# No input image pre-processing required for the MNIST dataset.
transform = transforms.ToTensor()

# choose the training and test datasets.
train_data = datasets.MNIST(root= 'data', train = True, download=True, transform=transform)
test_data  = datasets.MNIST(root='data',  train =False,  download=True, transform=transform)

# prepare dataLoaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 30, num_workers = num_workers)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size = 30, num_workers = num_workers)

# Note:: STEP 2 
# 1. Take a look at the data and make sure it is loaded correctly.
# 2. Make initial observations about patterns in the data.

# Obtain one batch of training images.
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
# below we define our figure and make it an 25*4 inch^2 image.
#analyserObj = inputImageAnalysis.inputImageAnalysis()
#analyserObj.multipleImagePlotter(images, labels)

# Uncomment and pass a single image for further analysis
#analyserObj.analyser(images[1]) 

# calling nn model
# Initializing the network
model = mlp_network.Net()
print(model)

# Cross Entropy Function = applies softmax to the output layer and then calculate log loss.
# Specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Training and Learning from a batch of data
# 1. clear the gradients of all optimized variables
# 2. Forward pass: compute predicted outputs by passing inputs to the model.
# 3. Calculate the loss
# 4. Backward Pass : Compute gradient of the loss with respect to model parameters.
# 5. Perform a single optimisation step. (parameter update)
# 6. Update average training loss.

# Number of Epochs to train the model.
n_epochs = 70

# prep model for training
model.train() 

for epoch in range(n_epochs):

    # monitor train loss
    train_loss = 0.0

    #####################################
    # Train the model
    #####################################

    for data, target in train_loader:

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model.
        output = model(data)

        # calculate the loss
        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimisation step (parameter update)
        optimizer.step()

        # update running training loss
        train_loss += loss.item()*data.size(0)

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch+1, train_loss))
    
# Saving the trained model
saveModelObj = reusingTrainedModel.reuseTrainedModels()
saveModelObj.saveModel(model_=model)

