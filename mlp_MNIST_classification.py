import torch
import numpy as np 
import matplotlib.pyplot as plt


from torchvision import datasets
from torchvision.transforms import transforms
import inputImageAnalysis

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
test_data  = datasets.MNIST(root='data',  train=False,  download=True, transform=transform)

# prepare dataLoaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 30, num_workers = num_workers)
test_loader  = torch.utils.data.DataLoader(train_data, batch_size = 30, num_workers = num_workers)

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


