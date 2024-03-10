import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CnnNet(nn.Module):

    def __init__(self, weight):
        super(CnnNet, self).__init__()

        # initialize weights of the convolutional layer using weights of 4 defined filters
        #print("shape of the weight",weight.shape)
        k_depth, k_height, k_width = weight.shape[1:]
        #print("Sizes: \n", weight.shape)

        # assume there are 4 grayscale filters
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_depth, k_height, k_width), bias=False, stride=1)
        #
        self.conv.weight = torch.nn.Parameter(weight)

        self.pool = nn.MaxPool2d(2,2)

        # Adding Fully Connected Layers
                # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512

        #linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(4*110*110, hidden_1)
        #linear layer (hidden_1 -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        #linear layer (hidden_2-> output)
        self.fc3 = nn.Linear(hidden_2, 2)

        # dropout layer (p = 0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_img):
        #print("input image size: ",input_img.shape)
   
        conv_x = self.conv(input_img)
        activated_x = F.relu(conv_x)
        pooled_x = self.pool(activated_x)
        # returns both layers
        # Need to add fully-connected layers

        # flatten image input
        # changes the shape of input tensor
        x = pooled_x.view(-1, 4*110 * 110)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, wit hrelu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

