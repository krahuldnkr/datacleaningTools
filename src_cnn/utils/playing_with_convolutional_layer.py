import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# define a neural network with a single convolutional layer with four filters
class Net(nn.Module):

    """Some Information about MyModule"""
    def __init__(self, weight):
        super(Net, self).__init__()

        # initialize weights of the convolutional layer using weights of 4 defined filters
        k_height, k_width = weight.shape[2:]
        print("sizes: \n", weight.shape)
        # assumes there are 4 grayscale filters
        self.conv        = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias = False)
        self.conv.weight = torch.nn.Parameter(weight) 

        # then we add a maxpooling layer with a kernel size of 2 x 2. A maxpooling layer 
        # reduces the x-y size of an input and only keeps the most active pixel values;
        # reducing the x-y size of a patch by a factor of 2. Only the maximum pixel values
        # in 2 x 2 remain in the new pooled output.

        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # applies pooling layer
        pooled_x = self.pool(activated_x)

        # return both layers
        return conv_x, activated_x, pooled_x 

# instantiate the model and set the weights
filter_vals = np.array([[-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1 ,1 ,1]])

print('Filter shape: ', filter_vals.shape)

# Defining four different filters,
# all of which are linear combinations of the 'filter vals' defined above

# define four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3

filters = np.array([filter_1, filter_2, filter_3, filter_4])
print('Filter 1: \n', filter_1)

weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
print("numpy2torch dim: ", weight)

model = Net(weight)

# print out the layer in the network
print(model)