"""Defines architecture of MLP network for MNIST classification.

Returns:
    double : output of the nn model.
"""
import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class Net(nn.Module):
    
    def __init__(self):
        """Initializes MLP network."""
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512

        #linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28*28, hidden_1)
        #linear layer (hidden_1 -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        #linear layer (hidden_2-> output)
        self.fc3 = nn.Linear(hidden_2, 10)

        # dropout layer (p = 0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Compiles linear parts with the rectification units.

        Args:
            x (image matrix): input image data.

        Returns:
            vector : predicted output for each of the 10 digits.
        """
        # flatten image input
        # changes the shape of input tensor
        x = x.view(-1, 28 * 28)
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
    
#  Training Loss: 0.004936         Validation Loss: 0.085157   Number of Epochs = 70    dropOut= 0.2
#  --> here we found out that test accuracy was 97% (expected : 98%)
#  --> To improve, now using epochs = 50 (early stopping to improve test accuracy).
#  __>  choose early stopping as first strategy because, from observation validation loss seemed to saturate early.     
# with 50, 40, still test accuracy was 97%.
    # Accuracy increased back to 98% when data for validation is added back for training. --> MORE THE DATA TO TRAIN ON, MORE THE ACCURACY.
    # but challenge is can we increase the loss frm 97 to 98 without increeasing the data.

# Possible Techniques
    # image preprocessing --> add some noise to the input data/ rotate, crop and scale. etc.
    # applying dropouts (try combinations).
    # play with the architecture.

# with dropout = 0.3, test accuracy --> 9813/10000.
# with dropout = 0.5, test accuracy --> 9807/10000.     
# image normalisation reduces the accuracy.