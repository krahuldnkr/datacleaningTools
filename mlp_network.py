import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class Net(nn.Module):
    
    def __init__(self):
        """_summary_
        """
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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
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



