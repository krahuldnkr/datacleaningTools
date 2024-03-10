"""Saving and Loading Models."""
# We can save trained networks then load them later to TRAIN MORE or use them for predictions.
import torch

class reuseTrainedModels:

    def saveModel(self, model_):

        print("Our model: \n\n", model_, '\n')
        print("The state dict keys: \n\n", model_.state_dict().keys())
        torch.save(model_.state_dict(), 'checkpoint.pth')
    
    def loadModel(self, model_):
        # we can load the state_dict_
        state_dict_ = torch.load('checkpoint.pth')
        print(state_dict_.keys())

        # load the state dict in to the network
        model_.load_state_dict(state_dict_)

        return model_
