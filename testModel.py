import torch
import torch.nn as nn
import reusingTrainedModel
import mlp_network
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

class modelTester():
    def __init__(self):
        
        #how many samples per batch to load
        self.batch_size = 20

    def loadTestData(self):

        # number of subprocesses to use for data loading
        num_workers = 0

        # convert data to torch.FloatTensor
        transform = transforms.ToTensor()
        # choose the test datasets
        test_data = datasets.MNIST(root = 'data', train = False, transform = transform)
        # prepare test data loader
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, num_workers=num_workers)

        return test_loader
    
    def aggModelStats(self):

        # creating model object
        model_ = mlp_network.Net()

        # load the model to be tested
        loadObj = reusingTrainedModel.reuseTrainedModels()
        model_  = loadObj.loadModel(model_=model_)

        # initialise lists to monitor test loss and accuracy
        test_loss     = 0.0
        class_correct = list(0. for i in range(10))
        class_total   = list(0. for i in range(10))

        # collect test data
        test_loader_ = self.loadTestData()

        # define the criterion
        criterion = nn.CrossEntropyLoss()

        # prep model for training
        model_.eval() 

        for data, target in test_loader_:
            
            #forward pass: compute predicted outputs by passing inputs to the model
            output = model_(data)
            
            # calculate the loss
            loss = criterion(output, target)

            #updating test loss
            test_loss += loss.item() *data.size(0)

            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)

            # compare predictions to true lable
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            # calculate test accuracy for each object class
            for i in range(self.batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1


        # calculate and print avg test loss
        test_loss = test_loss/len(test_loader_.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(10):

            if(class_total[i] > 0):
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)'%(

                    str(i), 100*class_correct[i]/class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of NA%: N/A (no training examples)')

        print()
        print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (100.*(np.sum(class_correct)/np.sum(class_total)),np.sum(class_correct),np.sum(class_total)))


modelTestObj = modelTester()
modelTestObj.aggModelStats()
print("Testing Complpeted")
