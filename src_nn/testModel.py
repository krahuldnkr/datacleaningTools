import torch
import torch.nn as nn
import reusingTrainedModel
import mlp_network
import numpy as np
import matplotlib.pyplot as plt

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
    
    def testModelStats(self):

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
            correct = pred.eq(target.data)

            # calculate test accuracy for each object class
            for i in range(self.batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            
        self.aggModelStats(test_loss, class_correct, class_total, test_loader_=test_loader_)

    def aggModelStats(self, test_loss, class_correct, class_total, test_loader_):

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

    def visualizeTestResults(self):

        test_loader = self.loadTestData()
        
        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, labels = next(dataiter)

        # creating model object
        model_ = mlp_network.Net()

        # load the model to be tested
        loadObj = reusingTrainedModel.reuseTrainedModels()
        model_  = loadObj.loadModel(model_=model_)

        # get sample outputs
        output = model_(images)
        
        # convert output probabilities to predicted class
        _, preds = torch.max(output, 1)
        
        # prep images for display
        images = images.numpy()

        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(25,4))

        for idx in np.arange(20):
            ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                         color = ("green" if preds[idx] == labels[idx] else "red"))
        plt.show()

modelTestObj = modelTester()
modelTestObj.testModelStats()
print("Testing Completed")

modelTestObj.visualizeTestResults()
