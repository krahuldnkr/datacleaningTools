# Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt 

# PyTorch dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from cnn_network import CnnNet

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5 # un-normalize
    plt.imshow(np.transpose(img, (1, 2, 0))) # convert from tensor image

def visImageBatch(images):
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])
        print("labels: ",labels[idx] )
    plt.show() 

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# Data transform to convert data to a tensor and apply normalisation
# augment train and validation datset with RandomHorizontalFlip and RandomRotation
train_transform = transforms.Compose([transforms.Resize(255),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transform = transforms.Compose([transforms.Resize(255),
                                                                            transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                     
                                     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]) 

# Choose the training and test datasets
train_data = datasets.ImageFolder('../Cat_Dog_data/Cat_Dog_data/train', transform=train_transform)
test_data = datasets.ImageFolder('../Cat_Dog_data/Cat_Dog_data/test', transform=test_transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare dataloaders (combine data set and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = valid_sampler, num_workers=num_workers)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# Visualize a Batch of Training Data
# Obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display
print("image dimension: ",images.shape)

# specify the image classes
classes = ['cat', 'dog']


# plot the images in the batch, along with the corresponding labels
#visImageBatch(images)

# Lets look at the normalised RGB color channels as three separate, grayscale intensity images
# print(images.shape)
# rgb_img = np.squeeze(images[3])
# print(rgb_img.shape)
# channels = ['red channel', 'green channel', 'blue channel']

# fig = plt.figure(figsize = (36, 36))
# for idx in np.arange(rgb_img.shape[0]):
    
#     ax = fig.add_subplot(1, 3, idx+1, xticks =[], yticks=[])
#     img = rgb_img[idx]
#     ax.imshow(img, cmap='gray')
#     ax.set_title(channels[idx])
#     width, height = img.shape
#     thresh = img.max()/2.5
    
#     for x in range(width):
#         for y in range(height):
#             val = round(img[x][y], 2) if img[x][y] != 0 else 0
#             ax.annotate(str(val), xy=(y,x),
#                         horizontalalignment='center',
#                         verticalalignment='center', size=8,
#                         color = 'white' if img[x][y] < thresh else 'black')
#     plt.show()

filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1],[-1, -1, 1, 1],[-1, -1, 1, 1]])

# Defining Four different filters,
# all of which are linear combinations of the 'filter_vals' defined above

# define four fiters
filter_1 = [filter_vals, filter_vals, filter_vals]
filter_2 = [-filter_vals, -filter_vals, -filter_vals]
filter_3 = [filter_vals.T, filter_vals.T, filter_vals.T]
filter_4 = [-filter_vals.T,-filter_vals.T, -filter_vals.T]
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# instantiate the model and set the weights
weight = torch.from_numpy(filters).type(torch.FloatTensor)
model = CnnNet(weight=weight)
# Move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# Value of learning rate determines how model converges to a small error.
# Specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()#define categorical cross-entropy loss function
# Specify Optimiser
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)# Define the optimizer with learning rate

# number of epochs to train the model
n_epochs = 30

valid_loss_min = np.Inf # track changes in validation loss

# for epoch in range(1, n_epochs + 1):

#     # keep track of training and validation loss
#     train_loss = 0.0
#     valid_loss = 0.0

#     ###################
#     # train the model #
#     ###################

#     model.train()
#     for data, target in train_loader:
#         # move tensors to GPU i CUDA is available
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()

#         # clear the gradients of all optimised variables
#         optimizer.zero_grad()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model(data)
#         #print("model output", output.shape)
#         # calculate the batch loss
#         loss = criterion(output, target)
#         # backward pass: compute gradient of the loss w.r.t. model parameters
#         loss.backward()
#         # perform a single optimisation step (parameter update)
#         optimizer.step()
#         # update training loss
#         train_loss += loss.item()*data.size(0)

#     ######################
#     # validate the model #
#     ######################
#     model.eval()
#     for data, target in valid_loader:
#         # move tensors to gpu If CUDA is available
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()

#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model(data)
#         # calculate the batch loss
#         loss = criterion(output, target)
#         # update average validation loss
#         valid_loss += loss.item()*data.size(0)

#     # calculate average losses
#     train_loss = train_loss/len(train_loader.sampler)
#     valid_loss = valid_loss/len(valid_loader.sampler) 

#     # print training/validation statistics
#     print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss)) 

#     # save model if validation loss has decreased
#     if valid_loss <= valid_loss_min:
#         print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,
#                                                                                        valid_loss))
#         torch.save(model.state_dict(), 'model_cat_dog_classifier.pt')
#         valid_loss_min = valid_loss

# Load the model with the lowest validation loss
model.load_state_dict(torch.load('model_cat_dog_classifier.pt'))

    # Test the Trained Network
    # Testing the model on previously unseen data, a good result will be a CNN that gets around 70% or more 
    # on these test images.

# Track Test Loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)'%(classes[i], 100*class_correct[i]/class_total[i],
                                                       np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)'% (classes[i]))


print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' %(
    100*np.sum(class_correct)/np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))