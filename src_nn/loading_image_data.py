"""Tutorial on loading image data for training and testing.
"""

import matplotlib.pyplot as plt
import torch

from torchvision import datasets, transforms
from PIL import Image



# Read an image
img = Image.open('Cat_Dog_data/Cat_Dog_data/test/cat/cat.16.jpg')

# Pre-processing of the input image
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
 
dataset   = datasets.ImageFolder('Cat_Dog_data/', transform = transform)

#imag = transform(img)
#imag.show()

dataloader = torch.utils.data.DataLoader(dataset, batch_size= 32, shuffle = True)

#for images, labels in dataloader:
#    pass

# Run this to test your data loader
#images, labels = next(iter(dataloader))
#helper.imshow(images[0], normalize=False)

# Step 1:Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

# Step 2: Loading dataset and applying transforms

train_data = datasets.ImageFolder("Cat_Dog_data/Cat_Dog_data/train", transform = train_transforms)
test_data  = datasets.ImageFolder("Cat_Dog_data/Cat_Dog_data/test",  transform= test_transforms)

# Step 3: Setting up the batch size and applying shuffle to the input data

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
testloader  = torch.utils.data.DataLoader(test_data, batch_size = 32)

print(trainloader.dataset)

images, labels = next(iter(trainloader))

print(images.size())