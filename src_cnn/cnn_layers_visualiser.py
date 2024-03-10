import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from src_cnn.utils.playing_with_convolutional_layer import Net

class viz_layer():
    # helper function for visualising the output of a given layer
    # default number of filters is 4
    def viz_layer_(self,layer, n_filters = 4):
        fig = plt.figure(figsize=(20, 20))

        for i in range(n_filters):
            ax = fig.add_subplot(1, n_filters, i+1, xticks = [], yticks=[])
            # grab layer outputs
            ax.imshow(  np.squeeze(layer[0,i].data.numpy()), cmap='gray')
            ax.set_title('Output %s' % str(i+1))
        plt.show()

        
image_    = mpimg.imread("../Cat_Dog_data/Cat_Dog_data/test/cat/cat.12143.jpg")

gray_img  = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)
print("converted gray image shape: ", gray_img.shape)

#plt.imshow(gray_, cmap='gray')
#plt.show()

# convert the image into an input tensor
gray_img = gray_img.astype("float32")/255
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get the convolutional layer (pre and post activation)
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
conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)

# visualise the output of a conv layer
viz_layer__ = viz_layer()
viz_layer__.viz_layer_(conv_layer)

# after a ReLu is applied
# visualize the output of an activated conv layer
viz_layer__.viz_layer_(activated_layer)

# visualize the output of the pooling layer
viz_layer__.viz_layer_(pooled_layer)