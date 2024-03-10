import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

# Read in the image
image = mpimg.imread('../Cat_Dog_data/Cat_Dog_data/test/cat/cat.16.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print("shape of the gray image", gray.shape)
# TODO:: Create a custom kernel
# Sobel filter is commonly used in EDGE DETECTION and in finding 
# patterns in intensity in an image.

# Create a custom kernel.
# 3 x 3 array for horizontal edge detection
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])
filtered_image_y = cv2.filter2D(gray, -1, sobel_y) # inputs: grayscale image, bit-depth, kernel

# 3 x 3 array for vertical edge detection
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
filtered_image_x = cv2.filter2D(gray, -1, sobel_x)

# 3 x 3 array for blur
sobel_blur = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
filtered_image_blur = cv2.filter2D(image, -1, sobel_blur)


plt.imshow(gray, cmap = 'gray')
plt.show()
# plt.imshow(filtered_image_y, cmap = 'gray')
# plt.show()
# plt.imshow(filtered_image_x, cmap = 'gray')
# plt.show()
# plt.imshow(sobel_blur, cmap = 'gray')
# plt.show()

# normalize, rescale entries to lie in [0, 1]
gray_img = gray.astype("float32")/255

# plot image
# plt.imshow(gray_img, cmap='gray')
# plt.show()

## TODO: Feel free to modify the numbers here, to try out another filter!
filter_vals = np.array([[-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1]])

print('Filter shape: ', filter_vals.shape)

# Defining four different filters,
# all of which are linear combinations of the 'filter_vals' defined above.

# defining four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3

filters = np.array([filter_1, filter_2, filter_3, filter_4])

# For an example, print out the values of filter 1
print('Filter 1: \n', (filter_1))

# Visualize all four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    row, cols = filters[i].shape

    for r in range(row):
        for c in range(cols):
            ax.annotate(str(filters[i][r][c]), xy=(c, r),
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        color = 'white' if filters[i][r][c]<0 else 'black')
            
plt.show()
