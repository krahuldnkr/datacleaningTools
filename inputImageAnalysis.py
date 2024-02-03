import matplotlib.pyplot as plt
import numpy as np

class inputImageAnalysis:

    def analyser(self, images):
        """_summary_

        Args:
            images (_type_): _description_
        """
        img = np.squeeze(images)
        fig = plt.figure(figsize = (12, 12))

        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        width, height = img.shape
        thresh = img.max()/2.5

        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(str(val), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment ='center',
                            color='white' if img[x][y]<thresh else 'black')
        plt.show()

    def multipleImagePlotter(self, images, labels):
        """_summary_

        Args:
            images (_type_): _description_
        """
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            # returns an axis object.
            # first two integers define the division of the figure image,
            # and the last one actually shows where in that division the subplot should go.
            # so here we will get 2*10 figure, and the subplot is placed at 'idx+1' in the formed matrix.
            ax = fig.add_subplot(2, int(20/2), idx+1, xticks = [], yticks = [])

            # squeeze is used to remove single dim. entries from the shape of an array.
            ax.imshow(np.squeeze(images[idx]), cmap='gray')

            # .item() gets the value contained in a Tensor
            ax.set_title(str(labels[idx].item()))
    
        plt.show()
