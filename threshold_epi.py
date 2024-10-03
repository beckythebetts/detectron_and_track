import SETTINGS
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import cv2
from scipy.ndimage import label
import sys
import numpy as np
import h5py

import bilateral_filter

def test_filter_and_threshold(test_threshold_value, iterations, d, sigmaColour, sigmaSpace):
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        test_image = f['Images']['Epi'][list(f['Images']['Epi'].keys())[0]][...]
        filtered_image = bilateral_filter.apply_bilateral_filter(test_image, iterations, d, sigmaColour, sigmaSpace)
        mask = np.where(filtered_image > test_threshold_value, 1, 0)
        thresholded_image = np.stack((filtered_image, filtered_image, filtered_image), axis=-1)
        print(test_image.shape)
        thresholded_image[:,:,1] = np.where(mask, 1, thresholded_image[:,:,1])

        fig = plt.figure()
        grid = ImageGrid(fig, (0,0,1,1), nrows_ncols=(1, 3))
        for ax, im in zip(grid, (test_image, filtered_image, thresholded_image)):
            ax.imshow(im)
            ax.axis('off')
        # plt.imshow(test_image)
        plt.show()

def apply_threshold(threshold=SETTINGS.THRESHOLD):
    print('\n--------------------\nTHRESHOLDING - ', SETTINGS.CLASSES['epi'], '\n--------------------')
    with h5py.File(SETTINGS.DATASET, 'r+') as f:

        num_frames = f['Images'].attrs['Number of frames']
        for i, frame in enumerate(f['Images']['Epi']):
            sys.stdout.write(
                f'\rAdding frame {i + 1} / {num_frames}')
            sys.stdout.flush()
            image = f['Images']['Epi'][frame][...]
            image = np.where(image > threshold, 1, 0)
            image, num_objects = label(image)
            f.create_dataset(f'Segmentations/Epi/{frame}', dtype='i2', data=image)
            f['Segmentations']['Epi'].attrs['Threshold'] = threshold

def main():
    test_filter_and_threshold(250, 20, -1, 5, 5)
    #apply_threshold()


if __name__ == '__main__':
    main()