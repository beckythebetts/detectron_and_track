import SETTINGS
import utils
import matplotlib.pyplot as plt
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
        plt.imshow(test_image)
        plt.show()
        test_image = bilateral_filter(test_image, iteratiions, d, sigmaColour, sigmaSpace)
        plt.imshow(test_image)
        plt.show()
        mask = np.where(test_image > test_threshold_value, 1, 0)
        test_image = np.stack((test_image, test_image, test_image), axis=-1)
        print(test_image.shape)
        test_image[:,:,1] = np.where(mask, 1, test_image[:,:,1])
        plt.imshow(test_image)
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
    #test_threshold(250)
    apply_threshold()


if __name__ == '__main__':
    main()