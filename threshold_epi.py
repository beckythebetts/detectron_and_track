import SETTINGS
import utils
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.ndimage import label
import sys
import numpy as np
import h5py

def test_threshold(test_threshold_value):
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        test_image = f['Images']['Epi'][list(f['Images']['Epi'].keys())[0]][...]
        mask = np.where(test_image > test_threshold_value, 1, 0)
        test_image = np.stack((test_image, test_image, test_image), axis=-1)
        print(test_image.shape)
        test_image[:,:,1] = np.where(mask, 1, test_image[:,:,1])
        plt.imshow(test_image)
        plt.show()

def apply_threshold(threshold=SETTINGS.THRESHOLD):
    print('\n--------------------\nTHRESHOLDING - ', SETTINGS.CLASSES['epi'], '\n--------------------')
    with h5py.File(SETTINGS.DATASET, 'r+') as f:
        if 'Epi' in f['Segmentations']:
            del (f['Segmentations']['Epi'])
        num_frames = f['Images'].attrs['Number of frames']
        for i, frame in enumerate(f['Images']['Epi']):
            sys.stdout.write(
                f'\rAdding frame {i + 1} / {num_frames}')
            sys.stdout.flush()
            image = f['Images']['Epi'][frame][...]
            image = np.where(image > threshold, 1, 0)
            image, num_objects = label(image)
            f.create_dataset(f'Segmentations/Epi/{frame}', dtype='i2', data=image)

def main():
    test_threshold(255)
    #apply_threshold()


if __name__ == '__main__':
    main()