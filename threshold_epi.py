import SETTINGS
import utils
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.ndimage import label
import sys
import numpy as np
import h5py


def main():
    print('--------------------\nTHRESHOLDING - ', SETTINGS.CLASSES['epi'], '\n--------------------')
    with h5py.File(SETTINGS.DATASET, 'r+') as f:
        if 'Epi' in f['Segmentations']:
            del(f['Segmentations']['Epi'])
        num_frames = f['Images'].attrs['Number of frames']
        for i, frame in enumerate(f['Images']['Epi']):
            sys.stdout.write(
                f'\rAdding frame {i + 1} / {num_frames}')
            sys.stdout.flush()
            image = f['Images']['Epi'][frame][...]
            image = np.where(image > SETTINGS.THRESHOLD, 1, 0)
            image, num_objects = label(image)
            f.create_dataset(f'Segmentations/Epi/{frame}', dtype='i2', data=image)


if __name__ == '__main__':
    main()