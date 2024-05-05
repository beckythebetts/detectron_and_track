import SETTINGS
import utils
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.ndimage import label
import sys
import numpy as np


def main():
    print('--------------------\nTHRESHOLDING - ', SETTINGS.CLASSES['epi'], '\n--------------------')
    images_dir = SETTINGS.DIRECTORY / 'inference_dataset' / 'epi'
    segmented_dir = SETTINGS.DIRECTORY / 'segmented' / 'epi'
    utils.remake_dir(segmented_dir)
    num_frames = len([im for im in images_dir.iterdir()])
    for i, image in enumerate(images_dir.iterdir()):
        sys.stdout.write(
            f'\rAdding frame {i + 1} / {num_frames}')
        sys.stdout.flush()
        image = utils.read_tiff(image)
        image = np.where(image > SETTINGS.THRESHOLD, 1, 0)
        image, num_objects = label(image)
        utils.save_tiff(image.astype(np.uint16), segmented_dir/ ("{0:04}".format(i) + '.tif'))

if __name__ == '__main__':
    main()