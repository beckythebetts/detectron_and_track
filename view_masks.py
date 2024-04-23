import SETTINGS
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

import utils
import SETTINGS

def min_max_scale(image):
    min = np.min(image)
    max = np.max(image)
    return (image - min) / (max - min)

def save_mask2(am_mask_tiff, ye_mask_tiff, im_tiff, save_path):
    am_mask, ye_mask, im = plt.imread(am_mask_tiff), plt.imread(ye_mask_tiff), plt.imread(im_tiff)
    plt.axis('off')
    im = min_max_scale(im).T
    im_RGB = np.stack((im, im, im), axis=0)
    split_am_masks = [np.where(am_mask == i + 1, 1, 0) for i in range(0, np.max(am_mask)) if i in am_mask]
    split_ye_masks = [np.where(ye_mask == i + 1, 1, 0) for i in range(0, np.max(ye_mask)) if i in ye_mask]
    for single_mask in split_am_masks:
        outline=cv2.morphologyEx(single_mask.T.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((6, 6), np.uint8))
        im_RGB[0] = np.where(outline, 1, im_RGB[0])
        im_RGB[1] = np.where(outline, 0, im_RGB[1])
        im_RGB[2] = np.where(outline, 1, im_RGB[2])
    for single_mask in split_ye_masks:
        outline=cv2.morphologyEx(single_mask.T.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((6, 6), np.uint8))
        im_RGB[0] = np.where(outline, 0, im_RGB[0])
        im_RGB[1] = np.where(outline, 0, im_RGB[1])
        im_RGB[2] = np.where(outline, 1, im_RGB[2])
    plt.imshow(im_RGB.T)
    plt.show()
    Image.fromarray((im_RGB*255).T.astype(np.uint8)).save(save_path)

def main():
    directory = SETTINGS.DIRECTORY
    labelled_directory = (directory / 'inference_dataset' / 'labelled_images')
    image_directory = (directory / 'inference_dataset' / 'images')
    masks_1 = (directory / 'inference_dataset' / 'masks' / 'Amoeba')
    masks_2 = (directory / 'inference_dataset' / 'masks' / 'Yeast')
    utils.remake_dir(labelled_directory)

    for image in image_directory.glob('*'):
        save_mask2(masks_1 / )