import SETTINGS
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import sys

from multiprocessing import Pool

import utils
import SETTINGS

class Masks:
    def __init__(self, dir):
        self.dir = dir / 'inference_dataset'
        self.categories = [entry.name for entry in (self.dir / 'masks').iterdir()]
        self.length = len([name for name in os.listdir(str(self.dir / 'images'))])



    def save_masks(self, max_images=10):
        utils.remake_dir(self.dir / 'labelled_images')
        colours = np.random.uniform(0.5, 1.0, size=(len(self.categories), 3))
        image_paths = list(sorted((self.dir / 'images').iterdir()))[:max_images]

        with Pool() as pool:
            results = pool.starmap(process_image,
                                   [(image_path, self.dir, self.categories, colours) for image_path in image_paths])

        for result in results:
            image_path, im_RGB = result
            Image.fromarray((im_RGB * 255).T.astype(np.uint8)).save(
                self.dir / 'labelled_images' / str(image_path.stem + '_labelled.png'))


def min_max_scale(image):
    min = np.min(image)
    max = np.max(image)
    return (image - min) / (max - min)

def process_image(image_path, dir, categories, colours):
    image = plt.imread(image_path)
    image = min_max_scale(image).T
    im_RGB = np.stack((image, image, image), axis=0)
    for i, category in enumerate(categories):
        mask = plt.imread(dir / 'masks' / category / str(image_path.stem + '_mask.tif'))
        split_mask = [np.where(mask == i + 1, 1, 0) for i in range(0, np.max(mask)) if i + 1 in mask]
        for single_mask in split_mask:
            outline = cv2.morphologyEx(single_mask.T.astype(np.uint8), cv2.MORPH_GRADIENT,
                                       np.ones((6, 6), np.uint8))
            im_RGB[0] = np.where(outline, colours[i, 0], im_RGB[0])
            im_RGB[1] = np.where(outline, colours[i, 1], im_RGB[1])
            im_RGB[2] = np.where(outline, colours[i, 2], im_RGB[2])
    return (image_path, im_RGB)

def main():
    directory = SETTINGS.DIRECTORY
    masks = Masks(directory)
    masks.save_masks(SETTINGS.MAX_IMAGES)

if __name__ == '__main__':
    main()