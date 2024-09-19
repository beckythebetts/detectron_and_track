import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path

def apply_bilateral_filter(image_array, iterations, d, sigmaColor, sigmaSpace):
    for _ in range(iterations):
        image_array = cv2.bilateralFilter(image_array, d, sigmaColor, sigmaSpace)
    return image_array
def test_filter():
    with h5py.File(r'Datasets\04_short_testing.h5', 'r') as f:
        # test_image = f['Images']['Phase']['0000'][:200, :200]
        test_image = plt.imread(Path('RAW_DATA') / '04' / 'training_dataset' / 'kfold_50_2_2' / 'images' / '00.jpg')
        test_image = test_image[:100, :100]
        filtered_images = [apply_bilateral_filter(test_image, i, -1, 5, 5) for i in range(0, 100, 10)]
        fig = plt.figure(figsize =(10,4))
        fig.tight_layout()
        grid = ImageGrid(fig, (0, 0, 1, 1), nrows_ncols=(2, 5))
        for ax, im in zip(grid, filtered_images):
            ax.imshow(im, cmap='gray')
            ax.axis('off')

        plt.show()
        # test_image = f['Images']['Phase']['0000'][:]
        # test_image = plt.imread(r'RAW_DATA\test_filter\Training_Data\train\Images\snap00.png')
        #
        # plt.imsave('bifilter.png', apply_bilateral_filter(test_image, 20, -1, 2, 2), cmap='gray')

def filter_ims(directory):
    for im in directory.iterdir():
        im_array = plt.imread(str(im))
        plt.imsave(str(im.parent / (im.stem + '.png')), apply_bilateral_filter(im_array, 50, -1, 5, 5))

def main():
    filter_ims(Path('RAW_DATA') / '04' / 'training_dataset' / 'kfold_50_5_5_png' / 'images')
    #test_filter()

if __name__ == '__main__':
    main()