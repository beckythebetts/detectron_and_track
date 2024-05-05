import SETTINGS
import utils
import matplotlib.pyplot as plt


def main():
    images_dir = SETTINGS.DIRECTORY / 'inference_dataset' / 'epi'
    segmented_dir = SETTINGS.DIRECTORY / 'segmented' / 'epi'
    #for image in images_dir.iterdir():
    image = utils.read_tiff([image for image in images_dir.iterdir()][0])
    from skimage.filters import try_all_threshold
    fig, ax = try_all_threshold(image, figsize=(10, 8), verbose=False)
    plt.show()

if __name__ == '__main__':
    main()