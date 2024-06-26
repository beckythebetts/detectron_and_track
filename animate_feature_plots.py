import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from pathlib import Path
import torch
import numpy as np

import SETTINGS
import utils
import mask_funcs


def init(lines):
    for line in lines:
        line.set_data([], [])
    return lines


def update(frame, lines, data, axs):
    for i, line in enumerate(lines):
        line.set_data(data.index[:frame], data.iloc[:frame, i])
        axs[i].relim()  # Recalculate limits
        axs[i].autoscale_view()
    return lines



def animate_plot_features(cell_number):
    utils.remake_dir(SETTINGS.DIRECTORY / 'features_animated')
    plt.rcParams["font.family"] = 'serif'
    features_path = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(cell_number)+'.txt')
    data = pd.read_csv(features_path, sep='\t')
    fig, axs = plt.subplots(5, sharex=True, figsize=(10, 10))

    lines = []
    ylabels = ['Area', 'Speed', 'Perimeter' , 'Distance to nearest yeast', 'Yeast pixels']
    for i in range(5):
        line = axs[i].plot([], [], color='k')[0]
        lines.append(line)
        #print(data.columns.values.tolist())
        axs[i].set(xlim=[0, 1200], ylabel=ylabels[i])
        axs[i].grid()

    fig.suptitle('Amoeba ' + features_path.stem)
    axs[-1].set(xlabel='frames')



    ani = FuncAnimation(fig, update, fargs=(lines, data, axs), frames=range(1, 1000), init_func=lambda: init(lines), blit=True)

    # Save the animation as a GIF
    animation_path = SETTINGS.DIRECTORY / 'features_animated' / (
            features_path.stem + '.gif')  # Replace with your save directory path
    ani.save(animation_path, writer='pillow', fps=100)

def my_animate_plot(cell_number):
    utils.remake_dir(SETTINGS.DIRECTORY / 'features_animated')
    plt.rcParams["font.family"] = 'serif'
    features_path = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(cell_number) + '.txt')
    data = pd.read_csv(features_path, sep='\t')
    for frame_num in range(1200):
        fig, axs = plt.subplots(5, sharex=True, figsize=(10, 10))
        ylabels = ['Area', 'Speed', 'Perimeter', 'Distance to nearest yeast', 'Yeast pixels']
        for i in range(5):
            axs[i].plot(data.index[:frame_num], data.iloc[:frame_num, i], color='k')
            axs[i].relim()  # Recalculate limits
            axs[i].autoscale_view()
            axs[i].set(xlim=[0, 1200], ylabel=ylabels[i])
            axs[i].grid()

        fig.suptitle('Amoeba ' + features_path.stem)
        axs[-1].set(xlabel='frames')
        plt.savefig(SETTINGS.DIRECTORY / 'features_animated' / ('{0:04}'.format(frame_num)+'.jpg'))

def show_cell(cell_index):
    utils.remake_dir(SETTINGS.DIRECTORY / 'show_cell')
    for frame in (SETTINGS.DIRECTORY  / 'inference_dataset'/ 'phase').iterdir():
        image = torch.tensor(utils.read_tiff(frame)).cuda()
        epi_image = torch.tensor(utils.read_tiff(
            SETTINGS.DIRECTORY / 'inference_dataset' / 'epi' / (frame.stem + '.tif')).astype(
            np.float32)).cuda()
        mask = torch.tensor(utils.read_tiff(
            SETTINGS.DIRECTORY / 'tracked' / 'phase' / (frame.stem + '.tif')).astype(
            np.int16)).cuda()
        outline = mask_funcs.mask_outline(torch.where(mask == int(cell_index), 1, 0), thickness=3)
        epi_image_normalised = (epi_image - epi_image.min()) / (epi_image.max() - epi_image.min()) * 255
        im_rgb = torch.stack((image, image, image), axis=0)

        im_rgb[0] = torch.where(outline, 255, im_rgb[0])
        im_rgb[1] = torch.where(outline, 255, im_rgb[1])
        im_rgb[2] = torch.where(outline, 0, im_rgb[2])

        im_rgb[0] = torch.where(epi_image > SETTINGS.THRESHOLD, epi_image_normalised, im_rgb[0])
        im_rgb[1] = torch.where(epi_image > SETTINGS.THRESHOLD, 0, im_rgb[1])
        im_rgb[2] = torch.where(epi_image > SETTINGS.THRESHOLD, 0, im_rgb[2])

        im_rgb = im_rgb.permute(1, 2, 0)

        utils.save_tiff((im_rgb).cpu().numpy().astype(np.uint8),
                        SETTINGS.DIRECTORY / 'show_cell' / ("{0:04}".format(int(frame.stem)) + '.jpg'))

import os
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip

def read_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def resize_images(img1, img2, height=None):
    if height is None:
        height = min(img1.shape[0], img2.shape[0])
    ratio1 = height / img1.shape[0]
    ratio2 = height / img2.shape[0]
    img1_resized = cv2.resize(img1, (int(img1.shape[1] * ratio1), height))
    img2_resized = cv2.resize(img2, (int(img2.shape[1] * ratio2), height))
    return img1_resized, img2_resized

def combine_images(img1, img2):
    return np.hstack((img1, img2))

def create_combined_frames(images1, images2):
    combined_frames = []
    for img1, img2 in zip(images1, images2):
        img1_resized, img2_resized = resize_images(img1, img2)
        combined_frame = combine_images(img1_resized, img2_resized)
        combined_frames.append(combined_frame)
    return combined_frames

def create_video_from_frames(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], fps=fps)
    clip.write_videofile(output_path, codec='libx264')

def main():
    #my_animate_plot(7)
    #show_cell(7)

    # Paths to the folders containing images
    folder1 = '04/features_animated'
    folder2 = '04/show_cell'

    # Read images from folders
    images1 = read_images_from_folder(folder1)
    images2 = read_images_from_folder(folder2)

    # Ensure both sets have the same number of images
    assert len(images1) == len(images2), "The two sets of images must have the same number of images."

    # Create combined frames
    combined_frames = create_combined_frames(images1, images2)

    # Path to the output video file
    output_video_path = '04/output_video.mp4'

    # Create video from combined frames
    create_video_from_frames(combined_frames, output_video_path)

if __name__ == '__main__':
    main()

