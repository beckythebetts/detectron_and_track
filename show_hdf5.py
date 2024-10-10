import h5py
import imagej
import numpy as np
import torch
import sys
import time
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
import cv2
import matplotlib.pyplot as plt


import mask_funcs
import SETTINGS
#from jnius import autoclass

#hdf5_file = 'Datasets/04_short.h5'
#hdf5_file = 'Datasets/danhighres/dan10.h5'
hdf5_file = SETTINGS.DATASET
#ij = imagej.init('sc.fiji:fiji:2.1.0', mode='gui')
ij = imagej.init('2.1.0', mode='interactive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def make_rgb(greyscale_im, axis=1):
#     return np.stack((greyscale_im, greyscale_im, greyscale_im), axis=axis)[:, np.newaxis]

# def show_separate_channels(first_frame=0, last_frame=50):
#     print('\nSHOWING SEPARATE CHANNELS')
#     with h5py.File(hdf5_file, 'r') as f:
#         data = np.array([
#             [
#                 f['Images'][channel][frame][:]
#                 for channel in f['Images'].keys()
#             ]
#             for frame in f['Images']['Phase'].keys()
#         ], dtype='uint8')
#         ij_image = ij.py.to_java(data)
#         ij.ui().show('images', ij_image)
#     time.sleep(99999)

# def show_merged_channels_old(first_frame=0, last_frame=50):
#     print('\nSHOWING MERGED IMAGES')
#     with h5py.File(hdf5_file, 'r') as f:
#         phase_data = np.array([f['Images']['Phase'][frame][:]
#                                for frame in list(f['Images']['Phase'].keys())[first_frame:last_frame]], dtype='uint8')
#         epi_data = np.array([f['Segmentations']['Epi'][frame][:]
#                                for frame in list(f['Images']['Epi'].keys())[first_frame:last_frame]], dtype='uint8')
#
#     epi_data[epi_data > 0] = 255
#     epi_channel = make_rgb(epi_data)
#     epi_channel[:,:,1:3] = 0
#     merged_data = (0.75*(make_rgb(phase_data).astype(np.float32) + 0.25*epi_channel.astype(np.float32))).astype(np.uint8)
#     # only merge red channel (0)
#     # merged_im = make_rgb(phase_data)
#     # merged_im[:, :, 0] = (phase_data).astype(np.float32)+epi_channel[:, :, 0]
#     merged_image = ij.py.to_dataset(merged_data, dim_order=['time', 'z', 'ch', 'row', 'col'])
#     ij.ui().show(merged_image)
#     ij.py.run_macro(macro='run("Make Composite")')
#     time.sleep(99999)

def show_merged_channels(first_frame=0, last_frame=50):
    print('\nSHOWING MERGED IMAGES')
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = np.array([f['Images']['Phase'][frame][:]
                               for frame in list(f['Images']['Phase'].keys())[first_frame:last_frame]], dtype='uint8')
        epi_data = np.array([f['Images']['Epi'][frame][:]
                               for frame in list(f['Images']['Epi'].keys())[first_frame:last_frame]], dtype='uint8')
    merged_im = np.stack((phase_data, phase_data, phase_data), axis=1)
    merged_im[:, 0][epi_data > SETTINGS.THRESHOLD] = epi_data[epi_data > SETTINGS.THRESHOLD]
    merged_image = ij.py.to_dataset(merged_im, dim_order=['t', 'ch', 'row', 'col'])
    ij.ui().show(merged_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)
# def show_tracked_images_old():
#     print('\nPREPARING TRACKED IMAGES\n')
#     with h5py.File(hdf5_file, 'r') as f:
#         phase_data = np.array([f['Images']['Phase'][frame][:]
#                                for frame in f['Images']['Phase'].keys()][:3], dtype='uint8')
#         segmentation_data = np.array([f['Segmentations']['Phase'][frame][:]
#                                       for frame in list(f['Segmentations']['Phase'].keys())][:3], dtype='int16')
#     max_cell_index=np.max(segmentation_data)
#     colour_dict = {cell_index: torch.tensor(np.random.uniform(0, (2 ** 8) - 1, size=3).astype('uint8')).to(device) for
#                    cell_index in np.arange(1, max_cell_index + 1)}
#     rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
#     tracked = np.zeros(rgb_phase.shape)
#     for i, (phase_image, segmentation) in enumerate(
#             zip(torch.tensor(rgb_phase).to(device), torch.tensor(segmentation_data).to(device))):
#         sys.stdout.write(
#             f'\rFrame {i + 1}')
#         sys.stdout.flush()
#         for cell_index in torch.unique(segmentation)[1:]:
#             outline = mask_funcs.mask_outline(torch.where(segmentation == cell_index.item(), 1, 0), thickness=3)
#             phase_image[outline] = colour_dict[cell_index.item()]
#
#         tracked[i] = phase_image.cpu().numpy()
#     tracked_image = ij.py.to_dataset(tracked, dim_order=['time', 'row', 'col', 'ch'])
#     ij.ui().show(tracked_image)
#     ij.py.run_macro(macro='run("Make Composite")')
#     time.sleep(99999)

def show_tracked_images(first_frame=0, last_frame=50):
    print('\nPREPARING TRACKED IMAGES\n')
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = np.array([f['Images']['Phase'][frame][:]
                               for frame in f['Images']['Phase'].keys()][first_frame:last_frame], dtype='uint8')
        segmentation_data = np.array([f['Segmentations']['Phase'][frame][:]
                                      for frame in list(f['Segmentations']['Phase'].keys())][first_frame:last_frame], dtype='int16')
    max_cell_index = np.max(segmentation_data)
    LUT = torch.randint(low=10, high=255, size=(max_cell_index+1, 3)).to(device)
    LUT[0] = torch.tensor([0, 0, 0]).to(device)
    rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
    tracked = np.zeros(rgb_phase.shape)
    for i, (phase_image, segmentation) in enumerate(
            zip(rgb_phase, segmentation_data)):
        segmentation = torch.tensor(segmentation).to(device)
        phase_image = torch.tensor(phase_image).to(device).int()
        sys.stdout.write(
            f'\rFrame {i + 1}')
        sys.stdout.flush()
        outlines = mask_funcs.mask_outlines(segmentation).int()
        outlines = LUT[outlines]
        phase_image =  (torch.where(outlines>0, outlines, phase_image))
        phase_image = phase_image.cpu().numpy().astype(np.uint8)
        tracked[i] = phase_image
    tracked_image = ij.py.to_dataset(tracked, dim_order=['t', 'row', 'col', 'ch'])
    ij.ui().show(tracked_image)
    ij.py.run_macro(macro='run("Make Composite")')
    #ij.py.run_macro(macro='run("8-bit")')
    time.sleep(99999)

def show_cell(cell_idx, first_frame=0, last_frame=50, frame_size=150):
    print(f'\nSHOWING CELL {cell_idx}')
    with h5py.File(hdf5_file, 'r') as f:
        xcentres = np.array([f['Features'][f'Cell{cell_idx:04}']['MorphologicalFeatures'][int(framei)]['xcentre'] for framei in
                            range(first_frame, last_frame)])
        ycentres = np.array([f['Features'][f'Cell{cell_idx:04}']['MorphologicalFeatures'][int(framei)]['ycentre'] for framei in
                            range(first_frame, last_frame)])
        xmins, xmaxs, ymins, ymaxs = (xcentres-frame_size/2).astype(int), (xcentres+frame_size/2).astype(int), (ycentres-frame_size/2).astype(int), (ycentres+frame_size/2).astype(int)

        phase_data = np.array([f['Images']['Phase'][frame][xmin:xmax,ymin:ymax]
                               for frame, xmin, xmax, ymin, ymax in zip(list(f['Images']['Phase'].keys())[first_frame:last_frame], xmins, xmaxs, ymins, ymaxs)], dtype='uint8')
        epi_data = np.array([f['Images']['Epi'][frame][xmin:xmax,ymin:ymax]
                             for frame, xmin, xmax, ymin, ymax in zip(list(f['Images']['Epi'].keys())[first_frame:last_frame], xmins, xmaxs, ymins, ymaxs)], dtype='uint8')
        mask_data = np.array([f['Segmentations']['Phase'][frame][xmin:xmax,ymin:ymax]
                               for frame, xmin, xmax, ymin, ymax in zip(list(f['Segmentations']['Phase'].keys())[first_frame:last_frame], xmins, xmaxs, ymins, ymaxs)], dtype='uint8')

    cell_mask = (mask_data == cell_idx)
    print(np.unique(cell_mask))
    if not cell_mask.any():
        raise Exception(f'Cell of index {cell_idx} not found')
    cell_outline = mask_funcs.mask_outline(torch.tensor(cell_mask).byte().to(device)).cpu().numpy()
    merged_im = np.stack((phase_data, phase_data, phase_data), axis=1)
    merged_im[:, 0][epi_data > SETTINGS.THRESHOLD] = epi_data[epi_data > SETTINGS.THRESHOLD]
    merged_im[:, 1][cell_outline] = 255
    merged_image = ij.py.to_dataset(merged_im, dim_order=['t', 'ch', 'row', 'col'])
    ij.ui().show(merged_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)
def main():
    #show_separate_channels()
    #show_merged_channels()
    #show_tracked_images()
    show_cell(3)
if __name__ == '__main__':
    main()