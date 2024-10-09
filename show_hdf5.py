import h5py
import imagej
import numpy as np
import torch
import sys
import time


import mask_funcs
import SETTINGS
#from jnius import autoclass

#hdf5_file = 'Datasets/04_short.h5'
#hdf5_file = 'Datasets/danhighres/dan10.h5'
hdf5_file = SETTINGS.DATASET
#ij = imagej.init('sc.fiji:fiji:2.1.0', mode='gui')
ij = imagej.init('2.1.0', mode='interactive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_rgb(greyscale_im, axis=1):
    return np.stack((greyscale_im, greyscale_im, greyscale_im), axis=axis)[:, np.newaxis]

def show_separate_channels():
    with h5py.File(hdf5_file, 'r') as f:
        print('Gathering data')
        data = np.array([
            [
                f['Images'][channel][frame][:]
                for channel in f['Images'].keys()
            ]
            for frame in f['Images']['Phase'].keys()
        ], dtype='uint8')
        print('converting data')
        ij_image = ij.py.to_java(data)
        print('opening images')
        #ij.io().open(ij_image)
        ij.ui().show('images', ij_image)
    time.sleep(99999)

def show_merged_channels():
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = np.array([f['Images']['Phase'][frame][:]
                               for frame in list(f['Images']['Phase'].keys())[:20]], dtype='uint8')
        epi_data = np.array([f['Segmentations']['Epi'][frame][:]
                               for frame in list(f['Images']['Epi'].keys())[:20]], dtype='uint8')

    epi_data[epi_data > 0] = 255
    epi_channel = make_rgb(epi_data)
    epi_channel[:,:,1:3] = 0
    merged_data = ((make_rgb(phase_data).astype(np.float32) + epi_channel.astype(np.float32)) / (2)).astype(np.uint8)
    merged_image = ij.py.to_dataset(merged_data, dim_order=['time', 'z', 'ch', 'row', 'col'])
    ij.ui().show(merged_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)

def show_tracked_images():
    print('\nPREPARING TRACKED IMAGES\n')
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = np.array([f['Images']['Phase'][frame][:]
                               for frame in f['Images']['Phase'].keys()][:3], dtype='uint8')
        segmentation_data = np.array([f['Segmentations']['Phase'][frame][:]
                                      for frame in list(f['Segmentations']['Phase'].keys())][:3], dtype='int16')
    max_cell_index=np.max(segmentation_data)
    colour_dict = {cell_index: torch.tensor(np.random.uniform(0, (2 ** 8) - 1, size=3).astype('uint8')).to(device) for
                   cell_index in np.arange(1, max_cell_index + 1)}
    rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
    tracked = np.zeros(rgb_phase.shape)
    for i, (phase_image, segmentation) in enumerate(
            zip(torch.tensor(rgb_phase).to(device), torch.tensor(segmentation_data).to(device))):
        sys.stdout.write(
            f'\rFrame {i + 1}')
        sys.stdout.flush()
        for cell_index in torch.unique(segmentation)[1:]:
            outline = mask_funcs.mask_outline(torch.where(segmentation == cell_index.item(), 1, 0), thickness=3)
            phase_image[outline] = colour_dict[cell_index.item()]

        tracked[i] = phase_image.cpu().numpy()
    tracked_image = ij.py.to_dataset(tracked, dim_order=['time', 'row', 'col', 'ch'])
    ij.ui().show(tracked_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)

def show_tracked_images_fast():
    print('\nPREPARING TRACKED IMAGES\n')
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = np.array([f['Images']['Phase'][frame][:]
                               for frame in f['Images']['Phase'].keys()][:3], dtype='uint8')
        segmentation_data = np.array([f['Segmentations']['Phase'][frame][:]
                                      for frame in list(f['Segmentations']['Phase'].keys())][:3], dtype='int16')
    max_cell_index = np.max(segmentation_data)
    colour_dict = {cell_index: torch.tensor(np.random.uniform(0, (2 ** 8) - 1, size=3).astype('uint8')).to(device)
                   for
                   cell_index in np.arange(1, max_cell_index + 1)}
    rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
    tracked = np.zeros(rgb_phase.shape)
    for i, (phase_image, segmentation) in enumerate(
            zip(torch.tensor(rgb_phase).to(device), torch.tensor(segmentation_data).to(device))):
        sys.stdout.write(
            f'\rFrame {i + 1}')
        sys.stdout.flush()
        #expanded_segmentation = [segmentation[segmentation==idx] for idx in torch.unique(segmentation)]
        expanded_segmentation = (segmentation.unsqueeze(0) == torch.unique(segmentation).unsqueeze(1).unsqueeze(2))
        print(expanded_segmentation.shape)
    #     for cell_index in torch.unique(segmentation)[1:]:
    #         outline = mask_funcs.mask_outline(torch.where(segmentation == cell_index.item(), 1, 0), thickness=3)
    #         phase_image[outline] = colour_dict[cell_index.item()]
    #
    #     tracked[i] = phase_image.cpu().numpy()
    # tracked_image = ij.py.to_dataset(tracked, dim_order=['time', 'row', 'col', 'ch'])
    # ij.ui().show(tracked_image)
    # ij.py.run_macro(macro='run("Make Composite")')
    # time.sleep(99999)

def main():
    #show_separate_channels()
    #show_merged_channels()
    show_tracked_images_fast()
if __name__ == '__main__':
    main()