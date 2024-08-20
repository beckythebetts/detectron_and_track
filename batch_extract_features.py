import torch
import numpy as np
import sys
import math
import pandas as pd
import gc
import matplotlib.pyplot as plt
import subprocess
import os
import h5py
import imageio
from pathlib import Path

import utils
import mask_funcs
import SETTINGS


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cell:
    def __init__(self, index):
        self.index = index
        self.features_dataset_name = f'Features/Cell{index:04}'

        with h5py.File(SETTINGS.DATASET, 'r+') as f:
            dtype = np.dtype([
                ('area', 'f4'),
                ('speed', 'f4'),
                ('perimeter', 'f4'),
                ('dist_nearest', 'f4'),
                ('eaten', 'f4'),
                ('xcentre', 'f4'),
                ('ycentre', 'f4')
            ])
            f.create_dataset(self.features_dataset_name, shape=(0,), maxshape=(None,), dtype=dtype)

    def write_features(self, features_list):
        with h5py.File(SETTINGS.DATASET, 'a') as f:
            features_dataset = f['Features'][f'Cell{self.index:04}']
            features_dataset.resize(len(features_dataset)+1, axis=0)
            features_dataset[-1] = features_list

class CellBatch:
    def __init__(self, indices):
        self.indices = indices
        self.expanded_indices = self.indices.unsqueeze(-1).unsqueeze(-1).expand((len(indices), *SETTINGS.IMAGE_SIZE))
        self.cells = [Cell(i) for i in self.indices]
        self.centres = None
        self.last_centres = None
        self.batch_size = len(self.indices)
        self.file = h5py.File(SETTINGS.DATASET, 'r+')
        self.frames_list = list(self.file['Segmentations']['Phase'].keys())
        self.frames_list.sort()
        # self.paths = sorted([p for p in (SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir()])
        #self.num_frames = len(self.paths)
        self.coord_grid_x, self.coord_grid_y = torch.meshgrid(torch.arange(SETTINGS.IMAGE_SIZE[0]).to(device),
                                                              torch.arange(SETTINGS.IMAGE_SIZE[1]).to(device))

    def run_feature_extraction(self):

        for i, frame_name in enumerate(self.frames_list):
            sys.stdout.write(f'\rFrame {i+1} | Cells {torch.min(self.indices)}-{torch.max(self.indices)} ')
            sys.stdout.flush()
            if i == 0:
                full_mask = torch.tensor(self.file['Segmentations']['Phase'][frame_name][()].astype(np.int16)).to(device)
                # full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).to(device)
                self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1,0)
                full_mask = None
            self.next_frame(i)
            self.read_features()
            self.epi_mask = None
            self.write_features()
            gc.collect()
        self.file.close()

    def read_frame(self, frame_index, type='Phase'):
        return torch.tensor(self.file['Segmentations'][type][self.frames_list[frame_index]][()].astype(np.int16)).to(device)

    def next_frame(self, frame_index):
        #full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).to(device)
        full_mask = self.read_frame(frame_index)
        self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
        full_mask = None
        self.epi_mask = self.read_frame(frame_index, type='Epi')
    def read_features(self):
        self.get_areas()
        self.get_centres()
        self.get_speeds()
        self.get_perimeters()
        self.get_eaten()
        self.get_nearest()


    def write_features(self):
        for i, cell in enumerate(self.cells):
            new_line = (self.areas[i],
                        self.speeds[i],
                        self.perimeters[i],
                        self.dists[i],
                        self.eaten[i],
                        self.centres[i, 0],
                        self.centres[i, 1])
            #new_line = '\n' + '\t'.join([str(a.item()) for a in (self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i], self.eaten[i], self.centres[i, 0], self.centres[i, 1])])
            cell.write_features(new_line)

    def get_areas(self):
        self.areas = torch.sum(self.masks, dim=(1, 2)).float()
        self.areas[self.areas == 0] = float('nan')

    def get_centres(self):
        if self.centres is not None:
            self.last_centres = self.centres

        x_centres = torch.sum(self.masks * self.coord_grid_x, dim=(1, 2)) / self.areas
        y_centres = torch.sum(self.masks * self.coord_grid_y, dim=(1, 2)) / self.areas

        self.centres = torch.stack((x_centres, y_centres), dim=1)


    def get_speeds(self):
        if self.last_centres is None:
            self.speeds = torch.full((self.batch_size,), float('nan'))

        else:
            self.speeds = torch.sqrt((self.centres[:, 0] - self.last_centres[:, 0])**2 + (self.centres[:, 1] - self.last_centres[:, 1])**2)
        del self.last_centres

    def get_perimeters(self):
        kernel = torch.tensor([[1, 1, 1],
                               [1, 9, 1],
                               [1, 1, 1]]).to(device)

        padded_masks = torch.nn.functional.pad(self.masks, (1, 1, 1, 1), mode='constant', value=0)
        conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
                                                 padding=0).squeeze()
        self.perimeters = torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2)).float()
        self.perimeters[self.perimeters == 0] = float('nan')

    def get_nearest(self):
        self.masks = None # Save memory
        non_zero_pixels = torch.nonzero(self.epi_mask)
        distances = torch.sqrt(torch.sum((self.centres.unsqueeze(0) - non_zero_pixels.unsqueeze(1))**2, dim=2))
        self.dists, i = torch.min(distances, dim=0)

    def get_eaten(self):
        intersection = torch.logical_and(self.masks, self.epi_mask.unsqueeze(0))
        self.eaten = intersection.sum(dim=(1, 2)).int()

def plot_tracks(save_as):
    tracks_plot = torch.zeros((*SETTINGS.IMAGE_SIZE, 3), dtype=torch.uint8).to(device)
    print('\nPLOTTING TRACKS\n')
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        for cell in f['Features']:
            colour = torch.tensor(np.random.uniform(0, 2 ** (8) - 1, size=3), dtype=torch.uint8).to(device)
            xcentres = torch.tensor(f['Features'][cell]['xcentre']).to(device)
            ycentres = torch.tensor(f['Features'][cell]['ycentre']).to(device)
            xcentres, ycentres = xcentres[~torch.isnan(xcentres)], ycentres[~torch.isnan(ycentres)]
            for i in range(len(xcentres) - 1):
                tracks_plot = utils.draw_line(tracks_plot, xcentres[i], xcentres[i+1], ycentres[i], ycentres[i+1], colour)
    print(tracks_plot.shape)
    imageio.imwrite(save_as, tracks_plot.cpu())

def plot_features(save_as):
    plt.rcParams["font.family"] = 'serif'
    print('\nPLOTTING FEATURES\n')
    utils.remake_dir(Path(save_as))
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        for cell in f['Features']:
            data = pd.DataFrame(f['Features'][cell][:])
            #print(data)
            fig, axs = plt.subplots(5, sharex=True, figsize=(10, 10))
            # colours=['firebrick', 'darkorange', 'yellowgreen', 'lightseagreen', 'royalblue']
            for i in range(5):
                axs[i].plot(data.iloc[:, i], color='k')
                axs[i].set(ylabel=data.columns.values.tolist()[i])
                axs[i].grid()

            fig.suptitle(cell)
            axs[-1].set(xlabel='frames')
            plt.savefig(Path(save_as) / (cell+'.png'))
            plt.close()

def show_eating(directory):
    utils.remake_dir(Path(directory))
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        for cell in f['Features']:
            data = pd.DataFrame(f['Features'][cell][:])
            eaten_frames = data.index[data['eaten']>=1].tolist()
            if len(eaten_frames) > 0:
                (Path(directory) / cell).mkdir()
                for eaten_frame in eaten_frames:
                    image = torch.tensor(np.array(f['Images']['Phase'][f'{eaten_frame:04}'])).to(device)
                    epi_image = torch.tensor(np.array(f['Images']['Epi'][f'{eaten_frame:04}'])).to(device)
                    mask = torch.tensor(np.array(f['Segmentations']['Phase'][f'{eaten_frame:04}'])).to(device)
                    outline = mask_funcs.mask_outline(torch.where(mask==int(cell[-4:]), 1, 0), thickness=2)
                    epi_image_normalised = (epi_image - epi_image.min()) / (epi_image.max() - epi_image.min()) * 255
                    im_rgb = torch.stack((image, image, image), axis=0)

                    im_rgb[0] = torch.where(outline, 255, im_rgb[0])
                    im_rgb[1] = torch.where(outline, 255, im_rgb[1])
                    im_rgb[2] = torch.where(outline, 0, im_rgb[2])

                    im_rgb[0] = torch.where(epi_image > SETTINGS.THRESHOLD, epi_image_normalised, im_rgb[0])
                    im_rgb[1] = torch.where(epi_image > SETTINGS.THRESHOLD, 0, im_rgb[1])
                    im_rgb[2] = torch.where(epi_image > SETTINGS.THRESHOLD, 0, im_rgb[2])


                    im_rgb = im_rgb.permute(1, 2, 0)

                    imageio.imwrite(Path(directory) / cell / ("{0:04}".format(eaten_frame) + '.jpg'), (im_rgb).cpu().numpy().astype(np.uint8))
                    #utils.save_tiff((im_rgb).cpu().numpy().astype(np.uint8), directory /("{0:04}".format(eaten_frame) + '.jpg'))

def show_eating_2(directory):
    print('\nSAVING IMAGES OF PHAGOCYTSOSIS\n')
    utils.remake_dir(Path(directory))
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        for cell in f['Features']:
            sys.stdout.write(f'\r{cell}')
            sys.stdout.flush()
            data = pd.DataFrame(f['Features'][cell][:])
            eaten_frames = data.index[data['eaten']>=10].tolist()
            if len(eaten_frames) > 0:
                consecutive_eaten_frames = utils.split_list_into_sequences(eaten_frames)
                for sequence in consecutive_eaten_frames:
                    if len(sequence) >= SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
                        (Path(directory) / cell / str(sequence[0])).mkdir(parents=True)
                        padded_sequence = np.insert(sequence, 0, np.arange(sequence.min()-10, sequence.min))
                        padded_sequence = np.append(padded_sequence, np.arange(sequence.max()+1, sequence.max()+11))
                        for eaten_frame in padded_sequence:
                            image = torch.tensor(np.array(f['Images']['Phase'][f'{eaten_frame:04}'])).to(device)
                            epi_image = torch.tensor(np.array(f['Images']['Epi'][f'{eaten_frame:04}'])).to(device)
                            mask = torch.tensor(np.array(f['Segmentations']['Phase'][f'{eaten_frame:04}'])).to(device)
                            outline = mask_funcs.mask_outline(torch.where(mask == int(cell[-4:]), 1, 0), thickness=2)
                            epi_image_normalised = (epi_image - epi_image.min()) / (epi_image.max() - epi_image.min()) * 255
                            im_rgb = torch.stack((image, image, image), axis=0)

                            im_rgb[0] = torch.where(outline, 255, im_rgb[0])
                            im_rgb[1] = torch.where(outline, 255, im_rgb[1])
                            im_rgb[2] = torch.where(outline, 0, im_rgb[2])

                            im_rgb[0] = torch.where(epi_image > SETTINGS.THRESHOLD, epi_image_normalised, im_rgb[0])
                            im_rgb[1] = torch.where(epi_image > SETTINGS.THRESHOLD, 0, im_rgb[1])
                            im_rgb[2] = torch.where(epi_image > SETTINGS.THRESHOLD, 0, im_rgb[2])

                            im_rgb = im_rgb.permute(1, 2, 0)

                            imageio.imwrite(Path(directory) / cell / str(sequence[0]) / ("{0:04}".format(eaten_frame) + '.jpg'),
                                            (im_rgb).cpu().numpy().astype(np.uint8))




def get_batches(batchsize):
    max_cell_index = 0
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        for i, frame in enumerate(f['Segmentations']['Phase']):
            frame = np.array(f['Segmentations']['Phase'][frame])
            max = np.max(frame)
            if max > max_cell_index:
                max_cell_index = max
    num_batches, remainder = divmod(max_cell_index, batchsize)
    batches = [np.arange(i*batchsize+1, min(((i+1)*batchsize)+1, max_cell_index)) for i in range(0, num_batches+1)]
    return batches

def main():
    # with h5py.File(SETTINGS.DATASET, 'r+') as f:
    #     if 'Features' in f:
    #         del(f['Features'])
    # batches = get_batches(SETTINGS.BATCH_SIZE)
    # print('\nFEATURE EXTRACTION\n')
    # with torch.no_grad():
    #     for batch in batches:
    #         current_cell_batch = CellBatch(torch.tensor(batch).to(device))
    #         current_cell_batch.run_feature_extraction()
    if SETTINGS.PLOT_FEATURES:
        plot_features(str(SETTINGS.DATASET.parent / (SETTINGS.DATASET.stem + 'feature_plots')))
    if SETTINGS.TRACKS_PLOT:
        plot_tracks(str(SETTINGS.DATASET.parent / (SETTINGS.DATASET.stem + 'tracks.png')))
    if SETTINGS.SHOW_EATING:
        show_eating_2(str(SETTINGS.DATASET.parent / (SETTINGS.DATASET.stem + 'show_eating')))



    # torch.cuda.empty_cache()
    # gc.enable()
    # with torch.no_grad():
    #     utils.remake_dir(SETTINGS.DIRECTORY / 'features')
    #     cell_batch = CellBatch(torch.tensor(np.arange(1, 101)).to(device))
    #     cell_batch.run_feature_extraction()
    # #     cell_batch = CellBatch(torch.tensor(np.arange(101, 201)).to(device))
    # #     cell_batch.run_feature_extraction()
    # #     cell_batch = CellBatch(torch.tensor(np.arange(201, 301)).to(device))
    # #     cell_batch.run_feature_extraction()
    # #     cell_batch = CellBatch(torch.tensor(np.arange(301, 401)).to(device))
    # #     cell_batch.run_feature_extraction()
    # # if SETTINGS.PLOT_FEATURES:
    #     plot_features()
    # if SETTINGS.TRACKS_PLOT:
    #     plot_tracks()
    # if SETTINGS.SHOW_EATING:
    #     show_eating()
if __name__ == '__main__':
    main()




