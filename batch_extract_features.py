import torch
import numpy as np
import sys
import math
import pandas as pd
import gc
import matplotlib.pyplot as plt
import subprocess
import os

import utils
import mask_funcs
import SETTINGS

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class Cell:
    def __init__(self, index):
        self.index = index
        self.file = self.file = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(self.index) + '.txt')
        with open(self.file, 'w') as f:
            f.write('area\tspeed\tperimeter\tdist_nearest')

    def write_features(self, line):
        with open(self.file, 'a') as f:
            f.write(line)
class CellBatch:
    def __init__(self, indices):
        self.indices = indices
        self.expanded_indices = self.indices.unsqueeze(-1).unsqueeze(-1).expand((len(indices), *SETTINGS.IMAGE_SIZE))
        self.cells = [Cell(i) for i in self.indices]
        self.centres = None
        self.last_centres = None
        self.batch_size = len(self.indices)
        self.paths = sorted([p for p in (SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir()])
        self.num_frames = len(self.paths)
        self.coord_grid_x, self.coord_grid_y = torch.meshgrid(torch.arange(SETTINGS.IMAGE_SIZE[0]).cuda(),
                                                              torch.arange(SETTINGS.IMAGE_SIZE[1]).cuda())
        self.memory_usage = SETTINGS.DIRECTORY / 'features_memory.txt'

    def print_gpu_memory(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        memory_used = result.stdout.decode('utf-8').strip().split('\n')[0]
        with open(self.memory_usage, 'a') as f:
            f.write(f'{memory_used}\n')

    def run_feature_extraction(self):
        for i, path in enumerate(self.paths):
            sys.stdout.write(f'\rFrame {i} | Cells {torch.min(self.indices)}-{torch.max(self.indices)} ')
            sys.stdout.flush()
            if i == 0:
                full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
                self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1,0)
                full_mask = None
            self.print_gpu_memory()
            torch.cuda.empty_cache()
            self.next_frame(path)
            self.print_gpu_memory()
            torch.cuda.empty_cache()
            self.read_features()
            self.print_gpu_memory()
            torch.cuda.empty_cache()
            self.epi_mask = None
            torch.cuda.empty_cache()
            self.print_gpu_memory()
            torch.cuda.empty_cache()
            self.write_features()
            torch.cuda.empty_cache()
            self.print_gpu_memory()
            torch.cuda.empty_cache()
            gc.collect()


    def next_frame(self, path):
        full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
        self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
        full_mask = None
        self.epi_mask = torch.tensor(
            utils.read_tiff(SETTINGS.DIRECTORY / 'segmented' / 'epi' / path.name).astype(np.int16)).cuda()

    def read_features(self):
        self.get_areas()
        self.get_centres()
        self.get_speeds()
        self.get_perimeters()
        self.masks = None
        self.get_nearest()
        self.get_eaten()

    def write_features(self):
        for i, cell in enumerate(self.cells):
            new_line = '\n' + '\t'.join([str(a.item()) for a in (self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i])])
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
                               [1, 1, 1]] ).cuda()

        padded_masks = torch.nn.functional.pad(self.masks, (1, 1, 1, 1), mode='constant', value=0)
        conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
                                                 padding=0).squeeze()
        self.perimeters = torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2)).float()
        self.perimeters[self.perimeters == 0] = float('nan')

    def get_nearest(self):
        non_zero_pixels = torch.nonzero(self.epi_mask)
        distances = torch.sqrt(torch.sum((self.centres.unsqueeze(0) - non_zero_pixels.unsqueeze(1))**2, dim=2))
        self.dists, i = torch.min(distances, dim=0)

    def get_eaten(self):
        #intersection = torch.logical_and(self.masks, self.epi_masks.unsqueeze(-1))
        print(self.masks.shape)
        #unique = torch.unique(self.masks[intersection], dim=)


def plot_features():
    print('\n----------\nPlotting Features\n----------\n')
    utils.remake_dir(SETTINGS.DIRECTORY / 'features_plots')
    for features_path in (SETTINGS.DIRECTORY / 'features').iterdir():
        data = pd.read_csv(features_path, sep='\t')
        fig, axs = plt.subplots(4, sharex=True, figsize=(10, 10))
        for i in range(4):
            axs[i].plot(data.iloc[:, i])
            axs[i].set(ylabel=data.columns.values.tolist()[i])
            axs[i].grid()

        fig.suptitle('Amoeba '+ features_path.stem)
        axs[-1].set(xlabel='frames')
        plt.savefig(SETTINGS.DIRECTORY / 'features_plots' / str(features_path.stem+'.png'))
        plt.close()


def main():
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.cuda.empty_cache()
    gc.enable()
    with torch.no_grad():
        utils.remake_dir(SETTINGS.DIRECTORY / 'features')
        cell_batch = CellBatch(torch.tensor(np.arange(1, 101)).cuda())
        cell_batch.run_feature_extraction()
    if SETTINGS.PLOT_FEATURES:
        plot_features()
if __name__ == '__main__':
    main()




