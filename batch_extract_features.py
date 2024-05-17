import torch
import numpy as np
import sys
import math
import pandas as pd
import gc
import matplotlib.pyplot as plt
import subprocess

import utils
import mask_funcs
import SETTINGS

# def print_gpu_memory_usage(stage):
#     print(f"[{stage}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
#           f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# def print_gpu_memory():
#     result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
#     memory_used = result.stdout.decode('utf-8').strip()
#     sys.stdout.write(
#         f'\rFrame {i} | Cells {torch.min(self.indices)}-{torch.max(self.indices)} | GPU memory used: {memory_used}')
#     sys.stdout.flush()
#     with open(memory_usage, 'a') as f:
#         f.write(memory_used)

class Cell:
    def __init__(self, index):
        self.index = index
        self.file = self.file = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(self.index) + '.txt')
        with open(self.file, 'w') as f:
            f.write('area\tspeed\tperimeter\tdist_nearest\tindex_nearest')

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
        self.masks = None

    def print_gpu_memory(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        memory_used = result.stdout.decode('utf-8').strip().split('\n')[0]
        # sys.stdout.write(
        #     f'\rGPU memory used: {memory_used}\n')
        # sys.stdout.flush()
        with open(self.memory_usage, 'a') as f:
            f.write(f'{sys.getsizeof(self.masks), sys.getsizeof(self.epi_mask), sys.getsizeof(self.epi_masks), sys.getsizeof(self.)}\n')

    def run_feature_extraction(self):
        for i, path in enumerate(self.paths):
            sys.stdout.write(f'\rFrame {i} | Cells {torch.min(self.indices)}-{torch.max(self.indices)} ')
            sys.stdout.flush()
            if i == 0:
                full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
                self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1,0)
                full_mask = None
            self.print_gpu_memory()
            self.next_frame(path)
            self.print_gpu_memory()
            self.read_features()
            self.print_gpu_memory()
            self.epi_mask = None
            self.print_gpu_memory()
            self.write_features()
            self.print_gpu_memory()
            torch.cuda.empty_cache()
            gc.collect()


    def next_frame(self, path):
        full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
        self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
        full_mask = None
        self.epi_mask = torch.tensor(
            utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / path.name).astype(np.int16)).cuda()



    def read_features(self):
        self.get_areas()
        self.get_centres()
        self.get_speeds()
        self.get_perimeters()
        self.masks = None
        self.get_nearest()

    def write_features(self):
        for i, cell in enumerate(self.cells):
            #print([type(str(a.item())) for a in (self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i], self.indices_of_nearest[i])])
            new_line = '\n' + '\t'.join([str(a.item()) for a in (self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i], self.indices_of_nearest[i])])
            cell.write_features(new_line)
            del new_line
        del self.areas, self.speeds, self.perimeters, self.dists, self.indices_of_nearest

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
        del padded_masks, conv_result

    def get_epi_centres(self):

        self.epi_indices, self.epi_areas = torch.unique(self.epi_mask, return_counts=True)

        self.expanded_epi_indices = self.epi_indices.unsqueeze(-1).unsqueeze(-1).expand((len(self.epi_indices), *SETTINGS.IMAGE_SIZE))
        self.epi_masks = torch.where(self.epi_mask.unsqueeze(0).expand(len(self.epi_indices), *SETTINGS.IMAGE_SIZE) == self.expanded_epi_indices, 1, 0)

        x_centres = torch.sum(self.epi_masks * self.coord_grid_x, dim=(1, 2)) / self.epi_areas
        y_centres = torch.sum(self.epi_masks * self.coord_grid_y, dim=(1, 2)) / self.epi_areas

        self.epi_centres = torch.stack((x_centres, y_centres), dim=1)

        self.epi_masks = None
        self.expanded_epi_indices = None
        del x_centres, y_centres

    def get_nearest(self):
        self.get_epi_centres()
        self.dists, self.indices_of_nearest = torch.tensor([]).cuda(), torch.tensor([]).cuda()
        centres_expanded = self.centres.unsqueeze(1)
        distances = torch.sqrt(torch.sum((centres_expanded - self.epi_centres) ** 2, dim=2))
        min_distances, min_indices = torch.min(distances, dim=1)
        self.dists = torch.cat((self.dists, min_distances.unsqueeze(1)), dim=0)
        self.indices_of_nearest = torch.cat((self.indices_of_nearest, self.epi_indices[min_indices]), dim=0)
        del centres_expanded, self.epi_centres, distances, min_distances, min_indices

def plot_features():
    print('\n---------- Plotting Features\n----------\n')
    utils.remake_dir(SETTINGS.DIRECTORY / 'features_plots')
    for features_path in (SETTINGS.DIRECTORY / 'features').iterdir():
        data = pd.read_csv(features_path, sep='\t')
        fig, axs = plt.subplots(4, sharex=True, figsize=(10, 10))
        for i in range(4):
            axs[i].plot(data.iloc[:, i])
            axs[i].set(ylabel=data.columns.values.tolist()[i])
            axs[i].grid()

        yeast_indexes = np.unique(data.iloc[:, 4])
        for yeast in yeast_indexes[np.isnan(yeast_indexes) == False]:
            #print(yeast)
            axs[3].plot(data.query('index_nearest == @yeast').loc[:, 'dist_nearest'], label=str(yeast), linestyle='', marker='.')
            axs[3].set(ylabel='nearest yeast')
        axs[3].grid()

        fig.suptitle('Amoeba '+ features_path.stem)
        axs[-1].set(xlabel='frames')
        #plt.legend(title='Index of yeast', ncol=2)
        #plt.tight_layout()
        plt.savefig(SETTINGS.DIRECTORY / 'features_plots' / str(features_path.stem+'.png'))


def main():
    torch.cuda.empty_cache()
    gc.enable()
    with torch.no_grad():
        utils.remake_dir(SETTINGS.DIRECTORY / 'features')
        cell_batch = CellBatch(torch.tensor(np.arange(1, 11)).cuda())
        cell_batch.run_feature_extraction()
    if SETTINGS.PLOT_FEATURES:
        plot_features()
if __name__ == '__main__':
    main()




