import torch
import numpy as np
import sys
import math
import pandas as pd

import utils
import mask_funcs
import SETTINGS

# def print_gpu_memory_usage(stage):
#     print(f"[{stage}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
#           f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
#
# class Cell:
#     def __init__(self, index):
#         self.index = index
#         self.file = self.file = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(self.index) + '.txt')
#         with open(self.file, 'w') as f:
#             f.write('area\tspeed\tperimeter\tdist_nearest\tindex_nearest')
#
#     def write_features(self, line):
#         with open(self.file, 'a') as f:
#             f.write(line)
# class CellBatch:
#     def __init__(self, indices):
#         self.indices = indices
#         self.expanded_indices = self.indices.unsqueeze(-1).unsqueeze(-1).expand((len(indices), *SETTINGS.IMAGE_SIZE))
#         self.cells = [Cell(i) for i in self.indices]
#         self.centres = None
#         self.last_centres = None
#         self.batch_size = len(self.indices)
#         self.paths = sorted([p for p in (SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir()])
#         self.num_frames = len(self.paths)
#         self.coord_grid_x, self.coord_grid_y = torch.meshgrid(torch.arange(SETTINGS.IMAGE_SIZE[0]).cuda(),
#                                                               torch.arange(SETTINGS.IMAGE_SIZE[1]).cuda())
#
#     def run_feature_extraction(self):
#         for i, path in enumerate(self.paths):
#             sys.stdout.write(f'\rFrame {i} | Cells {torch.min(self.indices)}-{torch.max(self.indices)}')
#             sys.stdout.flush()
#             if i == 0:
#                 full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
#                 self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1,0)
#                 full_mask = None
#             print_gpu_memory_usage('Start')
#             self.next_frame(path)
#             print_gpu_memory_usage('Read new frames')
#             self.read_features()
#             print_gpu_memory_usage('Read features')
#             self.epi_mask = None
#             print_gpu_memory_usage('Deleted mask')
#             self.write_features()
#             print_gpu_memory_usage('Written features')
#             torch.cuda.empty_cache()
#             print_gpu_memory_usage('Emptied cache')
#
#     def next_frame(self, path):
#         full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
#         self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
#         full_mask = None
#         self.epi_mask = torch.tensor(
#             utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / path.name).astype(np.int16)).cuda()
#
#
#
#     def read_features(self):
#         self.get_areas()
#         self.get_centres()
#         self.get_speeds()
#         self.get_perimeters()
#         del self.masks
#         self.get_nearest()
#
#     def write_features(self):
#         for i, cell in enumerate(self.cells):
#             #print([type(str(a.item())) for a in (self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i], self.indices_of_nearest[i])])
#             new_line = '\n' + '\t'.join([str(a.item()) for a in (self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i], self.indices_of_nearest[i])])
#             cell.write_features(new_line)
#             del new_line
#         del self.areas, self.speeds, self.perimeters, self.dists, self.indices_of_nearest
#
#     def get_areas(self):
#         self.areas = torch.sum(self.masks, dim=(1, 2))
#
#     def get_centres(self):
#         if self.centres is not None:
#             self.last_centres = self.centres
#
#
#         x_centres = torch.sum(self.masks * self.coord_grid_x, dim=(1, 2)) / self.areas
#         y_centres = torch.sum(self.masks * self.coord_grid_y, dim=(1, 2)) / self.areas
#
#         self.centres = torch.stack((x_centres, y_centres), dim=1)
#
#
#     def get_speeds(self):
#         if self.last_centres is None:
#             self.speeds = torch.full((self.batch_size,), float('nan'))
#
#         else:
#             self.speeds = torch.sqrt((self.centres[:, 0] - self.last_centres[:, 0])**2 + (self.centres[:, 1] - self.last_centres[:, 1])**2)
#         del self.last_centres
#
#     def get_perimeters(self):
#         kernel = torch.tensor([[1, 1, 1],
#                                [1, 9, 1],
#                                [1, 1, 1]] ).cuda()
#
#         padded_masks = torch.nn.functional.pad(self.masks, (1, 1, 1, 1), mode='constant', value=0)
#         conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
#                                                  padding=0).squeeze()
#         self.perimeters = torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2))
#         del padded_masks, conv_result
#
#     def get_epi_centres(self):
#
#         self.epi_indices, self.epi_areas = torch.unique(self.epi_mask, return_counts=True)
#
#         self.expanded_epi_indices = self.epi_indices.unsqueeze(-1).unsqueeze(-1).expand((len(self.epi_indices), *SETTINGS.IMAGE_SIZE))
#         self.epi_masks = torch.where(self.epi_mask.unsqueeze(0).expand(len(self.epi_indices), *SETTINGS.IMAGE_SIZE) == self.expanded_epi_indices, 1, 0)
#
#         x_centres = torch.sum(self.epi_masks * self.coord_grid_x, dim=(1, 2)) / self.epi_areas
#         y_centres = torch.sum(self.epi_masks * self.coord_grid_y, dim=(1, 2)) / self.epi_areas
#
#         self.epi_centres = torch.stack((x_centres, y_centres), dim=1)
#
#         self.epi_masks = None
#         self.expanded_epi_indices = None
#         del x_centres, y_centres
#
#     def get_nearest(self):
#         self.get_epi_centres()
#         self.dists, self.indices_of_nearest = torch.tensor([]).cuda(), torch.tensor([]).cuda()
#         centres_expanded = self.centres.unsqueeze(1)
#         distances = torch.sqrt(torch.sum((centres_expanded - self.epi_centres) ** 2, dim=2))
#         min_distances, min_indices = torch.min(distances, dim=1)
#         self.dists = torch.cat((self.dists, min_distances.unsqueeze(1)), dim=0)
#         self.indices_of_nearest = torch.cat((self.indices_of_nearest, self.epi_indices[min_indices]), dim=0)
#         del centres_expanded, self.epi_centres, distances, min_distances, min_indices

# def plot_features():
#     for dir in (SETTINGS.DIRECTORY / 'features').iterdir():
#         data = pd.read_csv(dir), sep='\t')
#         fig, axs = plt.subplots(5, sharex=True, figsize=(10, 10))
#         for i in range(4):
#             axs[i].plot(data.iloc[:, i])
#             axs[i].set(ylabel=data.columns.values.tolist()[i])
#             axs[i].grid()
#
#         yeast_indexes = np.unique(data.iloc[:, 5])
#         for yeast in yeast_indexes[np.isnan(yeast_indexes) == False]:
#             print(yeast)
#             axs[4].plot(data.query('index_nearest == @yeast').loc[:, 'dist_nearest'], label=str(yeast), linestyle='', marker='.')
#             axs[4].set(ylabel='nearest yeast')
#         axs[4].grid()
#
#         fig.suptitle('Amoeba '+index)
#         axs[-1].set(xlabel='frames')
#         plt.legend(title='Index of yeast', ncol=2)
#         plt.tight_layout()
#         plt.savefig(dir / str('Amoeba_'+index+'.png'))

#
# def main():
#     torch.cuda.empty_cache()
#     with torch.no_grad():
#         utils.remake_dir(SETTINGS.DIRECTORY / 'features')
#         cell_batch = CellBatch(torch.tensor(np.arange(1, 101)).cuda())
#         cell_batch.run_feature_extraction()
#
# if __name__ == '__main__':
#     main()

import torch
import numpy as np
import sys
import gc
import utils
import SETTINGS

def print_memory_stats(stage):
    print(f"[{stage}]")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
          f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    stats = torch.cuda.memory_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

class Cell:
    def __init__(self, index):
        self.index = index
        self.file = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(self.index) + '.txt')
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
        self.coord_grid_x, self.coord_grid_y = torch.meshgrid(
            torch.arange(SETTINGS.IMAGE_SIZE[0]),
            torch.arange(SETTINGS.IMAGE_SIZE[1])
        )
        self.coord_grid_x, self.coord_grid_y = self.coord_grid_x.cuda(), self.coord_grid_y.cuda()
        print_memory_stats("Init")

    def run_feature_extraction(self):
        for i, path in enumerate(self.paths):
            sys.stdout.write(f'\rFrame {i} | Cells {torch.min(self.indices)}-{torch.max(self.indices)}')
            sys.stdout.flush()
            if i == 0:
                full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
                self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
                del full_mask
                torch.cuda.empty_cache()
                gc.collect()
            self.next_frame(path)
            print_memory_stats("Read new frames")
            self.read_features()
            print_memory_stats("Read features")
            self.write_features()
            del self.epi_mask
            torch.cuda.empty_cache()
            gc.collect()
            print_memory_stats(f"Frame {i}")
            print('\n')  # Ensure each frame's usage is on a new line

    def next_frame(self, path):
        full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
        self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
        del full_mask
        torch.cuda.empty_cache()  # Clear cache after deleting full_mask
        gc.collect()
        self.epi_mask = torch.tensor(
            utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / path.name).astype(np.int16)).cuda()

    def read_features(self):
        self.get_areas()
        self.get_centres()
        self.get_speeds()
        self.get_perimeters()
        del self.masks
        torch.cuda.empty_cache()  # Empty cache after deleting masks to free up memory
        gc.collect()
        self.get_nearest()

    def write_features(self):
        for i, cell in enumerate(self.cells):
            new_line = '\n' + '\t'.join([str(a.item()) for a in (self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i], self.indices_of_nearest[i])])
            cell.write_features(new_line)
        del self.areas, self.speeds, self.perimeters, self.dists, self.indices_of_nearest
        torch.cuda.empty_cache()  # Empty cache after deleting feature tensors to free up memory
        gc.collect()

    def get_areas(self):
        self.areas = torch.sum(self.masks, dim=(1, 2))

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

    def get_perimeters(self):
        kernel = torch.tensor([[1, 1, 1], [1, 9, 1], [1, 1, 1]]).cuda()

        padded_masks = torch.nn.functional.pad(self.masks, (1, 1, 1, 1), mode='constant', value=0)
        conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(), padding=0).squeeze()
        self.perimeters = torch.sum((conv_result >= 10) & (conv_result <= 16), dim=(1, 2))
        del padded_masks, conv_result

    def get_epi_centres(self):
        self.epi_indices, self.epi_areas = torch.unique(self.epi_mask, return_counts=True)
        self.expanded_epi_indices = self.epi_indices.unsqueeze(-1).unsqueeze(-1).expand((len(self.epi_indices), *SETTINGS.IMAGE_SIZE))
        self.epi_masks = torch.where(self.epi_mask.unsqueeze(0).expand(len(self.epi_indices), *SETTINGS.IMAGE_SIZE) == self.expanded_epi_indices, 1, 0)

        x_centres = torch.sum(self.epi_masks * self.coord_grid_x, dim=(1, 2)) / self.epi_areas
        y_centres = torch.sum(self.epi_masks * self.coord_grid_y, dim=(1, 2)) / self.epi_areas

        self.epi_centres = torch.stack((x_centres, y_centres), dim=1)

        del self.epi_masks, self.expanded_epi_indices, x_centres, y_centres

    def get_nearest(self):
        self.get_epi_centres()
        centres_expanded = self.centres.unsqueeze(1)
        distances = torch.sqrt(torch.sum((centres_expanded - self.epi_centres) ** 2, dim=2))
        min_distances, min_indices = torch.min(distances, dim=1)
        self.dists = min_distances
        self.indices_of_nearest = self.epi_indices[min_indices]
        del centres_expanded, self.epi_centres, distances, min_distances, min_indices

def main():
    torch.cuda.empty_cache()
    with torch.no_grad():
        utils.remake_dir(SETTINGS.DIRECTORY / 'features')
        cell_batch = CellBatch(torch.tensor(np.arange(1, 101)).cuda())
        cell_batch.run_feature_extraction()
        torch.cuda.empty_cache()
        gc.collect()
        print_memory_stats("End")

if __name__ == '__main__':
    main()


