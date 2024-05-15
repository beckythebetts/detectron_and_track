import torch
import numpy as np
import sys
import math

import utils
import mask_funcs
import SETTINGS


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

    def run_feature_extraction(self):
        for i, path in enumerate(self.paths):
            sys.stdout.write(f'\rFrame {i} | Cells {torch.min(self.indices)}-{torch.max(self.indices)}')
            sys.stdout.flush()
            if i == 0:
                full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
                self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1,0)
            self.next_frame(path)
            self.read_features()
            self.write_features()

    def next_frame(self, path):
        full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
        self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
        self.epi_mask = torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / path.name).astype(np.int16)).cuda()


    def read_features(self):
        self.get_areas()
        self.get_centres()
        self.get_speeds()
        self.get_perimeters()
        self.get_nearest()

    def write_features(self):
        for i, cell in enumerate(self.cells):
            print([type(a) for a in (self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i], self.indices_of_nearest[i])])
            new_line = '\n' + '\t'.join([self.areas[i], self.speeds[i], self.perimeters[i], self.dists[i], self.indices_of_nearest[i]])
            cell.write_features(new_line)

    def get_areas(self):
        self.areas = torch.sum(self.masks, dim=(1, 2))

    def get_centres(self):
        if self.centres is not None:
            self.last_centres = self.centres
        self.coord_grid_x, self.coord_grid_y = torch.meshgrid(torch.arange(SETTINGS.IMAGE_SIZE[0]).cuda(), torch.arange(SETTINGS.IMAGE_SIZE[1]).cuda())

        x_centres = torch.sum(self.masks * self.coord_grid_x, dim=(1, 2)) / self.areas
        y_centres = torch.sum(self.masks * self.coord_grid_y, dim=(1, 2)) / self.areas

        self.centres = torch.stack((x_centres, y_centres), dim=1)


    def get_speeds(self):
        if self.last_centres is None:
            self.speeds = torch.full((self.batch_size,), float('nan'))

        else:
            self.speeds = torch.sqrt((self.centres[:, 0] - self.last_centres[:, 0])**2 + (self.centres[:, 1] - self.last_centres[:, 1])**2)

    def get_perimeters(self):
        kernel = torch.tensor([[1, 1, 1],
                               [1, 9, 1],
                               [1, 1, 1]] ).cuda()

        padded_masks = torch.nn.functional.pad(self.masks, (1, 1, 1, 1), mode='constant', value=0)
        conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
                                                 padding=0).squeeze()
        perimeters = torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2))
        self.perimeters = perimeters

    def get_epi_centres(self):
        self.epi_indices, self.epi_areas = torch.unique(self.epi_mask, return_counts=True)
        self.expanded_epi_indices = self.epi_indices.unsqueeze(-1).unsqueeze(-1).expand((len(self.epi_indices), *SETTINGS.IMAGE_SIZE))
        self.epi_masks = torch.where(self.epi_mask.unsqueeze(0).expand(len(self.epi_indices), *SETTINGS.IMAGE_SIZE) == self.expanded_epi_indices, 1, 0)

        x_centres = torch.sum(self.epi_masks * self.coord_grid_x, dim=(1, 2)) / self.epi_areas
        y_centres = torch.sum(self.epi_masks * self.coord_grid_y, dim=(1, 2)) / self.epi_areas

        self.epi_centres = torch.stack((x_centres, y_centres), dim=1)

    def get_nearest(self):
        self.get_epi_centres()
        self.dists, self.indices_of_nearest = torch.tensor([]).cuda(), torch.tensor([]).cuda()
        centres_expanded = self.centres.unsqueeze(1)
        distances = torch.sqrt(torch.sum((centres_expanded - self.epi_centres) ** 2, dim=2))
        min_distances, min_indices = torch.min(distances, dim=1)
        self.dists = torch.cat((self.dists, min_distances.unsqueeze(1)), dim=0)
        self.indices_of_nearest = torch.cat((self.indices_of_nearest, self.epi_indices[min_indices]), dim=0)

def main():
    utils.remake_dir(SETTINGS.DIRECTORY / 'features')
    cell_batch = CellBatch(torch.tensor(np.arange(1, 51)).cuda())
    cell_batch.run_feature_extraction()

if __name__ == '__main__':
    main()