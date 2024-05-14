import torch
import numpy as np
import sys

import utils
import mask_funcs
import SETTINGS


class Cell:
    def __init__(self, index):
        self.index = index
        self.file = self.file = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(self.index) + '.txt')
        with open(self.file, 'w') as f:
            f.write('dist_moved\tarea\tcircularity\toverlap\tdist_nearest\tindex_nearest')

    #def write_features(self):
class CellBatch:
    def __init__(self, indices):
        self.indices = indices
        self.expanded_indices = self.indices.unsqueeze(-1).unsqueeze(-1).expand((len(indices), *SETTINGS.IMAGE_SIZE))
        self.cells = [Cell(i) for i in self.indices]
        self.centres = None
        self.last_centres = None


    def run_feature_extraction(self):
        for i, path in enumerate(sorted([p for p in (SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir()])):
            sys.stdout.write(f'\rFrame {i} | Cells {torch.min(self.indices)}-{torch.max(self.indices)}')
            sys.stdout.flush()
            if i == 0:
                full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
                self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1,0)
            self.next_frame(path)
            self.read_features()
            self.write_features()

    def next_frame(self, path):
        #self.last_masks = self.masks
        full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
        self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
        self.epi_mask = torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / path.name).astype(np.int16)).cuda()

    def read_features(self):
        self.get_areas()
        self.get_centres()
        self.get_speeds()
        self.get_perimeters()


    def write_features(self):
        print(self.perimeters)

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
            self.speeds = 'nan'

        else:
            self.speeds = ((self.centres[:, 0] - self.last_centres[:, 0])**2 + (self.centres[:, 1] - self.last_centres[:, 1])**2)**0.5

    def get_perimeters(self):
        perimeters = torch.zeros(len(self.indices))
        kernel = torch.tensor([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]] ).cuda()
        coords = self.masks.nonzero()
        for coord in coords:
            b, row, col = coord.tolist()
            masks_patch = self.masks[b, row-1:row+1, col-1:col+1]
            print(masks_patch.shape, kernel.shape)
            if masks_patch*kernel < 8:
                perimeters[b] += 1
        self.perimeters = perimeters



def main():
    cell_batch = CellBatch(torch.tensor(np.arange(1, 50)).cuda())
    cell_batch.run_feature_extraction()

if __name__ == '__main__':
    main()