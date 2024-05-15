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
        self.batch_size = len(self.indices)


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
        self.get_nearest()

    def write_features(self):
        print(self.perimeters, self.dists, self.index_of_nearest)

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
        kernel = torch.tensor([[1, 1, 1],
                               [1, 9, 1],
                               [1, 1, 1]] ).cuda()

        padded_masks = torch.nn.functional.pad(self.masks, (1, 1, 1, 1), mode='constant', value=0)
        conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
                                                 padding=0).squeeze()
        perimeters = torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2))
        self.perimeters = perimeters

    def get_nearest(self):
        #print(self.centres)
        dists = torch.zeros(len(self.indices)).cuda()
        indices_of_nearest = torch.full((len(self.indices),), -1).cuda()
        self.expanded_epi_mask = self.epi_mask.unsqueeze(0).expand(len(self.indices), *SETTINGS.IMAGE_SIZE)
        radius = 0
        centres_copy = self.centres.clone().detach()
        while any(indices_of_nearest == -1):
            circle_masks = torch.stack([mask_funcs.torch_circle(centre, radius) for centre in centres_copy], dim=0)
            intersections = torch.logical_and(circle_masks, self.expanded_epi_mask>0)
            for i in range(self.batch_size):
                if centres_copy[i].isnan().any():
                    dists[i] = 'nan'
                    indices_of_narest[i] = 'nan'
                else:
                    unique, count = torch.unique(self.epi_mask[intersections[i]], return_counts=True)
                    if len(unique) > 0:
                        dists[i] = radius
                        indices_of_nearest[i] = unique[torch.argmax(count)]
                        centres_copy[i] = [float('nan'), float('nan')]
            radius += 1




        self.dists, self.index_of_nearest = dists, indices_of_nearest


def main():
    cell_batch = CellBatch(torch.tensor(np.arange(1, 51)).cuda())
    cell_batch.run_feature_extraction()

if __name__ == '__main__':
    main()