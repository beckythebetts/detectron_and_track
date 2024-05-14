import torch
import numpy as np

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


    def run_feature_extraction(self):
        for i, path in enumerate(sorted([p for p in (SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir()])):
            if i == 0:
                full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
                self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1,0)
            self.next_frame(path)
            self.read_features()
            self.write_features()
    def next_frame(self, path):
        self.last_masks = self.masks
        full_mask = torch.tensor(utils.read_tiff(path).astype(np.int16)).cuda()
        self.masks = torch.where(full_mask.unsqueeze(0).expand(len(self.indices), *full_mask.shape) == self.expanded_indices, 1, 0)
        self.epi_mask = torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / path.name).astype(np.int16)).cuda()

    def read_features(self):
        self.centres = self.get_centres()
        return centres

    def write_features(self):
        print(self.centres)
    def get_centres(self):
        coords = torch.nonzero(self.masks)
        x_means = torch.sum(coords, dim=1)

def main():
    cell_batch = CellBatch(torch.tensor(np.arange(1, 10)).cuda())
    cell_batch.run_feature_extraction()

if __name__ == '__main__':
    main()