import SETTINGS
import torch
import utils
from skimage import measure
import numpy as np
import sys

import mask_funcs

class Cell:

    def __init__(self, index):
        self.index = index
        self.file = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(self.index) + '.txt')
        with open(self.file, 'w') as f:
            f.write('dist_moved\tarea\tcircularity\toverlap\tdist_nearest\tindex_nearest')
        self.index_exists = False
    def write_features(self):
        self.last_mask = torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'phase' / '0000.tif').astype(np.int16)).cuda()
        for mask_path in (SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir():
            full_mask = torch.tensor(utils.read_tiff(mask_path).astype(np.int16)).cuda()
            if self.index in full_mask:
                self.index_exists = True
                self.mask = torch.where(full_mask==self.index, 1, 0)
                self.centre = self.cell_centre()
                dist, index_of_nearest = self.nearest(torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / mask_path.name).astype(np.int16)).cuda())
                new_row = '\n' + '\t'.join(str(self.cell_centre()), str(self.speed()), str(self.area()), str(self.circularity()), str(self.overlap()), str(dist), str(index_of_nearest))
                with open(self.file, 'a') as f:
                    f.write(new_row)
                self.last_mask = self.mask.clone()

    def cell_centre(self):
        return mask_funcs.find_centre(self.mask)

    def speed(self):
        x_1, y_1 = self.centre
        x_0, y_0 = mask_funcs.centre(self.last_mask)
        return ((x_1-x_0)**2 + (y_1-y_0)**2)**0.5

    def area(self):
        self.area = torch.sum(self.mask)
        return self.area

    def circularity(self):
        perimeter = measure.perimeter(self.mask.cpu().numpy())
        return 4*np.pi*self.area(time) / (measure.perimeter(self.masks)**2)

    def overlap(self):
        intersection = torch.logical_and(self.mask, self.last_mask)
        union = torch.logical_or(self.mask, self.old_mask)
        return intersection / union

    def nearest(self, other_frame):
        dist = 0
        index_of_nearest = None
        while index_of_nearest is None:
            sys.stdout.write(f'\rDistance {dist}')
            sys.stdout.flush()
            circle_mask = mask_funcs.torch_circle(self.centre, 0)
            intersection = torch.logical_and(circle_mask, other_frame)
            unique_values, counts = torch.unique(other_frame[intersection], return_counts=True)
            if len(unique_values) > 0:
                index_of_nearest = unique_values[torch.argmax(counts)]
            else:
                dist += 1
        return dist, index_of_nearest

def main():
    print('\n--------------------\nEXTRACTING FEATURES\n--------------------')

    utils.remake_dir(SETTINGS.DIRECTORY / 'features')
    reached_max_index = False
    cell_index = 1
    while not reached_max_index:
        sys.stdout.write(f'\rCell {cell_index}')
        sys.stdout.flush()
        cell = Cell(cell_index)
        cell.write_features()
        if not cell.index_exists:
            reached_max_index=True
        cell_index += 1
    print(f'Completed, {cell_index-1} cells')


if __name__ == '__main__':
    main()