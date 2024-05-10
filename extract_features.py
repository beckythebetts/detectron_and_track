import torch
from skimage import measure
import numpy as np
import sys
import time

import utils
import mask_funcs
import SETTINGS
class Cell:

    def __init__(self, index):
        self.index = index
        self.file = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(self.index) + '.txt')
        with open(self.file, 'w') as f:
            f.write('dist_moved\tarea\tcircularity\toverlap\tdist_nearest\tindex_nearest')
        self.index_exists = False
        self.missing = 0
    def write_features(self):
        self.last_mask = torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'phase' / '0000.tif').astype(np.int16)).cuda()
        for mask_path in sorted((SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir()):
            print(mask_path.stem)
            time_ = time.time()
            full_mask = torch.tensor(utils.read_tiff(mask_path).astype(np.int16)).cuda()
            print(time.time()-time_, 'read mask')
            if self.index in full_mask:
                self.missing=0
                self.index_exists = True
                time_ = time.time()
                self.mask = torch.where(full_mask==self.index, 1, 0)
                print(time.time()-time_, 'extract mask')
                self.centre = self.cell_centre()
                # epi_frame = torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / mask_path.name).astype(np.int16)).cuda()
                # print(torch.unique(epi_frame))
                time_ = time.time()
                dist, index_of_nearest = self.nearest(torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / mask_path.name).astype(np.int16)).cuda())
                print(time.time() - time_, 'find nearest')
                time_ = time.time()
                new_row = '\n' + '\t'.join([str(self.speed().item()), str(self.area().item()), str(self.circularity().item()), str(self.overlap().item()), str(dist), str(index_of_nearest.item())])
                print(time.time() - time_, 'other features')
                self.last_mask = self.mask.clone()
            else:
                new_row = '\n' + '\t'.join(np.full(6, 'nan'))
                self.missing += 1
            time_ = time.time()
            with open(self.file, 'a') as f:
                f.write(new_row)
            print(time.time() - time_, 'writing data')

    def cell_centre(self):
        return mask_funcs.find_centre(self.mask)

    def speed(self):
        x_1, y_1 = self.centre
        x_0, y_0 = mask_funcs.find_centre(self.last_mask)
        return ((x_1-x_0)**2 + (y_1-y_0)**2)**0.5

    def area(self):
        return torch.sum(self.mask)

    def circularity(self):
        perimeter = measure.perimeter(self.mask.cpu().numpy())
        return 4*np.pi*self.area() / (measure.perimeter(self.mask.cpu().numpy())**2)

    def overlap(self):
        intersection = torch.logical_and(self.mask, self.last_mask)
        union = torch.logical_or(self.mask, self.last_mask)
        return torch.sum(intersection) / torch.sum(union)

    def nearest(self, other_frame):
        dist = 0
        index_of_nearest = None
        while index_of_nearest is None:
            circle_mask = mask_funcs.torch_circle(self.centre, dist)
            intersection = torch.logical_and(circle_mask, other_frame>0)
            unique_values, counts = torch.unique(other_frame[intersection], return_counts=True)
            if len(unique_values) > 0:
                index_of_nearest = unique_values[torch.argmax(counts)]
            else:
                dist += 1
        return dist, index_of_nearest

def batch_write_features(cells):
    last_mask = torch.tensor(utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'phase' / '0000.tif').astype(np.int16)).cuda()
    for mask_path in sorted((SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir()):
        full_mask = torch.tensor(utils.read_tiff(mask_path).astype(np.int16)).cuda()
        indices = torch.tensor([int(cell.index) for cell in cells]).cuda()
        print(indices)
        # mask_indices = torch.tensor(indices).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # mask_indices = mask_indices.expand(-1, *full_mask.shape)
        #mask_indices = torch.tensor([torch.full(full_mask.shape, index) for index in indices]).cuda()
        mask_indices = torch.tensor([torch.full(full_mask.shape, index) for index in indices]).cuda()
        mask_batch = torch.where(full_mask == mask_indices, 1, 0)
        for cell, mask in zip(cells, mask_batch):
            if cell.index in full_mask:
                cell.index_exists = True
                cell.mask = mask
                cell.centre = cell.cell_centre()
                dist, index_of_nearest = cell.nearest(torch.tensor(
                    utils.read_tiff(SETTINGS.DIRECTORY / 'tracked' / 'epi' / mask_path.name).astype(np.int16)).cuda())
                new_row = '\n' + '\t'.join(
                    [str(cell.speed().item()), str(cell.area().item()), str(cell.circularity().item()),
                     str(cell.overlap().item()), str(dist), str(index_of_nearest.item())])
                cell.last_mask = cell.mask.clone()
            else:
                new_row = '\n' + '\t'.join(np.full(6, 'nan'))
            with open(cell.file, 'a') as f:
                f.write(new_row)
def main():
    print('\n--------------------\nEXTRACTING FEATURES\n--------------------')

    utils.remake_dir(SETTINGS.DIRECTORY / 'features')
    #reached_max_index = False
    max_index = 200
    cell_index = 1
    batch_size = 10
    while cell_index < max_index:
        sys.stdout.write(f'\rCells {cell_index}-{cell_index + batch_size - 1}')
        sys.stdout.flush()

        batch_cells = []
        for i in range(batch_size):
            cell = Cell(cell_index)
            batch_cells.append(cell)
            cell_index += 1


        if batch_cells:
            batch_write_features(batch_cells)

    # while not reached_max_index:
    #     sys.stdout.write(f'\rCell {cell_index}')
    #     sys.stdout.flush()
    #     cell = Cell(cell_index)
    #     cell.write_features()
    #     if not cell.index_exists:
    #         reached_max_index=True
    #     cell_index += 1
    print(f'Completed, {cell_index-1} cells')


if __name__ == '__main__':
    main()