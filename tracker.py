import numpy as np
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import time
import sys
import torch

from Cells import Cell
import mask_funcs
import SETTINGS
import utils

class Tracker:

    def __init__(self, name):
        self.name = name
        self.mask_ims = sorted([mask for mask in (SETTINGS.DIRECTORY / 'inference_dataset' / 'masks' / self.name).iterdir()])
        self.cells = [Cell(masks=mask[np.newaxis, :, :], index=i+1, type=self.name) for i, mask in enumerate(mask_funcs.split_mask(self.new_frame(0)))]
        utils.remake_dir(SETTINGS.DIRECTORY / 'tracking' / self.name)
        if SETTINGS.VIEW_TRACKS:
            utils.remake_dir(SETTINGS.DIRECTORY / 'tracking_view' / self.name)

    def max_cell_index(self):
        if len(self.cells) == 0:
            return 0
        else:
            return np.max([cell.index for cell in self.cells])

    def new_frame(self, index):
        return plt.imread(self.mask_ims[index])

    def join_new_frame(self, index):
        if index > SETTINGS.TRACK_CLIP_LENGTH:
            for cell in self.cells:
                cell.clip_track()

        #orig_new_mask = self.new_frame(index).astype(np.float32)
        orig_new_mask = torch.tensor(self.new_frame(index).astype(np.float32), dtype=torch.float32).cuda()
        new_mask = torch.tensor(orig_new_mask, dtype=torch.float32).cuda()  # Convert to PyTorch tensor and move to GPU
        old_masks = [torch.tensor(cell.masks[-1], dtype=torch.float32).cuda() for cell in
                     self.cells]  # Convert to PyTorch tensors and move to GPU

        for i, old_cell_mask in enumerate(old_masks):
            intersection = torch.logical_and(old_cell_mask, new_mask != 0)
            values, counts = torch.unique(new_mask[intersection], return_counts=True)
            if len(values) > 0:
                max_value = values[counts.argmax()]
                new_cell_mask = torch.where(new_mask == max_value, torch.tensor(1.0, dtype=torch.float32).cuda(),
                                            torch.tensor(0.0, dtype=torch.float32).cuda())
                self.cells[i].masks = np.vstack(
                    (self.cells[i].masks, new_cell_mask.cpu().numpy()[np.newaxis,:, :]))  # Convert back to NumPy array for storing
                new_mask = torch.where(new_mask == max_value, torch.tensor(0.0, dtype=torch.float32).cuda(), new_mask)
                self.cells[i].missing_count = 0
            else:
                if self.cells[i].missing_count < SETTINGS.TRACK_CLIP_LENGTH and not torch.logical_and(old_cell_mask,
                                                                                                      orig_new_mask > 0).any():
                    self.cells[i].masks = np.vstack(
                        (self.cells[i].masks, old_cell_mask.cpu().numpy()[np.newaxis, :, :]))  # Convert back to NumPy array for storing
                    self.cells[i].missing_count += 1
                else:
                    self.cells[i].masks = np.vstack(
                        (self.cells[i].masks, np.zeros((1, 1200, 1200))))  # Keep as NumPy array

        for new_cell_mask in mask_funcs.split_mask(new_mask, use_torch=True):
            if not torch.logical_and(new_cell_mask, self.last_frame() > 0).any():
                self.cells = np.append(self.cells, Cell(masks=torch.vstack((torch.zeros(
                    (len(self.cells[0].masks) - 1, 1200, 1200), dtype=torch.float32).cuda(),
                                                                            new_cell_mask.unsqueeze(0))),
                                                        index=self.max_cell_index() + 1, type=self.name))

        mask = torch.tensor([cell.missing_count < SETTINGS.TRACK_CLIP_LENGTH for cell in self.cells],
                            dtype=torch.bool).cuda()
        self.cells = self.cells[mask]

    def run_tracking(self):
        num_frames = len(self.mask_ims)
        times = np.array([])
        try:
            os.remove(SETTINGS.DIRECTORY / 'tracking' / (self.name+'_times.txt'))
        except:
            pass

        for i in range(num_frames):
            try:
                last_time = round(end_time - start_time, 1)
                times = np.append(times, last_time)
                average = round(np.average(times), 1)
            except:
                last_time = ' - '
                average =' - '
            sys.stdout.write(f'\rAdding frame {i + 1} / {num_frames}\t|\tLast frame took {last_time} s\t|\tAverage time per frame is {average} s')
            sys.stdout.flush()
            start_time = time.time()
            if i > 0:
                self.join_new_frame(i)
                with open(SETTINGS.DIRECTORY / 'tracking' / (self.name + '_times.txt'), 'a') as f:
                    f.write(str(last_time)+'\n')
            im = Image.fromarray((self.last_frame()).astype(np.uint16))
            im.save(SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(i)+'.tif'))
            if SETTINGS.VIEW_TRACKS:
                self.show_last_frame(i)
            end_time = time.time()



    def last_frame(self):
        frame = np.empty((1200, 1200))
        for cell in self.cells:
            frame += cell.masks[-1, :, :]*cell.index
        return frame

    def show_last_frame(self, index):
        frame = np.empty((1200, 1200, 3))
        for cell in self.cells:
            for i, col in enumerate(cell.color):
                frame[:, :, i] += (cell.masks[-1, :, :]*col)
        # plt.imshow(frame)
        # plt.show()
        im = Image.fromarray((frame*255).astype(np.uint8))
        im.save(SETTINGS.DIRECTORY / 'tracking_view' / self.name / ("{0:03}".format(index) + '.tif'))
        #im.save(Path('20x') / 'view_tracks' / self.name/("{0:03}".format(index)+'.tif'))

    def show_mask(self):
        plt.imshow(self.cells[5].masks[0])
        plt.show()

def main():
    amoeba_tracker = Tracker('Amoeba')
    amoeba_tracker.run_tracking()
# yeast_tracker = Tracker('yeast')
# yeast_tracker.run_tracking()

if __name__ == '__main__':
    main()
