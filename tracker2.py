import numpy as np
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import time
import sys
import torch
import gc

from Cells import Cell
import mask_funcs
import SETTINGS
import utils

class Tracker:

    def __init__(self, name):
        self.name = name
        self.mask_ims = sorted([mask for mask in (SETTINGS.DIRECTORY / 'inference_dataset' / 'masks' / self.name).iterdir()])
        #self.index = 1
        self.old_frame = torch.tensor(plt.imread(self.mask_ims[0]).astype(np.int16)).cuda()
        self.new_frame = torch.tensor(plt.imread(self.mask_ims[1]).astype(np.int16)).cuda()
        utils.remake_dir(SETTINGS.DIRECTORY / 'tracking' / self.name)

    def update_new_frame(self):
        updated_new_frame = torch.zeros((1200, 1200)).cuda()
        highest_index = torch.max(self.old_frame)
        for new_mask in mask_funcs.split_mask(self.new_frame, use_torch=True):
            intersection = torch.logical_and(new_mask, self.old_frame != 0)
            indexes, counts = torch.unique(self.old_frame[intersection], return_counts=True)
            if len(indexes) > 0 and torch.max(counts) > 0.5*torch.sum(new_mask):
                new_index = indexes[torch.argmax(counts)]
            else:
                new_index = highest_index + 1
                highest_index = new_index

            updated_new_frame += new_mask*new_index
        self.new_frame = updated_new_frame

    def track(self):

        for i in range(len(self.mask_ims)):
            try:
                last_time = round(end_time - start_time, 1)
                times = np.append(times, last_time)
                average = round(np.average(times), 1)
            except:
                last_time = ' - '
                average = ' - '
            sys.stdout.write(
                f'\rAdding frame {i + 1} / {len(self.mask_ims)}\t|\tLast frame took {last_time} s\t|\tAverage time per frame is {average} s')
            sys.stdout.flush()
            start_time = time.time()


            self.new_frame = torch.tensor(plt.imread(self.mask_ims[i+1]).astype(np.int16)).cuda()
            im = Image.fromarray(self.old_frame.cpu().numpy().astype(np.int16))
            im.save(SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(i) + '.tif'))
            self.update_new_frame()
            self.old_frame = self.new_frame


            end_time = time.time()

def main():
    my_tracker = Tracker('Amoeba')
    my_tracker.track()

if __name__ == '__main__':
    main()