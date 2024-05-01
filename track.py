import numpy as np
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import time
import sys
import torch
import gc
import cv2
import torch.nn.functional as F

from Cells import Cell
import mask_funcs
import SETTINGS
import utils

class Tracker:

    def __init__(self, name):
        self.name = name
        self.mask_ims = sorted([mask for mask in (SETTINGS.DIRECTORY / 'inference_dataset' / 'masks' / self.name).iterdir()])
        self.images = sorted([image for image in (SETTINGS.DIRECTORY / 'inference_dataset' / 'images').iterdir()])
        self.old_frame = torch.tensor(utils.read_tiff(self.mask_ims[0]).astype(np.int16)).cuda()
        self.new_frame = torch.tensor(utils.read_tiff(self.mask_ims[1]).astype(np.int16)).cuda()
        self.max_index = torch.max(self.old_frame)
        self.missing_cells = {} #key = index of missing cell, value = num frames missing for


    def update_new_frame(self):
        updated_new_frame = torch.zeros((1200, 1200)).cuda()
        for new_mask in mask_funcs.split_mask(self.new_frame, use_torch=True):
            intersection = torch.logical_and(new_mask, self.old_frame != 0)
            indexes, counts = torch.unique(self.old_frame[intersection], return_counts=True)
            if len(indexes) > 0 and torch.max(counts) > 0.5*torch.sum(new_mask):
                new_index = indexes[torch.argmax(counts)]
                self.old_frame = torch.where(self.old_frame==indexes[torch.argmax(counts)], 0, self.old_frame)
            else:
                new_index = self.max_index + 1
                self.max_index = new_index
            updated_new_frame += new_mask*int(new_index)
        old_mask_dict = mask_funcs.split_mask(self.old_frame, use_torch=True, return_indices=True)
        for key in old_mask_dict:
            if key in self.missing_cells:
                self.missing_cells[key] += 1
            else:
                self.missing_cells[key] = 0
            if self.missing_cells[key] < SETTINGS.FRAME_MEMORY:
                updated_new_frame = torch.where(old_mask_dict[key]==1, key, updated_new_frame)
            else:
                del self.missing_cells[key]

        self.new_frame = updated_new_frame

    def track(self):
        print('----------\nTRACKING\n----------')
        utils.remake_dir(SETTINGS.DIRECTORY / 'tracking' / self.name)
        # im = Image.fromarray(self.old_frame.cpu().numpy().astype(np.int16))
        # im.save(SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(0) + '.tif'))
        utils.save_tiff(self.old_frame.to(dtype=torch.int16).cpu().numpy().astype(np.uint16),
                        SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:04}".format(0) + '.tif'))

        for i in range(1, len(self.mask_ims)):
            sys.stdout.write(
                f'\rAdding frame {i+1} / {len(self.mask_ims)}')
            sys.stdout.flush()

            self.new_frame = torch.tensor(utils.read_tiff(self.mask_ims[i]).astype(np.int16)).cuda()
            self.update_new_frame()
            self.old_frame = self.new_frame
            utils.save_tiff(self.old_frame.to(dtype=torch.int16).cpu().numpy().astype(np.uint16), SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:04}".format(i) + '.tif'))

    def show_tracks(self):
        print('\n----------\nDISPLAYING\n----------')
        self.tracked_masks = sorted([mask for mask in (SETTINGS.DIRECTORY / 'tracking' / self.name).iterdir()])
        view_track_dir = SETTINGS.DIRECTORY / 'tracking' / (self.name+'_view')
        utils.remake_dir(view_track_dir)
        #total_num_cells = np.max(utils.read_tiff(self.tracked_masks[-1]))
        #colours = torch.tensor(np.random.uniform(0, 1, size=(total_num_cells+1, 3))).cuda()
        colour_dict = {}
        for i in range(len(self.tracked_masks)):
            sys.stdout.write(
                f'\rAdding frame {i + 1} / {len(self.mask_ims)}')
            sys.stdout.flush()
            mask = torch.tensor(utils.read_tiff(self.tracked_masks[i]).astype(np.int16)).cuda()
            image = utils.torch_min_max_scale(torch.tensor(utils.read_tiff(self.images[i]).astype(np.int16)).cuda())
            im_rgb = torch.stack((image, image, image), axis=0)
            #print(mask.shape)
            #split_mask = [torch.where(mask == i + 1, 1, 0) for i in range(0, torch.max(mask)) if i + 1 in mask]
            for j in range(torch.max(mask)):
                if j+1 in mask:
                    if j+1 not in colour_dict.keys():
                        colour_dict[j+1] = torch.tensor(np.random.uniform(0.3, 1, size=3)).cuda()
                    single_mask = torch.where(mask==j+1, 1, 0)
                    outline = mask_funcs.mask_outline(single_mask, 3)
                    for c in range(3):
                        im_rgb[c] = torch.where(outline, colour_dict[j+1][c], im_rgb[c])
            im_rgb = im_rgb.permute(1, 2, 0)
            Image.fromarray((im_rgb*255).cpu().numpy().astype(np.uint8)).save(view_track_dir / ("{0:04}".format(i) + '.jpg'))


def main():
    my_tracker = Tracker('Amoeba')
    if SETTINGS.TRACK:
        my_tracker.track()
    if SETTINGS.VIEW_TRACKS:
        my_tracker.show_tracks()


if __name__ == '__main__':
    main()