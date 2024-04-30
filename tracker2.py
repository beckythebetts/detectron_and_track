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


    def update_new_frame(self):
        # print(torch.max(self.new_frame))
        # print(self.new_frame)
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
        print('----------\nTRACKING\n----------')
        utils.remake_dir(SETTINGS.DIRECTORY / 'tracking' / self.name)
        #cv2.imwrite(str(SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(0) + '.tif')), self.old_frame.cpu().numpy().astype(np.int16))
        im = Image.fromarray(self.old_frame.cpu().numpy().astype(np.int16))
        im.save(SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(0) + '.tif'))
        for i in range(1, len(self.mask_ims)):
            sys.stdout.write(
                f'\rAdding frame {i+1} / {len(self.mask_ims)}')
            sys.stdout.flush()

            self.new_frame = torch.tensor(utils.read_tiff(self.mask_ims[i]).astype(np.int16)).cuda()
            self.update_new_frame()
            self.old_frame = self.new_frame
            # cv2.imwrite(str(SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(i) + '.tif')),
            #             self.old_frame.cpu().numpy().astype(np.int16))
            # im = Image.fromarray(self.old_frame.cpu().numpy().astype(np.int16))
            # print(np.shape(im))
            # im.save(SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(i) + '.tif'))
            utils.save_tiff(self.old_frame.cpu().numpy().astype(np.uint16), SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(i) + '.tif'))

    def show_tracks(self):
        print('\n----------\nDISPLAYING\n----------')
        self.tracked_masks = sorted([mask for mask in (SETTINGS.DIRECTORY / 'tracking' / self.name).iterdir()])
        view_track_dir = SETTINGS.DIRECTORY / 'tracking' / (self.name+'_view')
        utils.remake_dir(view_track_dir)
        total_num_cells = np.max(utils.read_tiff(self.tracked_masks[-1]))
        colours = torch.tensor(np.random.uniform(0, 1, size=(total_num_cells+1, 3))).cuda()
        for i in range(len(self.tracked_masks)):
            sys.stdout.write(
                f'\rAdding frame {i + 1} / {len(self.mask_ims)}')
            sys.stdout.flush()
            mask = torch.tensor(utils.read_tiff(self.tracked_masks[i]).astype(np.int16)).cuda()
            image = utils.torch_min_max_scale(torch.tensor(utils.read_tiff(self.images[i]).astype(np.int16)).cuda())
            im_rgb = torch.stack((image, image, image), axis=0)
            #print(mask.shape)
            #split_mask = [torch.where(mask == i + 1, 1, 0) for i in range(0, torch.max(mask)) if i + 1 in mask]
            for j in range(torch.max(mask)+1):
                if j+1 in mask:
                    single_mask = torch.where(mask==j+1, 1, 0)
                    #print(single_mask.shape)
                    expanded_mask = F.max_pool2d(single_mask.float().unsqueeze(1), kernel_size=3, stride=1, padding=1) > 0
                    #print(expanded_mask.shape)
                    outline = (expanded_mask.byte().squeeze() - single_mask).bool()
                    #print(outline.shape)
                    for c in range(3):
                        im_rgb[c] = torch.where(outline, colours[j, c], im_rgb[c])
            im_rgb = im_rgb.permute(1, 2, 0)
            Image.fromarray((im_rgb*255).cpu().numpy().astype(np.uint8)).save(view_track_dir / (str(i)+'.jpg'))


def main():
    my_tracker = Tracker('Amoeba')
    if SETTINGS.TRACK:
        my_tracker.track()
    if SETTINGS.VIEW_TRACKS:
        my_tracker.show_tracks()
    # test = cv2.imread('03/000.tif')
    # print(np.shape(test))
if __name__ == '__main__':
    main()