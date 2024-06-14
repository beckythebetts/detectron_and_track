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

import mask_funcs
import SETTINGS
import utils

def show_frame(image, mask, save_as):
    colour_dict = {}
    mask = torch.tensor(utils.read_tiff(mask).astype(np.uint8)).cpu()
    #image = utils.torch_min_max_scale(torch.tensor(utils.read_tiff(image)).cpu())
    image = torch.tensor(utils.read_tiff(image).astype(np.int16)).cpu()
    print(image.shape)
    # im_rgb = torch.stack((image, image, image), axis=0)
    im_rgb = image.permute(2, 0, 1)

    # print(mask.shape)
    # split_mask = [torch.where(mask == i + 1, 1, 0) for i in range(0, torch.max(mask)) if i + 1 in mask]
    for j in range(torch.max(mask)):
        if j + 1 in mask:
            if j + 1 not in colour_dict.keys():
                colour_dict[j + 1] = torch.tensor(np.random.uniform(0, 2 ** (8) - 1, size=3)).cpu()
            single_mask = torch.where(mask == j + 1, 1, 0)
            outline = mask_funcs.mask_outline(single_mask, 3)
            for c in range(3):
                print(outline.shape, colour_dict[j + 1][c], im_rgb[c].shape, im_rgb.shape)
                im_rgb[c] = torch.where(outline, colour_dict[j + 1][c], im_rgb[c])
    im_rgb = im_rgb.permute(1, 2, 0)
    # Image.fromarray((im_rgb*(2**16-1)).cpu().numpy().astype(np.uint16)).save(view_track_dir / ("{0:04}".format(i) + '.jpg'))
    # print(im_rgb.shape)
    utils.save_tiff((im_rgb).cpu().numpy().astype(np.uint8), save_as)

def main():
    show_frame(Path('ims_for_report/0.22/00.jpg'), Path('ims_for_report/0.22/00_mask.tif'), Path('ims_for_report/0.22/view.jpg'))

if __name__ == '__main__':
    main()