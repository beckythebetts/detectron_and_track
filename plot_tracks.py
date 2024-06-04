import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

import SETTINGS
import utils
import mask_funcs

def plot_tracks(dir):
    track_plot = torch.zeros(*SETTINGS.IMAGE_SIZE, 3)
    colours_dict = {}
    for file in sorted([i for i in dir.iterdir()]):
        frame = torch.tensor(utils.read_tiff(file).astype(np.int32)).cuda()
        for i in torch.unique(frame):
            centre = mask_funcs.find_centre(torch.where(frame==i, 1, 0))
            if i not in colours_dict:
                colours_dict[i] = torch.tensor(np.random.uniform(0, 2**(8)-1, size=3)).cuda()
            track_plot[centre, :] = colours_dict[i]
    utils.save_tiff(track_plot, dir.parents[2] / 'track_plot.png')


def main():
    dir = SETTINGS.DIRECTORY / 'tracked' / 'phase'
    plot_tracks(dir)

if __name__ == '__main__':
    main()
