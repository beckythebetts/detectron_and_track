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
        self.index = 1
        self.old_frame = torch.tensor(plt.imread(self.mask_ims[self.index - 1]).astype(np.int16)).cuda()
        self.new_frame = torch.tensor(plt.imread(self.mask_ims[self.index]).astype(np.int16)).cuda()