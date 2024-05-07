import torch

import mask_funcs

class Cell:

    def __init__(self, index, mask, last_mask):
        self.index = index
        self.mask = mask
        self.last_mask = last_mask

    def speed(self):
        x_1, y_1 = mask_funcs.centre(self.mask)
        x_0, y_0 = mask_funcs.centre(self.last_mask)
        return ((x_1-x_0)**2 + (y_1-y_0)**2)**0.5

    def area(self):
        return torch.sum(self.mask)

    def circularity