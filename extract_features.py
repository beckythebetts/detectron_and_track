import torch
from skimage import measure
import numpy as np

import mask_funcs

class Cell:

    def __init__(self, index, mask, last_mask):
        self.index = index
        self.mask = mask
        self.last_mask = last_mask

    def centre(self):
        self.centre = mask_funcs.centre(self.mask)
        return self.centre

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
            circle_mask = mask_funcs.torch_circle(self.centre, 0)
            intersection = torch.logical_and(circle_mask, other_frame)
            unique_values, counts = torch.unqiue(other_frame[intersection], return_counts=True)
            if len(indexes) > 0:
                index_of_nearest = unique_values[torch.argmax(counts)]
            else:
                dist += 1
        return dist, index_of_nearest