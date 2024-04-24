import numpy as np
import random
import matplotlib.pyplot as plt
import os
from pathlib import Path
from skimage import measure
import mask_funcs
import SETTINGS

class Cell:

    def __init__(self, masks, index, type):
        self.masks = masks
        self.index = index
        self.type = type
        seed = int.from_bytes(os.urandom(4), byteorder='big')
        random.seed(seed)
        self.color = [random.random(), random.random(), random.random()]
        self.missing_count = 0
        #self.file = Path(r'20x') / str('info_'+type) / (str(index)+'.txt')
        # self.file = SETTINGS.DIRECTORY / 'tracking' / str('info_'+self.type) / (str(index)+'.txt')
        # with open(self.file, 'w') as f:
        #     f.write('dist_moved\tarea\tcircularity\toverlap\tdist_nearest\tindex_nearest')

    def clip_track(self):
        self.masks = self.masks[-SETTINGS.TRACK_CLIP_LENGTH:, :, :]

    def write_features(self, time=None):
        time = len(self.masks) if time is None else time
        dist_nearest, index_nearest = self.nearest(time)
        new_row = '\n' + '\t'.join([str(self.dist_moved(time)), str(self.area(time)), str(self.circularity(time)),
                                    str(self.overlap(time)), str(dist_nearest), str(index_nearest)])
        with open(self.file, 'a') as f:
            f.write(new_row)

    def centre(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan
        else:
            return measure.centroid(self.masks[time])

    def dist_moved(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan
        else:
            if time < 1:
                return 0
            else:
                return np.linalg.norm(self.centre(time) - self.centre(time-1))

    def area(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan
        else:
            return np.sum(self.masks[time])

    def circularity(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan
        else:
            if measure.perimeter(self.masks[time])==0:
                return 0
            else:
                return 4*np.pi*self.area(time) / (measure.perimeter(self.masks[time])**2)

    def overlap(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]) or not np.any(self.masks[time]):
            return np.nan
        else:
            iou = mask_funcs.cal_iou(self.masks[time], self.masks[time-1])
            if iou == 0:
                return np.nan
            else:
                return iou

    def nearest(self, time=None, type='yeast'):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan, np.nan

        else:
            frame_to_search = plt.imread(Path(type) / ''.join([str(time), '.tif']))
            distance = 0
            index_of_nearest = None
            while index_of_nearest is None:
                circle_mask = mask_funcs.create_circle(self.centre(time)[::-1], distance)
                search_mask = np.where(frame_to_search > 0, 1, 0)
                intersection = np.logical_and(circle_mask, search_mask)
                if not np.any(intersection):
                    distance += 1
                else:
                    unique_values, counts = np.unique(frame_to_search[intersection], return_counts=True)
                    index_of_nearest = unique_values[np.argmax(counts)]
            return distance, index_of_nearest
