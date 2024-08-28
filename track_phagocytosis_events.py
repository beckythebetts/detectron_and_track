import numpy as np
import torch
import h5py

import SETTINGS
import utils

# class PhagocyticEvent:
#     def __init__(self, frames):
#         self.frames = start_frame
#         self.pathogen_indices = self.get_pathogen_indices()
#
#     def get_pathogen_indices(self):
#
#         self.pathogen_indices = [torch.unique(array) for array in
#                                  torch.where(intersection, self.epi_mask.unsqueeze(0), 0)]
#         self.pathogen_indices = [a[a.nonzero()] for a in self.pathogen_indices]
#
#
# class PhagocyticCell:
#     def __init__(self, cell_name, consecutive_eaten_frames):
#         self.cell_name = cell_name
#         self.cell_index = int(cell_name[-4:])
#         self.consecutive_eaten_frames = consecutive_eaten_frames
#         for sequence in consecutive_eaten_frames:
#             if len(sequence) >= SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
#                 phagocytic_event = PhagocyticEvent()
#
#     def track_internal_pathogens(self):



def track_phagocytic_events(hdf5file):
    with h5py.File(hdf5file, 'r') as f:
        for cell in f['Features']:
            phago_events = f['Features'][cell]['PhagocyticFrames'][:]
            for event in phago_events:
                print(event)




def main():
    hdf5file = SETTINGS.DATASET
    track_phagocytic_events(hdf5file)

if __name__ == '__main__':
    main()

