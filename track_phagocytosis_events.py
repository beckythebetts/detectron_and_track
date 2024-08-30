import numpy as np
import torch
import h5py
import sys

import SETTINGS
import utils

class PhagocyticEvent:

    def __init__(self, frames, pathogen_indices):
        self.frames = frames
        self.pathogen_indices = pathogen_indices

    def add_frame(self, frame, index):
        self.frames = np.append(self.frames, frame)
        self.pathogen_indices = np.append(self.pathogen_indices, index)

    def save_event(self, cell):



def track_phagocytic_events(hdf5file):
    with h5py.File(hdf5file, 'r') as f:
        for cell in f['Features']:
            sys.stdout.write(f'\r{cell}')
            sys.stdout.flush()
            phago_events = f['Features'][cell]['PhagocyticFrames'][:]
            frames = f['Features'][cell]['PhagocyticFrames']['frame'][:]
            if len(frames) > SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
                sequences = utils.split_list_into_sequences(frames)
                for sequence in sequences:
                    # if sequence of frames contains duplicate values => multiple pathogens are observed simultaneously => need tracking
                    if len(sequence) == len(set(sequence)):
                        # if only one pathogen observed, no need to track
                        indices = [event[1] for event in phago_events if event[0] in sequence]
                        phago_event = PhagocyticEvent(sequence, indices)
                    # else:
                    #     # this phagocytic event involves multiple pathogens, so each must be tracked individually
                    #     for frame in np.unique(sequence):







def main():
    hdf5file = SETTINGS.DATASET
    track_phagocytic_events(hdf5file)

if __name__ == '__main__':
    main()

