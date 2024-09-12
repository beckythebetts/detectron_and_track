import numpy as np
import torch
import h5py
import sys
from pathlib import Path
import imageio

import SETTINGS
import utils
import mask_funcs
from nearest_neighbour_tracking import Track, NearestNeighbourTracking

class PhagocyticEvent:

    def __init__(self, frames, pathogen_indices):
        self.frames = np.array(frames)
        self.pathogen_indices = np.array(pathogen_indices)

    def add_frame(self, frame, index):
        self.frames = np.append(self.frames, frame)
        self.pathogen_indices = np.append(self.pathogen_indices, index)

    def save_event(self, cell):
        with h5py.File(SETTINGS.DATASET, 'r+') as f:
            dtype = np.dtype([('frame', 'f4'), ('pathogen_index', 'f4')])
            data = np.zeros(self.frames.size, dtype=dtype)
            data['frame'] = self.frames
            data['pathogen_index'] = self.pathogen_indices
            try:
                f.create_dataset(f'Features/{cell}/{int(self.frames[0])}_{int(self.frames[-1])}', data=data)
            except ValueError:
                f.create_dataset(f'Features/{cell}/{int(self.frames[0])}_{int(self.frames[-1])}_1', data=data)
            if len(self.frames) != len(np.unique(self.frames)):
                print('PROBLEM FOUND', self.frames, self.pathogen_indices)

def track_phagocytosis_events(hdf5file):
    with h5py.File(hdf5file, 'r') as f:
        for cell in f['Features']:
            sys.stdout.write(f'\r{cell}')
            sys.stdout.flush()
            phago_events = f['Features'][cell]['PhagocyticFrames']
            frames = phago_events['frame'][:]
            pathogen_indices = phago_events['pathogen_index'][:]
            if len(frames) > SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
                sequences = utils.split_list_into_sequences(frames, return_indices=True)
                for sequence in sequences:
                    if len(sequence) > SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
                        # get pathogen_centres form epi masks:
                        centres = [mask_funcs.get_centre(np.where(f['Segmentations']['Epi'][f'{int(frame):04}'][:]==index, 1, 0)) for frame, index in zip(frames[sequence], pathogen_indices[sequence])]
                        # NN tracker
                        tracker = NearestNeighbourTracking(frames[sequence], pathogen_indices[sequence], centres)
                        tracker.track()
                        for track in tracker.tracked:
                            event = PhagocyticEvent(track.track_dict.keys(), track.track_dict.values())
                            event.save_event(cell)

# def track_phagocytic_events_2(hdf5file):
#     with h5py.File(hdf5file, 'r+') as f:
#         for cell in f['Features']:
#             sys.stdout.write(f'\r{cell}')
#             sys.stdout.flush()
#             phago_events = f['Features'][cell]['PhagocyticFrames']
#             frames = phago_events['frame'][:]
#             pathogen_indices = phago_events['pathogen_index'][:]
#             #print(phago_events['pathogen_index'])
#             if len(frames) > SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
#                 sequences = utils.split_list_into_sequences(frames)
#                 for sequence in sequences:
#                     if len(sequence) > SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
#                         # if sequence of frames contains duplicate values => multiple pathogens are observed simultaneously => need tracking
#                         if len(sequence) == len(np.unique(sequence)):
#                             # if only one pathogen observed, no need to track
#                             indices = [event[1] for event in phago_events[:] if event[0] in sequence]
#                             #print(indices)
#                             phago_event = PhagocyticEvent(sequence, indices)
#                             phago_event.save_event(cell)
#                         else:
#                             # this phagocytic event involves multiple pathogens, so each must be tracked individually
#                             #print('FOUND ', cell, sequence[0] )
#                             phagocytosis_events = []
#                             for i, frame in enumerate(np.unique(sequence)):
#                                 epi_mask = f['Segmentations']['Epi'][f'{int(frame):04}'][:]
#                                 if i == 0:
#                                     new_indices = pathogen_indices[np.argwhere(frames==frame)]
#                                     new_centres = np.array([mask_funcs.get_centre(np.where(epi_mask == index, 1, 0)) for index in new_indices])
#                                     for new_index in new_indices:
#                                         phagocytosis_events.append(PhagocyticEvent([frame], new_index))
#                                 else:
#                                     old_indices = new_indices
#                                     old_centres = new_centres
#                                     new_indices = pathogen_indices[np.argwhere(frames==frame)]
#                                     new_centres = np.array([mask_funcs.get_centre(np.where(epi_mask == index, 1, 0)) for index in new_indices])
#                                     #calculate distances between all combos
#                                     #distances = mask_funcs.dist_between_points(old_centres, new_centres[:, np.newaxis])
#                                     distances = np.linalg.norm(old_centres[:, np.newaxis]-new_centres[np.newaxis,], axis=2)
#                                     #print(old_indices, new_indices)
#                                     if len(old_indices) >= len(new_indices):
#                                         old_old_indices = old_indices.copy()
#                                         for j, new_index in enumerate(new_indices):
#                                             old_index = old_indices[np.argmin(distances[:, j])]
#
#                                             distances = np.delete(distances, np.argmin(distances[:, j]), axis=0)
#                                             old_old_indices = np.delete(old_old_indices, np.argmin(distances[:, j]))
#                                             for phagocytosis_event in phagocytosis_events:
#                                                 if phagocytosis_event.pathogen_indices[-1] == old_index:
#                                                     phagocytosis_event.add_frame(frame, new_index)
#                                         if len(old_old_indices) > 0:
#                                             for old_old_index in old_indices:
#
#                                     else:
#                                         new_new_indices = new_indices.copy()
#                                         for j, old_index in enumerate(old_indices):
#                                             new_index = new_indices[np.argmin(distances[j])]
#                                             print('*', distances)
#                                             distances = np.delete(distances, np.argmin(distances[j]), axis=1)
#                                             print(distances)
#                                             new_new_indices = np.delete(new_new_indices, np.argmin(distances[j]))
#                                             for phagocytosis_event in phagocytosis_events:
#                                                 if phagocytosis_event.pathogen_indices[-1] == old_index:
#                                                     phagocytosis_event.add_frame(frame, new_index)
#                                         for new_new_index in new_new_indices:
#                                             phagocytosis_events.append(PhagocyticEvent([frame], new_index))
#                             for phagocytosis_event in phagocytosis_events:
#                                 phagocytosis_event.save_event(cell)
#
#
#                             #print(cell, frame, phago_events['pathogen_index'][np.argwhere(frames==frame)])


def del_events(dataset):
    with h5py.File(dataset, 'r+') as f:
        for cell in f['Features'].keys():
            for phago_event in f['Features'][cell]:
                #print(phago_event)
                if phago_event != 'MorphologicalFeatures' and phago_event != 'PhagocyticFrames':
                    del(f['Features'][cell][phago_event])

def show_phagocytic_events(dataset, save_directory):
    with h5py.File(dataset, 'r') as f:
        for cell in f['Features'].keys():
            sys.stdout.write(f'\r{cell}')
            sys.stdout.flush()
            for phago_event in f['Features'][cell]:
                #print(phago_event)
                if phago_event != 'MorphologicalFeatures' and phago_event != 'PhagocyticFrames':
                    # del(f['Features'][cell][phago_event])
                    (Path(save_directory) / cell / phago_event).mkdir(parents=True)
                    frames = f['Features'][cell][phago_event]['frame'][:]
                    for frame in np.arange(np.max((min(frames)-10, 0)), np.min((max(frames)+11, SETTINGS.NUM_FRAMES))):
                        phase_image = np.array(f['Images']['Phase'][f'{int(frame):04}'])
                        phase_mask = np.array(f['Segmentations']['Phase'][f'{int(frame):04}'])
                        outline = mask_funcs.mask_outline(torch.where(torch.tensor(phase_mask) == int(cell[-4:]), 1, 0), thickness=2).cpu().numpy()
                        epi_image = np.array(f['Images']['Epi'][f'{int(frame):04}'])
                        if frame in frames:
                            pathogen_index = f['Features'][cell][phago_event]['pathogen_index'][np.argwhere(frames==frame)]
                            print(pathogen_index)
                            epi_mask = np.where(f['Segmentations']['Epi'][f'{int(frame):04}'][:]==pathogen_index, 1, 0)
                        else:
                            epi_mask = np.zeros(phase_image.shape)
                        im_rgb = np.stack((phase_image, phase_image, phase_image), axis=0)

                        im_rgb[0] = np.where(outline, 255, im_rgb[0])
                        im_rgb[1] = np.where(outline, 255, im_rgb[1])
                        im_rgb[2] = np.where(outline, 0, im_rgb[2])

                        im_rgb[0] = np.where(epi_mask, epi_image, im_rgb[0])

                        im_rgb = im_rgb.transpose(1, 2, 0)

                        imageio.imwrite(Path(save_directory) / cell / phago_event / '{0:04}.jpg'.format(frame), im_rgb.astype(np.uint8))



def main():
    hdf5file = SETTINGS.DATASET
    del_events(hdf5file)
    track_phagocytosis_events(hdf5file)
    show_phagocytic_events(hdf5file, 'Datasets/filter_test/no_filter00_showeatingNEW')

if __name__ == '__main__':
    main()

