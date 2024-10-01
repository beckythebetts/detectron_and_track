import sys

from cellpose import models, core, plot, io
from cellpose.io import logger_setup
import h5py
import numpy as np

import SETTINGS
import threshold_epi

def segment(hdf5_file):
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d' % use_GPU)
    #logger_setup()

    model = models.Cellpose(gpu=use_GPU, model_type='cyto3')
    channels = [0, 0]
    with h5py.File(hdf5_file, 'r+') as f:
        if 'Segmentations' in f:
            del f['Segmentations']
        f['Segmentations']['Phase'].attrs['Model'] = str(SETTINGS.CELLPOSE_MODEL)
        threshold_epi.main()
        batchsize = 2 # batchsize for saving

        num_batches, remainder = divmod(SETTINGS.NUM_FRAMES, batchsize)
        batches = [np.arange(i * batchsize, min(((i + 1) * batchsize), SETTINGS.NUM_FRAMES)) for i in
                   range(0, num_batches + 1)]

        for i, batch in enumerate(batches):
            print(f'\nSegmenting Batch {i+1} / {num_batches}')
            ims = [f['Images']['Phase'][f'{int(frame):04}'][:] for frame in batch]

            masks, flows, styles, diams = model.eval(ims, diameter=0, flow_threshold=0.2, channels=channels)

            print(f'\nSaving Batch {i+1} / {num_batches}')
            for j, (mask, frame) in enumerate(zip(masks, batch)):
                sys.stdout.write(f'\r{j+1} / {batchsize}')
                sys.stdout.flush()

                # #NUMPY
                # epi_mask = f['Images']['Epi'][f'{int(frame):04}'][:]
                # for idx in np.unique(mask):
                #     if idx != 0:
                #         maski = np.where(mask==idx, 1, 0)
                #         if np.sum(np.logical_and(maski, epi_mask)) > np.sum(maski):
                #             mask = np.where(mask==idx, 0, mask)
                #
                # #TORCH
                # epi_mask = torch.tensor(f['Images']['Epi'][f'{int(frame):04}'][:]).to(device)
                # mask = torch.tensor(mask).to(device)


                f.create_dataset(f'Segmentations/Phase/{int(frame):04}', dtype='i2', data=mask)

def main():
    segment(SETTINGS.DATASET)

if __name__ == '__main__':
    main()