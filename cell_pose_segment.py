from cellpose import models, core, plot, io
from cellpose.io import logger_setup
import h5py

import SETTINGS

def segment(hdf5_file):
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d' % use_GPU)
    logger_setup()

    model = models.Cellpose(gpu=use_GPU, model_type='cyto3')
    channels = [0, 0]
    with h5py.File(hdf5_file, 'r+') as f:
        if 'Segmentations' in f:
            del f['Segmentations']

        batchsize = 10 # batchsize for saving

        num_batches, remainder = divmod(SETTINGS.NUM_FRAMES, batchsize)
        batches = [np.arange(i * batchsize + 1, min(((i + 1) * batchsize) + 1, SETTINGS.NUM_FRAMES)) for i in
                   range(0, num_batches + 1)]
        for batch in batches:
            ims = [f['Images']['Phase'][f'{int(frame):04}'][:] for frame in batch]

            masks, flows, styles, diams = model.eval(ims, diameter=30, flow_threshold=None, channels=channels)

            for mask, frame in zip(masks, batch):
                f.create_dataset(f'Segmentations/Phase/{int(frame):04}', dtype='i2', data=masks)

def main():
    segment(SETTINGS.DATASET)

if __name__ == '__main__':
    main()