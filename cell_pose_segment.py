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
        ims = ims = [f['Images']['Phase'][frame][:] for frame in f['Images']['Phase']]

    masks, flows, styles, diams = model.eval(image_arrays, diameter=30, flow_threshold=None, channels=channels)

    with h5py.File(hdf5_file, 'r+') as f:
        for mask in masks:
            f.create_dataset(f'Segmentations/Phase/{int(frame):04}', dtype='i2', data=class_mask_np)

def main():
    segment(SETTINGS.DATASET)

if __name__ == '__main__':
    main()