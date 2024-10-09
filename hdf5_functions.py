import h5py
from pathlib import Path
import numpy as np

def rename_datasets(file):
    with h5py.File(file, 'r+') as f:
        for frame in f['Images']['Epi'].keys():
            if len(frame) > 4:
                del f['Images']['Epi'][frame]
def copy_features(orig_dataset, new_dataset):
    with h5py.File(str(orig_dataset), 'r') as orig:
        data = orig['Features']
        with h5py.File(str(new_dataset), 'w') as new:
            orig.copy(data, new, 'Features')
def del_stuff(dataset):
    with h5py.File(str(dataset), 'r+') as f:
        del f['Segmentations']
        del f['Features']

def copy_file(input_file, output_file):
    #like hdf5 repack
    with h5py.File(input_file, 'r') as in_:
        with h5py.File(output_file, 'w') as out_:
            def recursively_copy_group(in_, out_):
                for name, item in in_.items():
                    if isinstance(item, h5py.Dataset):
                        # Apply compression, shuffle, and chunking
                        out_.create_dataset(
                            name,
                            data=item[...]  # Read the data into memory
                            # compression='gzip',  # Apply gzip compression
                            # compression_opts=9,  # Maximum compression level (1-9)
                            # shuffle=True,  # Apply shuffle filter
                            # chunks=True  # Enable chunking
                        )
                    elif isinstance(item, h5py.Group):
                        # Recursively copy groups
                        subgroup = out_.create_group(name)
                        recursively_copy_group(item, subgroup)
            recursively_copy_group(in_, out_)

def set_image_attributes(file):
    with h5py.File(file, 'r+') as f:
        Images = f['Images']
        Images.attrs['Camera'] = 'Andor sCMOS'
        Images.attrs['Pixel Size / um'] = 11
        Images.attrs['Objective magnification'] = 30
        Images.attrs['Image size / pixels'] = [2048, 2048]
        Images.attrs['Phase exposure / ms'] = 100
        Images.attrs['Epi exposure / ms'] = 10
        Images.attrs['Time interval / s'] = 3
        Images.attrs['Number of frames'] = 1200
        Images.attrs['Filter'] = False
        Images.attrs['NA'] = 0.5

        Images.attrs['Resolution / um'] = Images.attrs['Pixel Size / um'] / Images.attrs['Objective magnification']
        Images.attrs['FOV / um'] = Images.attrs['Image size / pixels'] * Images.attrs['Resolution / um']
def main():
    #rename_datasets('Datasets/danhighres/dan3.h5')
    set_image_attributes('Datasets/filter_test/no_filter01.h5')
if __name__ == '__main__':
    main()