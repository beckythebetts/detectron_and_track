import h5py
from pathlib import Path
import numpy as np

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
def main():
    #copy_features(Path('Datasets') / 'filter_test' / 'no_filter00.h5', Path('Datasets') / 'filter_test' / 'no_filter00feat.h5')
    #del_stuff('Datasets/filter_test/no_filter01.h5')
    copy_file(r'C:\Users\php23rjb\Downloads\temp\no_filter01_old.h5', r'C:\Users\php23rjb\Downloads\temp\no_filter01.h5')
if __name__ == '__main__':
    main()