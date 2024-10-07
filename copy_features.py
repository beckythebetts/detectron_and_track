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

def main():
    #copy_features(Path('Datasets') / 'filter_test' / 'no_filter00.h5', Path('Datasets') / 'filter_test' / 'no_filter00feat.h5')
    del_stuff('Datasets/filter_test/no_filter01.h5')
if __name__ == '__main__':
    main()