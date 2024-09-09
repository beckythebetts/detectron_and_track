import h5py
from pathlib import Path
import numpy as np

def copy_features(orig_dataset, new_dataset):
    with h5py.File(str(orig_dataset), 'r') as orig:
        data = orig['Features']
        with h5py.File(str(new_dataset), 'w') as new:
            orig.copy(data, new, 'Features')

def main():
    copy_features(Path('Datasets') / 'filter_test' / 'no_filter00.h5', Path('Datasets') / 'filter_test' / 'test.h5')

if __name__ == '__main__':
    main()