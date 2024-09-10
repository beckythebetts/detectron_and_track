import h5py
from pathlib import Path
import numpy as np

def copy_features(orig_dataset, new_dataset):
    with h5py.File(str(orig_dataset), 'r') as orig:
        data = orig['Features']
        with h5py.File(str(new_dataset), 'w') as new:
            orig.copy(data, new, 'Features')

def main():
    copy_features(Path('Datasets') / 'danhighres' / 'dan10.h5', Path('Datasets') / 'danhighres' / 'dan10features.h5')

if __name__ == '__main__':
    main()