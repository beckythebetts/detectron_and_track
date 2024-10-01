from pathlib import Path
import numpy as np
import os
import shutil

class CellposeKfold:
    def __init__(self, directory):
        self.directory = directory
def move_im()
def split_datasets(directory):
    im_names = np.unique([name.stem[2:] for im in (directory / 'all').iterdir() if 'im' in im])
    for im_name in im_names:
        train_dir = directory / f'test_with_{im_name}' / 'train'
        val_dir = directory / f'test_with_{im_name}' / 'validate'
        train_dir.mkdir(parents=True)
        val_dir.mkdir()
        shutil.copy(directory / 'all' / ('im', im_name, '.png'), val_dir)
        shutil.copy(directory / 'all' / ('ma', im_name, '.png'), val_dir)

