import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import utils
from PIL import Image
import os
import sys

def read_hdf5():
    with h5py.File('RAW_DATA/04/Phase.h5', 'r') as f:
        print(list(f['t00000']['s00']['0'].keys()))
        first_image = f['t00000']['s00']['0']['cells']
        print(f['t00000']['s00']['0'].attrs['cells'])
        image = np.array(first_image)
        image = np.squeeze(image)
        plt.imshow(image, cmap='gray')
        plt.show()

def create_hdf5(hdf5_filename, phase_tiffs_path, epi_tiffs_path):
    if os.path.exists(hdf5_filename):
        os.remove(hdf5_filename)
    with h5py.File(hdf5_filename, 'w') as f:
        Images = f.create_group('Images')
        # ****** Image MetaData ********
        Images.attrs['Camera'] = 'Andor Sona 4BV11'
        Images.attrs['Pixel Size / um'] = 11
        Images.attrs['Objective magnification'] = 30
        Images.attrs['Image size / pixels'] = [2048, 2048]
        Images.attrs['Phase exposure / ms'] = 2000
        Images.attrs['Epi exposure / ms'] = 10
        Images.attrs['Time interval / s'] = 3
        Images.attrs['Number of frames'] = 1200
        Images.attrs['Filter'] = False

        Images.attrs['Resolution / um'] = Images.attrs['Pixel Size / um'] / Images.attrs['Objective magnification']
        Images.attrs['FOV / um'] = Images.attrs['Image size / pixels'] * Images.attrs['Resolution / um']

        Phase = Images.create_group('Phase')
        Epi = Images.create_group('Epi')
        print('\nConverting Phase Images\n')
        i=1
        for im in Path(phase_tiffs_path).iterdir():
            sys.stdout.write(f'\rFrame {i} / {Images.attrs["Number of frames"]}')
            sys.stdout.flush()
            Phase.create_dataset(im.stem, data=np.array(Image.open(im, mode='r')))
            i += 1
        print('\nConverting Epi Images\n')
        i = 1
        for im in Path(epi_tiffs_path).iterdir():
            sys.stdout.write(
                f'\rFrame {i} / {Images.attrs["Number of frames"]}')
            sys.stdout.flush()
            Epi.create_dataset(im.stem, data=np.array(Image.open(im, mode='r')))
            i += 1





def main():
    create_hdf5(r'H:\rebecca\filter_test\no_filter\no_filter01.h5', r'H:\rebecca\filter_test\no_filter\01\phase', r'H:\rebecca\filter_test\no_filter\01\epi')

if __name__=='__main__':
    main()