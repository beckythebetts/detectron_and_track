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
        first_image = f['t00000']['s00']['0']['cells']
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
        Images.attrs['Camera'] = 'Daniels, higher reoslution one???? CHECK VALUES'
        Images.attrs['Pixel Size / um'] = 1
        Images.attrs['Objective magnification'] = 1
        Images.attrs['Image size / pixels'] = [1004,1002]
        Images.attrs['Phase exposure / ms'] = 1
        Images.attrs['Epi exposure / ms'] = 1
        Images.attrs['Time interval / s'] = 10
        Images.attrs['Number of frames'] = 541
        Images.attrs['Filter'] = False
        Images.attrs['NA'] = 1

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
    create_hdf5(r'Datasets\dan10.h5', r'C:\Users\php23rjb\Documents\detectron_and_track\RAW_DATA\Daniels\Higher_res\10sec_interval\Phase', r'C:\Users\php23rjb\Documents\detectron_and_track\RAW_DATA\Daniels\Higher_res\10sec_interval\Epi')

if __name__=='__main__':
    main()