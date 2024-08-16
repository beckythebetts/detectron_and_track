import h5py
import imagej
import numpy as np


#from jnius import autoclass

hdf5_file = 'Datasets/04_short.h5'
#ij = imagej.init('sc.fiji:fiji:2.1.0', mode='gui')
ij = imagej.init('2.1.0', mode='interactive')

def make_rgb(greyscale_im):
    return np.stack((greyscale_im, greyscale_im, greyscale_im), axis=1)[:, np.newaxis]

def show_separate_channels():

    with h5py.File(hdf5_file, 'r') as f:
        print('Gathering data')
        data = np.array([
            [
                f['Images'][channel][frame][:]
                for channel in f['Images'].keys()
            ]
            for frame in f['Images']['Phase'].keys()
        ], dtype='uint8')
        print('converting data')
        ij_image = ij.py.to_java(data)
        print('opening images')
        #ij.io().open(ij_image)
        ij.ui().show('images', ij_image)
    input("Close ImageJ [ENTER]")

def show_merged_channels_2():
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = np.array([f['Images']['Phase'][frame][:]
                               for frame in f['Images']['Phase'].keys()], dtype='uint8')
        epi_data = np.array([f['Images']['Epi'][frame][:]
                               for frame in f['Images']['Epi'].keys()], dtype='uint8')

        epi_channel = make_rgb(epi_data)
        epi_channel[:,:,1:3] = 0
        merged_data = ((make_rgb(phase_data).astype(np.float32) + epi_channel.astype(np.float32)) / (2)).astype(np.uint8)
        print(merged_data.shape)
        merged_image = ij.py.to_dataset(merged_data, dim_order=['time', 'z', 'ch', 'row', 'col'])
        ij.ui().show(merged_image)

    input("Close ImageJ [ENTER]")
    #return merged_data

def show_merged_channels():
    with h5py.File(hdf5_file, 'r') as f:
        frames = list(f['Images']['Phase'].keys())
        merged_images = []
        for t in frames:
            phase_image = f['Images']['Phase'][t][:][np.newaxis,:]
            epi_image = f['Images']['Epi'][t][:][np.newaxis,:]

            Phase = ij.py.to_java(phase_image)
            Epi = ij.py.to_java(epi_image)

            ij.ui().show('Phase', Phase)
            ij.ui().show('Epi', Epi)

            merged = ij.IJ.run("Merge Channels...", "c1=Phase c2=Epi create")

            merged_images.append(merged)

    hyperstack = np.stack(merged_images, axis=0)
    ij.ui().show("Merged Hyperstack", ij.py.to_java(hyperstack))
    input('Close ImageJ [ENTER]')

def main():
    show_separate_channels()
    #show_merged_channels_2()
if __name__ == '__main__':
    main()