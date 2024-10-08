import h5py
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class DataFrame:
    def __init__(self, interval, h5_files):
        self.interval = interval
        self.measurements = ['path_av_speed', 'displacement_av_speed', 'av_area', 'av_perimeter']
        self.dataframe = pd.DataFrame(columns=self.measurements)
        self.h5_files = h5_files
        for file in self.h5_files:
            self.add_file(file)
        #print(self.dataframe)
        self.dataframe.to_csv(Path('Datasets') / 'interval_test' / ('interval' + str(interval) + '.txt'))
        print(self.dataframe.mean())
        print(self.dataframe.std())

    def add_file(self, file_name):
        with h5py.File(Path('Datasets') / 'interval_test' / file_name, 'r') as f:
            for cell_num, cell in enumerate(f['Features']):
                features = f['Features'][cell]['MorphologicalFeatures']
                indices = np.where(~np.isnan(features['area']))[0]
                start_index, end_index = indices[0], indices[-1]
                x_0, y_0 = features['xcentre'][start_index], features['ycentre'][start_index]
                x_1, y_1 = features['xcentre'][end_index], features['ycentre'][end_index]
                displacement_av_speed = np.sqrt((x_1-x_0)**2 + (y_1-y_0)**2) / (end_index - start_index)
                path_av_speed = np.nanmean(features['speed'])
                av_area = np.nanmean(features['area'])
                av_perimeter = np.nanmean(features['perimeter'])
                self.dataframe.loc[len(self.dataframe)] = [path_av_speed, displacement_av_speed, av_area, av_perimeter]
                #pd.concat([self.dataframe, [path_av_speed, displacement_av_speed, av_area, av_perimeter]])


def plot():
    plt.rcParams["font.family"] = 'serif'

    interval15 = pd.read_csv(Path('RAW_DATA') / 'interval_test' / ('interval15.txt'))
    interval3 = pd.read_csv(Path('RAW_DATA') / 'interval_test' / ('interval3.txt'))

    interval15[['path_av_speed', 'displacement_av_speed']] = interval15[['path_av_speed', 'displacement_av_speed']] / 15
    interval3[['path_av_speed', 'displacement_av_speed']] = interval3[['path_av_speed', 'displacement_av_speed']] / 3

    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    fig.tight_layout()
    measurment_names = ['Average path length per second', 'Average displacement per second', 'Average area', 'Average Perimeter']
    for i, measurement in enumerate(['path_av_speed', 'displacement_av_speed', 'av_area', 'av_perimeter']):
        axs[i].hist(interval15[measurement], bins=75, alpha=0.5, color='Blue', label='15 second interval')
        axs[i].hist(interval3[measurement], bins=75, alpha=0.5, color='Red', label='3 second interval')
        axs[i].grid()
        axs[i].set_xlabel(measurment_names[i])
    fig.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.savefig(Path('RAW_DATA') / 'interval_test' / 'histograms.jpg')
    plt.show()


def main():
    #filter = FilterDataFrame(True, ['filter00.h5', 'filter01.h5'])
    #no_filter = FilterDataFrame(False, ['no_filter00.h5', 'no_filter01.h5'])
    #interval15 = DataFrame(15, ['15sec.hdf5'])
    #interval3 = DataFrame(3, ['3sec.hdf5'])
    plot()
if __name__ =='__main__':
    main()
