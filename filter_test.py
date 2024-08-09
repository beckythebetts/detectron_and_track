import h5py
import pandas as pd
from pathlib import Path
import numpy as np


class FilterDataFrame:
    def __init__(self, filter, h5_files):
        self.filter = filter
        print('Filter?', self.filter)
        self.measurements = ['path_av_speed', 'displacement_av_speed', 'av_area', 'av_perimeter']
        self.dataframe = pd.DataFrame(columns=self.measurements)
        self.h5_files = h5_files
        for file in self.h5_files:
            self.add_file(file)
        #print(self.dataframe)
        print(self.dataframe.mean())
        print(self.dataframe.std())

    def add_file(self, file_name):
        with h5py.File(Path('Datasets') / 'filter_test' / file_name, 'r') as f:
            for cell_num, cell in enumerate(f['Features']):
                features = f['Features'][cell]
                indices = np.where(~np.isnan(features['area']))[0]
                start_index, end_index = indices[0], indices[-1]
                x_0, y_0 = features['xcentre'][start_index], features['ycentre'][start_index]
                x_1, y_1 = features['xcentre'][end_index], features['ycentre'][end_index]
                displacement_av_speed = np.sqrt((x_1-x_0)**2 + (y_1-y_0)**2) / (end_index - start_index)
                path_av_speed = np.nanmean(features['speed'])
                av_area = np.nanmean(features['area'])
                av_perimeter = np.nanmean(features['perimeter'])
                self.dataframe.loc[cell_num+1] = [path_av_speed, displacement_av_speed, av_area, av_perimeter]
                #pd.concat([self.dataframe, [path_av_speed, displacement_av_speed, av_area, av_perimeter]])



def main():
    filter = FilterDataFrame(True, ['filter00.h5'])
    no_filter = FilterDataFrame(False, ['no_filter00.h5'])

if __name__ =='__main__':
    main()
