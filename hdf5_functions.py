import h5py

def rename_datasets(file):
    with h5py.File(file, 'r+') as f:
        for frame in f['Images']['Epi'].keys():
            if len(frame) > 4:
                del f['Images']['Epi'][frame]

def main():
    rename_datasets('Datasets/danhighres/dan3.h5')

if __name__ == '__main__':
    main()