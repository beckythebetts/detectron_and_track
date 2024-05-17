import pandas as pd
from tifffile import TiffFile
import shutil
from pathlib import Path
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt


input_path = Path('RAW_DATA') / '03'/ 'all_tiffs'
folder_even = input_path.parent / 'Phase' # folder of first (0) image
folder_odd = input_path.parent / 'Epi'
def split_channels(input_path, folder_even, folder_odd):
    # input folder is folder of individual tiffs, from alternating channels. Copies tiffs to relevant folders
    folder_even.mkdir(exist_ok=True)
    folder_odd.mkdir(exist_ok=True)
    odd = False
    for i, image in enumerate(input_path.iterdir()):
        if odd:
            shutil.copy(str(image), str(folder_odd / str('t'+str(int((i-1)/2))+'.tif')))
        else:
            shutil.copy(str(image), str(folder_even / str('t'+str(int(i/2))+'.tif')))
        odd = not odd

# Merge COCOC:
# pip install COCO_merger
# python -m COCO_merger.merge --src Json1.json Json2.json --out OUTPUT_JSON.json

def plot_mem():
    data = pd.read_csv('03_test/features_memory.txt')
    plt.plot(data)
    plt.show()
if __name__ == '__main__':
    #split_channels(input_path, folder_even, folder_odd)
    #split_coco()
    plot_mem()