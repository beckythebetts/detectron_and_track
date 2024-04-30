from pathlib import Path
import shutil
import torch

def remake_dir(path):
    if path.is_dir():
        check = input(f'Delete directory {str(path)}? [y,n]')
        if check.lower() != 'y':
            print('Cancelled')
            return
        shutil.rmtree(path)
    path.mkdir(parents=True)


def min_max_scale(image):
    min = np.min(image)
    max = np.max(image)
    return (image - min) / (max - min)

def torch_min_max_scale(image):
    min = torch.min(image)
    max = torch.max(image)
    return (image - min) / (max - min)