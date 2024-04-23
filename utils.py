from pathlib import Path
import shutil

def remake_dir(path):
    if path.is_dir():
        check = input(f'Delete directory {str(path)}? [y,n]')
        if check.lower() != 'y':
            print('Cancelled')
            return
        shutil.rmtree(path)
    path.mkdir()