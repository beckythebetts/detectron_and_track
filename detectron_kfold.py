import SETTINGS
import utils
from pathlib import Path
import shutil
from subprocess import call

from detectron_train import train
from detectron_eval import eval

class KFold:

    def __init__(self, directory):
        self.directory = directory

    def make_split(self, test_image_name):
        train_dir = self.directory / ('test'+test_image_name) / 'training_dataset' / 'train'
        utils.remake_dir(train_dir / 'images')

        val_dir = self.directory / ('test' + test_image_name) / 'training_dataset' / 'validate'
        utils.remake_dir(val_dir / 'images')

        train_jsons = []

        for f in (self.directory / 'images').iterdir():

            if f.stem == test_image_name:
                shutil.copy(f, val_dir / 'images' / f.name)
                shutil.copy(f.parents[1] / 'labels' / ('labels'+f.stem+'.json'), val_dir / 'labels.json')
            else:
                shutil.copy(f, train_dir / 'images' / f.name)
                train_jsons.append(f.parents[1] /'labels' / ('labels'+f.stem+'.json'))

            # Merge all train jsons
        json_0 = train_jsons[0]
        for i in range(1, len(train_jsons)):
            call(['python', '-m', 'COCO_merger.merge', '--src', json_0, train_jsons[i], '--out', train_dir / 'labels.json' ])
            json_0 = train_dir / 'labels.json'

    def split_all(self):
        for name in (self.directory / 'images').iterdir():
            self.make_split(name.stem)

    def train(self):
        for file in self.directory.glob('*test*'):
            train(file)


def main():
    my_kfold = KFold(Path('kfold_test'))
    my_kfold.split_all()
    my_kfold.train()

if __name__=='__main__':
    main()