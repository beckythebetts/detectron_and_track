import SETTINGS
import utils
from pathlib import Path
import shutil
from subprocess import call
from detectron2.data import MetadataCatalog, DatasetCatalog
import ast
import numpy as np
from collections import OrderedDict

from detectron_train import train
from detectron_eval import evaluator

class KFold:

    def __init__(self, directory):
        self.directory = directory

    def make_split(self, test_image_names):
        train_dir = self.directory / ('test'+str(test_image_names)) / 'training_dataset' / 'train'
        utils.remake_dir(train_dir / 'images')

        val_dir = self.directory / ('test' + str(test_image_names)) / 'training_dataset' / 'validate'
        utils.remake_dir(val_dir / 'images')

        train_jsons = []
        test_jsons = []

        for f in (self.directory / 'images').iterdir():

            if f.stem in test_image_names:
                shutil.copy(f, val_dir / 'images' / f.name)
                test_jsons.append(f.parents[1] / 'labels' / ('labels' + f.stem + '.json'))
            else:
                shutil.copy(f, train_dir / 'images' / f.name)
                train_jsons.append(f.parents[1] /'labels' / ('labels'+f.stem+'.json'))

        print(train_jsons, test_jsons)
        # Merge all train jsons
        if len(train_jsons) > 1:
            json_0 = train_jsons[0]
            for i in range(1, len(train_jsons)):
                call(['python', '-m', 'COCO_merger.merge', '--src', json_0, train_jsons[i], '--out', train_dir / 'labels.json' ])
                json_0 = train_dir / 'labels.json'
        else:
            shutil.copy(train_jsons[0], train_dir / 'labels.json')

        # merge all test jsons
        if len(test_jsons) > 1:
            json_0 = test_jsons[0]
            for i in range(1, len(test_jsons)):
                call(['python', '-m', 'COCO_merger.merge', '--src', json_0, test_jsons[i], '--out', val_dir / 'labels.json'])
                json_0 = val_dir / 'labels.json'
        else:
            shutil.copy(test_jsons[0], val_dir / 'labels.json')

    def split_all(self):
        for names in [('00', '01', '02', '03', '04'),
                      ('01', '02', '03', '04', '05'),
                      ('02', '03', '04', '05', '06'),
                      ('03', '04', '05', '06', '00'),
                      ('04', '05', '06', '00', '01'),
                      ('05', '06', '00', '01', '02'),
                      ('06', '00', '01', '02', '03')]:
            self.make_split(names)
        # for names in [('00', '01'), ('01', '10'), ('10', '11'), ('11', '10')]:
        #     self.make_split(names)

    def train(self):
        APs = []
        for file in self.directory.glob('*test*'):
            train(file)
            unregister_coco_instances('my_dataset_train')
            unregister_coco_instances('my_dataset_val')
            evaluator(file)
            unregister_coco_instances('my_dataset_train')
            unregister_coco_instances('my_dataset_val')
            APs.append(self.getAP(file))
        print(f'*****COMPLETED*****\nAP = {np.mean(APs)} +- {np.std(APs)}')

    def getAP(self, file):
        try:
            with open(file / 'model' / 'eval.txt', 'r') as f:
                AP_string = f.read()

            AP_string = AP_string.replace('nan', 'np.nan').strip()
            AP_dict = eval(AP_string)
            return AP_dict[0]['segm']['AP']
        except ValueError as e:
            print("Error:", e)

def unregister_coco_instances(name):
    if name in DatasetCatalog.list():
        DatasetCatalog.pop(name)
    if name in MetadataCatalog.list():
        MetadataCatalog.pop(name)


def main():
    my_kfold = KFold(Path('kfold_18_old'))
    my_kfold.split_all()
    my_kfold.train()

if __name__=='__main__':
    main()