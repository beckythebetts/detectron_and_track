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
        train_dir = self.directory / ('test'+str(test_image_names)) / 'Training_Data' / 'train'
        utils.remake_dir(train_dir / 'Images')

        val_dir = self.directory / ('test' + str(test_image_names)) / 'Training_Data' / 'validate'
        utils.remake_dir(val_dir / 'Images')

        train_jsons = []
        test_jsons = []

        for f in (self.directory / 'images').iterdir():

            if f.stem in test_image_names:
                shutil.copy(f, val_dir / 'Images' / f.name)
                test_jsons.append(f.parents[1] / 'labels' / ('labels' + f.stem + '.json'))
            else:
                shutil.copy(f, train_dir / 'Images' / f.name)
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
        for names in [('00',),
                      ('00a',),
                      ('01',),
                      ('01a',),
                      ('10',),
                      ('10a',),
                      ('11',),
                      ('11a',)]:
            self.make_split(names)
        # for names in [('00', '01'), ('01', '10'), ('10', '11'), ('11', '10')]:
        #     self.make_split(names)

    def train(self):
        APs = []
        for file in self.directory.glob('*test*'):
            train(directory=file)
            unregister_coco_instances('my_dataset_train')
            unregister_coco_instances('my_dataset_val')
            evaluator(directory=file)
            unregister_coco_instances('my_dataset_train')
            unregister_coco_instances('my_dataset_val')
            APs.append(self.getAP(file))
        print(f'*****COMPLETED*****\nAP = {np.mean(APs)} +- {np.std(APs)}')

    def getAP(self, file):
        try:
            with open(file / 'Model' / 'eval.txt', 'r') as f:
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

def merge_jsons(directory):
    all_jsons = [f for f in directory.glob('*.json')]
    json_0 = all_jsons[0]
    for i in range(1, len(all_jsons)):
        call(['python', '-m', 'COCO_merger.merge', '--src', json_0, all_jsons[i], '--out', directory / 'labels.json'])
        json_0 = directory / 'labels.json'

def main():
    my_kfold = KFold(Path('Models') / '04_50_5_5_png')
    my_kfold.split_all()
    my_kfold.train()

    # merge_jsons(Path('RAW_DATA/04/training_dataset/kfold/labels'))

if __name__=='__main__':
    main()