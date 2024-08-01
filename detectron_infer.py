import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import numpy as np
import os, json, cv2, random
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import sys
import glob
import h5py

import SETTINGS
import utils

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    print('--------------------\nSEGMENTING - ', SETTINGS.CLASSES['phase'], '\n--------------------')
    config_directory = SETTINGS.MASK_RCNN_MODEL / 'Model'
    #config_directory = Path('ims_for_report/phase/cfg_Ph_R50_DC5_1/kaggle/working/config_dir')

    with open(str(config_directory / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(config_directory / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth') # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    with h5py.File(SETTINGS.DATASET, 'r+') as f:
        if 'Segmentations' in f:
            del f['Segmentations']
        for frame, im in f['Images']['Phase'].items():

            sys.stdout.write(f'\rSegmenting image {int(frame)+1} / {f["Images"].attrs["Number of frames"]}')
            sys.stdout.flush()
            detectron_outputs = predictor(np.stack([np.array(im)]*3, axis=-1))
            class_masks = {class_name: torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
                                                        device=device)
                           for class_name in train_metadata['thing_classes']}

            for i, pred_class in enumerate(detectron_outputs["instances"].pred_classes):
                class_name = train_metadata['thing_classes'][pred_class]
                instance_mask = detectron_outputs["instances"].pred_masks[i].to(device=device)
                # ******* ADJUST FOR NON SQUARE IMAGES*********
                if SETTINGS.REMOVE_EDGE_CELLS:
                    if torch.any(torch.nonzero(instance_mask)==1) or torch.any(torch.nonzero(instance_mask)==SETTINGS.IMAGE_SIZE[0]-1):
                        continue
                class_masks[class_name] = torch.where(instance_mask,
                                                      torch.tensor(i + 1, dtype=torch.float32),
                                                      class_masks[class_name].to(dtype=torch.float32))
                class_masks[class_name] = class_masks[class_name].to(dtype=torch.int16)

            for class_name, class_mask in class_masks.items():
                class_mask_np = class_mask.cpu().numpy()
                mask = f.create_dataset(f'Segmentations/Phase/{frame:04d}', dtype='i2', data=class_mask_np)
                #f['Segmentations']['Phase'][i] = class_mask_np


if __name__ == '__main__':
    main()
