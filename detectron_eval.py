import torch, detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

import numpy as np
import os, json, cv2, random, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

import SETTINGS


def evaluator(directory=SETTINGS.MASK_RCNN_MODEL):
    setup_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    dataset_dir =  directory / 'Training_Data'
    config_directory = directory / 'Model'

    register_coco_instances("my_dataset_val", {}, str(dataset_dir / 'validate' / 'labels.json'),
                            str(dataset_dir / 'validate' / 'Images'))

    val_metadata = MetadataCatalog.get("my_dataset_val")
    val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

    with open(str(config_directory / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(config_directory / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth') # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_val", output_dir="./output", max_dets_per_image=1000)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    output = inference_on_dataset(predictor.model, val_loader, evaluator), 2
    print(output)
    with open(str(config_directory / 'eval.txt'), 'w') as f:
        f.write(str(output))

def eval_withcellpose(directory):
    config_directory = directory / 'Model'
    # config_directory = Path('ims_for_report/phase/cfg_Ph_R50_DC5_1/kaggle/working/config_dir')

    with open(str(config_directory / 'train_metadata.json')) as json_file:
        train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(config_directory / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth')  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    validation_ims = [plt.imread(str(im)) for im in (directory/'Training_Data'/'validate'/'Images').iterdir()]
    pred_masks = [predictor(np.stack([im]*3, axis=-1)) for im in validation_ims]

def main():
    evaluator()

if __name__ == '__main__':
    main()