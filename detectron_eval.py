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

setup_logger()

def main():
    directory = SETTINGS.DIRECTORY
    dataset_dir = directory / 'training_dataset'
    config_directory = directory / 'model'

    register_coco_instances("my_dataset_val", {}, str(dataset_dir / 'validate' / 'labels.json'),
                            str(dataset_dir / 'validate' / 'images'))

    val_metadata = MetadataCatalog.get("my_dataset_val")
    val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(config_directory)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_val", output_dir="./output", max_dets_per_image=1000)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator), 2)

if __name__ == '__main__':
    main()