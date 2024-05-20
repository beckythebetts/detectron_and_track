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

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

torch.cuda.empty_cache()

directory = SETTINGS.DIRECTORY


def train(directory):
    dataset_dir = directory / 'training_dataset'
    config_directory = directory / 'model'
    register_coco_instances("my_dataset_train", {}, str(dataset_dir / 'train' / 'labels.json'), str(dataset_dir / 'train' / 'images'))
    register_coco_instances("my_dataset_val", {},str(dataset_dir / 'validate' / 'labels.json'), str(dataset_dir / 'validate' / 'images'))

    train_metadata = MetadataCatalog.get("my_dataset_train")
    train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

    val_metadata = MetadataCatalog.get("my_dataset_val")
    val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

    if config_directory.is_dir():
        shutil.rmtree(str(config_directory))
    config_directory.mkdir()
    with open(str(config_directory / "train_metadata.json"), 'w') as json_file:
        json.dump(train_metadata.as_dict(), json_file)

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = str(config_directory)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    # cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000  # iteration = run through one batch
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32  # (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

    config_yaml_path = config_directory / 'config.yaml'
    with open(str(config_yaml_path), 'w') as file:
        yaml.dump(cfg, file)

def main():
    train(directory)

if __name__ == '__main__':
    main()
