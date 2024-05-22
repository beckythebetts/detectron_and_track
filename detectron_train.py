import torch, detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, launch, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, inference_context
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Instances

import numpy as np
import os, json, cv2, random, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

import SETTINGS



TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

torch.cuda.empty_cache()

directory = SETTINGS.DIRECTORY

class ValidationLossHook(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.val_losses = []

    def after_step(self):
        if self.trainer.iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            print('********DATASET!!*******', self.cfg.DATASETS.TEST[0])
            val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST)
            total_val_loss = 0
            num_batches = 0

            with inference_context(self.trainer.model), torch.no_grad():
                for data in val_loader:
                    loss_dict = self.trainer.model(data)
                    losses = sum(loss_dict.values())
                    total_val_loss += losses.item()
                    num_batches += 1

            avg_val_loss = total_val_loss / num_batches
            self.val_losses.append(avg_val_loss)

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, True, output_dir="./output/")

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, ValidationLossHook(self.cfg))
        return hooks

def train(directory):
    setup_logger()

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
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000  # iteration = run through one batch
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32  # (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[15, 20, 30, 50, 100]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.6, 1.0, 1.5]]
    # cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    # cfg.INPUT.MAX_SIZE_TRAIN = 600
    # cfg.INPUT.MIN_SIZE_TEST = 600
    # cfg.INPUT.MAX_SIZE_TEST = 600
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATASETS.TEST = ('my_dataset_val')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

    config_yaml_path = config_directory / 'config.yaml'
    with open(str(config_yaml_path), 'w') as file:
        yaml.dump(cfg, file)

    train_loss = []
    val_loss = trainer.storage.history("total_loss").values()
    val_loss = trainer.hooks[-2].val_losses  # Access the validation losses

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(config_directory / 'loss_plot.png')

def main():
    train(directory)

if __name__ == '__main__':
    main()
