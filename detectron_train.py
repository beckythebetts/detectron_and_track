import torch, detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

import numpy as np
import os, json, cv2, random, shutil
from pathlib import Path
import matplotlib.pyplot as plt

import custom_trainer

setup_logger()

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

directory = '02'
dataset_dir = Path('02') / 'training_dataset'
config_directory = Path('02') / 'model'

def main():
    register_coco_instances("my_dataset_train", {}, str(dataset_dir / 'train' / 'labels.json'), str(dataset_dir / 'train' / 'images'))
    register_coco_instances("my_dataset_val", {},str(dataset_dir / 'validate' / 'labels.json'), str(dataset_dir / 'validate' / 'images'))

    train_metadata = MetadataCatalog.get("my_dataset_train")
    train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

    val_metadata = MetadataCatalog.get("my_dataset_val")
    val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

    if os.path.isdir(str(config_directory)):
        shutil.rmtree(str(config_directory))
    os.mkdir(str(config_directory))
    with open(os.path.join(str(config_directory), "train_metadata.json"), 'w') as json_file:
        json.dump(train_metadata.as_dict(), json_file)

    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(config_directory)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    # cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000  # iteration = run through one batch
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = custom_trainer.MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

    with open(os.path.join(str(config_directory), 'metrics.json')) as metrics_file:
        metrics = [json.loads(line) for line in metrics_file]

    plt.scatter([line['iteration'] for line in metrics if 'total_loss' in line],
                [line['total_loss'] for line in metrics if 'total_loss' in line], marker='.', label='Training')
    plt.scatter([line['iteration'] for line in metrics if 'validation_loss' in line],
                [line['validation_loss'] for line in metrics if 'validation_loss' in line], marker='.',
                label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(config_directory / 'train_curve.png'))

    config_yaml_path = config_directory / 'config.yaml'
    with open(str(config_yaml_path), 'w') as file:
        yaml.dump(cfg, file)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    for d in val_dataset_dicts:  # select number of images for display
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=val_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imsave(str(Path(directory) / 'val_images' / d["file_name"]), out.get_image()[:, :, ::-1])

    evaluator = COCOEvaluator("my_dataset_val", output_dir="./output", max_dets_per_image=1000)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == '__main__':
    main()
