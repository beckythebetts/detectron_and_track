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
from cellpose import io, metrics

import numpy as np
import os, json, cv2, random, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

import SETTINGS

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def eval_with_cellpose(directory):
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
    predicted_masks = np.array([])
    for im in validation_ims:
        im_RGB = np.stack([np.array(im*256)] * 3, axis=-1)
        print('CHECK IMAGE', np.min(im_RGB), np.max(im_RGB))
        detectron_outputs = predictor(im_RGB)
        #print(detectron_outputs)
        class_masks = {class_name: torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
                                                    device=device)
                       for class_name in train_metadata['thing_classes']}

        for i, pred_class in enumerate(detectron_outputs["instances"].pred_classes):
            class_name = train_metadata['thing_classes'][pred_class]
            instance_mask = detectron_outputs["instances"].pred_masks[i].to(device=device)
            # ******* ADJUST FOR NON SQUARE IMAGES*********
            if SETTINGS.REMOVE_EDGE_CELLS:
                if torch.any(torch.nonzero(instance_mask) == 1) or torch.any(
                        torch.nonzero(instance_mask) == SETTINGS.IMAGE_SIZE[0] - 1):
                    continue
            class_masks[class_name] = torch.where(instance_mask,
                                                  torch.tensor(i + 1, dtype=torch.float32),
                                                  class_masks[class_name].to(dtype=torch.float32))
            class_masks[class_name] = class_masks[class_name].to(dtype=torch.int16)

        for class_name, class_mask in class_masks.items():
            class_mask_np = class_mask.cpu().numpy()
            predicted_masks = np.append(predicted_masks, class_mask_np)
    true_masks = [plt.imread(im) for im in (directory/'Training_Data'/'validate'/'Masks').iterdir()]

    thresholds = [0.5, 0.75, 0.9]
    APs, TPs, FPs, FNs = metrics.average_precision(true_masks, predicted_masks, threshold=thresholds)
    precisions = TPs / (TPs + FPs)
    recalls = TPs / (TPs + FNs)
    F1s = TPs / (TPs + 0.5 * (FPs + FNs))
    for i, im_name in enumerate(im_names):
        df = pd.DataFrame({'Precision': precisions[i],
                           'Recall': recalls[i],
                           'F1': F1s[i]},
                          index=thresholds)
        df.to_csv(str(directory / f'{im_name}_results.txt'), sep='\t')
        # view_frame.show_frame(str(directory / f'{im_name}im.png'), str(directory /f'{im_name}pred.png'), str(directory /f'{im_name}_view.png'))
        plt.imsave(str(directory / f'{im_name}_view.png'),
                   utils.show_segmentation(np.array(validation_ims[i]), np.array(predicted_masks[i]).astype(np.int16),
                                           np.array(true_masks[i]).astype(np.int16)))

def main():
    #evaluator()
    eval_with_cellpose(SETTINGS.MASK_RCNN_MODEL)

if __name__ == '__main__':
    main()