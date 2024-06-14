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

import SETTINGS
import utils

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    print('--------------------\nSEGMENTING - ', SETTINGS.CLASSES['phase'], '\n--------------------')
    directory = SETTINGS.DIRECTORY
    config_directory = directory / 'model'
    #config_directory = Path('ims_for_report/phase/cfg_Ph_R50_DC5_1/kaggle/working/config_dir')

    with open(str(config_directory / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(config_directory / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth') # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda"
    cfg.INPUT.MIN_SIZE_TEST = 1200
    # cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    input_images_directory = directory / 'inference_dataset' / 'phase'
    # input_images_directory = Path('ims_for_report/phase/images')

    output_directory = (directory / 'segmented' / 'phase')  # Replace this with the path to your desired output directo
    # output_directory = Path('ims_for_report/phase/segmented')
    utils.remake_dir(output_directory)
    num_images = len(os.listdir(input_images_directory))
    # Loop over the images in the input folder
    for i, image_filename in enumerate(os.listdir(input_images_directory)):
        sys.stdout.write(f'\rSegmenting image {i+1} / {num_images}')
        sys.stdout.flush()
        image_path = os.path.join(input_images_directory, image_filename)
        new_im = cv2.imread(image_path)

        outputs = predictor(new_im)  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        # Create a dictionary to store the mask for each class with unique integer labels
        # class_masks = {class_name: torch.zeros_like(outputs["instances"].pred_masks[0], dtype=torch.int16, device=torch.device("cuda:0"))
        #                for class_name in train_metadata['thing_classes']}
        class_masks = {class_name: torch.zeros_like(outputs["instances"].pred_masks[0], dtype=torch.int16,
                                                    device=device)
                       for class_name in train_metadata['thing_classes']}

        # Assign a unique integer label to each object in the mask
        for i, pred_class in enumerate(outputs["instances"].pred_classes):
            class_name = train_metadata['thing_classes'][pred_class]
            #instance_mask = outputs["instances"].pred_masks[i].to(device=torch.device("cuda:0"))
            instance_mask = outputs["instances"].pred_masks[i].to(device=device)
            # ******* ADJUST FOR NON SQUARE IMAGES*********
            # if SETTINGS.REMOVE_EDGE_CELLS:
            #     if torch.any(torch.nonzero(instance_mask)==1) or torch.any(torch.nonzero(instance_mask)==SETTINGS.IMAGE_SIZE[0]-1):
            #         continue
            class_masks[class_name] = torch.where(instance_mask,
                                          torch.tensor(i + 1, dtype=torch.float32),
                                          class_masks[class_name].to(dtype=torch.float32))
            class_masks[class_name] = class_masks[class_name].to(dtype=torch.int16)


        for class_name, class_mask in class_masks.items():
            class_mask_np = class_mask.cpu().numpy()
            image_name = Path(image_path).stem+'_mask.tif'
            Image.fromarray(class_mask_np.astype(np.uint16)).save(output_directory / image_name)

if __name__ == '__main__':
    main()
