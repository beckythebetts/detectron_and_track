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

import SETTINGS
import utils


def main():
    directory = SETTINGS.DIRECTORY
    config_directory = directory / 'model'

    with open(str(config_directory / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(config_directory / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth') # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda:0"
    predictor = DefaultPredictor(cfg)

    input_images_directory = directory / 'inference_dataset' / 'images'

    output_directory = (directory / 'inference_dataset' / 'masks')  # Replace this with the path to your desired output directo
    utils.remake_dir(output_directory)
    for class_name in train_metadata['thing_classes']:
        (output_directory / class_name).mkdir()

    # masks = np.empty(0)
    #
    # for image_path in input_images_directory.glob('*'):
    #     image = cv2.imread(str(image_path))
    #
    #     outputs = predictor(image)["instances"]
    #
    #     class_masks = {class_name: torch.zeros_like(outputs.pred_masks[0], dtype=torch.uint8, device=device)
    #                    for class_name in train_metadata.thing_classes}
    #
    #     for pred_class in outputs.pred_classes:
    #         full_mask = np.zeros(shape=(1, image.shape[0], image.shape[1]))
    #         class_indices = (outputs.pred_classes == pred_class).nonzero()
    #         for idx in class_indices:
    #             #print(type(outputs.pred_masks[idx]))
    #             full_mask += outputs.pred_masks[idx].cpu().numpy()
    #         Image.fromarray(full_mask.astype(np.uint16)).save(output_directory / pred_class / str(image.stem+'_mask.tif'))

    # Loop over the images in the input folder
    for image_filename in os.listdir(input_images_directory):
        image_path = os.path.join(input_images_directory, image_filename)
        new_im = cv2.imread(image_path)

        outputs = predictor(new_im)  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        # Create a dictionary to store the mask for each class with unique integer labels
        class_masks = {class_name: torch.zeros_like(outputs["instances"].pred_masks[0], dtype=torch.uint8, device=torch.device("cuda:0"))
                       for class_name in train_metadata['thing_classes']}

        # Assign a unique integer label to each object in the mask
        for i, pred_class in enumerate(outputs["instances"].pred_classes):
            class_name = train_metadata['thing_classes'][pred_class]
            class_masks[class_name] = torch.where(outputs["instances"].pred_masks[i].to(device=torch.device("cuda:0")),
                                          torch.tensor(i + 1, dtype=torch.float32),
                                          class_masks[class_name].to(dtype=torch.float32))
            class_masks[class_name] = class_masks[class_name].to(dtype=torch.uint8)

        for class_name, class_mask in class_masks.items():
            class_mask_np = class_mask.cpu().numpy()

            Image.fromarray(class_mask_np.astype(np.uint16)).save(
                output_directory / pred_class / str(image.stem + '_mask.tif'))

            # class_filename = os.path.splitext(image_filename)[0] + f"_mask.png"
            # class_output_path = os.path.join(output_directory, class_name, class_filename)
            # cv2.imwrite(class_output_path, class_mask_np.astype(np.uint8))

    print("Segmentation of all images completed.")

if __name__ == '__main__':
    main()