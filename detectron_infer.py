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

    input_images_directory = str(directory / 'inference_dataset' / 'images')

    output_directory = (directory / 'inference_dataset' / 'masks')  # Replace this with the path to your desired output directo
    utils.remake_dir(output_directory)
    for class_name in SETTINGS.CLASSES:
        (output_dir / classname).mkdir()
    if SETTINGS.SAVE_LABELLED_IMAGES:
        labelled_directory = (directory / 'inference_dataset' / 'labelled_images')
        utils.remake_dir(labelled_directory)

    #output_directory = str(output_directory)

    masks = np.empty(0)
    # Loop over the images in the input folder
    for image_filename in os.listdir(input_images_directory):
        image_path = os.path.join(input_images_directory, image_filename)
        new_im = cv2.imread(image_path)

        # Perform prediction on the new image
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
            class_filename = os.path.splitext(image_filename)[0] + f"_{class_name}_result.png"
            class_output_path = os.path.join(output_directory, class_filename)
            cv2.imwrite(class_output_path, class_mask_np.astype(np.uint8))

    print("Segmentation of all images completed.")

if __name__ == '__main__':
    main()



# (directory / 'val_images').mkdir()
#     for d in val_dataset_dicts:  # select number of images for display
#         im = cv2.imread(d["file_name"])
#         outputs = predictor(im)
#         v = Visualizer(im[:, :, ::-1],
#                        metadata=val_metadata,
#                        scale=0.5,
#                        instance_mode=ColorMode.IMAGE_BW
#                        # remove the colors of unsegmented pixels. This option is only available for segmentation models
#                        )
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         cv2.imwrite(str(Path(directory) / 'val_images' / d["file_name"]), out.get_image()[:, :, ::-1])
