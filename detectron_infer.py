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

import SETTINGS

directory = SETTINGS.DIRECTORY

config_directory = directory / model
with open(str(config_directory / 'train_metadata.json')) as json_file:
  train_metadata = json.load(json_file)
cfg = get_cfg()
cfg.merge_from_file(str(config_directory / 'config.yaml'))
cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth') # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.MODEL.DEVICE = "cuda:0"
predictor = DefaultPredictor(cfg)


# Directory path to the input images folder
input_images_directory = str(directory / 'inference_dataset' / 'images')

output_directory = (directory / 'inference_dataset' / 'masks')  # Replace this with the path to your desired output directo
if output_directory.is_dir():
    shutil.rmtree(str(output_directory))
output_dir.mkdir()
output_directory = str(output_directory)

masks = np.empty(0)
# Loop over the images in the input folder
for image_filename in os.listdir(input_images_directory):
    image_path = os.path.join(input_images_directory, image_filename)
    new_im = cv2.imread(image_path)

    # Perform prediction on the new image
    outputs = predictor(new_im)  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    # Create a dictionary to store the mask for each class with unique integer labels
    class_masks = {class_name: torch.zeros_like(outputs["instances"].pred_masks[0], dtype=torch.uint8, device=torch.device("cuda:0"))
                   for class_name in train_metadata.thing_classes}

    # Assign a unique integer label to each object in the mask
    for i, pred_class in enumerate(outputs["instances"].pred_classes):
        class_name = train_metadata.thing_classes[pred_class]
        class_masks[class_name] = torch.where(outputs["instances"].pred_masks[i].to(device=torch.device("cuda:0")),
                                      torch.tensor(i + 1, dtype=torch.float32),
                                      class_masks[class_name].to(dtype=torch.float32))
        # when running on gpu
        # class_masks[class_name] = torch.where(outputs["instances"].pred_masks[i].to(device=torch.device("cuda:0")),
        #                                      i + 1,
        #                                      class_masks[class_name])
        # AND
        # train_metadata['thing_classes'] -> train_metadata.thing_classes and torch.device("cpu") -> torch.device("cuda:0")
        class_masks[class_name] = class_masks[class_name].to(dtype=torch.uint8)

    # Save the masks for each class with unique integer labels
    for class_name, class_mask in class_masks.items():
        # Convert the tensor to a NumPy array and then to a regular (CPU) array
        class_mask_np = class_mask.cpu().numpy()

        # Create the output filename with _class_name_result.png extension
        class_filename = os.path.splitext(image_filename)[0] + f"_{class_name}_result.png"
        class_output_path = os.path.join(output_directory, class_filename)

        # Save the image with unique integer labels
        cv2.imwrite(class_output_path, class_mask_np.astype(np.uint8))
        #np.savetxt(class_output_path, class_mask_np, delimiter=',')

print("Segmentation of all images completed.")





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