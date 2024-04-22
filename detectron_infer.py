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



def min_max_scale(image):
    min = np.min(image)
    max = np.max(image)
    return (image - min) / (max - min)

# def label_mask(mask1, mask2, im, save_path):
#     am_mask, ye_mask, im = plt.imread(am_mask_tiff), plt.imread(ye_mask_tiff), plt.imread(im_tiff)
#     plt.axis('off')
#
#     # split_masks =[np.where(mask == i+1, 1, 0) for i in range(0, np.max(mask)) if i in mask]
#     # for single_mask in split_masks:
#     #     for color_mask in color_masks.T:
#     #         color_mask += np.random.uniform(low=0.1, high=1)*single_mask.T
#     # plt.imshow(color_masks)
#     # #plt.show()
#     im = min_max_scale(im).T
#     im_RGB = np.stack((im, im, im), axis=0)
#     split_am_masks = [np.where(am_mask == i + 1, 1, 0) for i in range(0, np.max(am_mask)) if i in am_mask]
#     split_ye_masks = [np.where(ye_mask == i + 1, 1, 0) for i in range(0, np.max(ye_mask)) if i in ye_mask]
#     for single_mask in split_am_masks:
#         outline=cv2.morphologyEx(single_mask.T.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((6, 6), np.uint8))
#         im_RGB[0] = np.where(outline, 1, im_RGB[0])
#         im_RGB[1] = np.where(outline, 0, im_RGB[1])
#         im_RGB[2] = np.where(outline, 1, im_RGB[2])
#     for single_mask in split_ye_masks:
#         outline=cv2.morphologyEx(single_mask.T.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((6, 6), np.uint8))
#         im_RGB[0] = np.where(outline, 0, im_RGB[0])
#         im_RGB[1] = np.where(outline, 0, im_RGB[1])
#         im_RGB[2] = np.where(outline, 1, im_RGB[2])
#     plt.imshow(im_RGB.T)
#     plt.show()
#     print(np.min(im_RGB.T))
#     Image.fromarray((im_RGB*255).T.astype(np.uint8)).save(save_path)

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
    for class_name in trin_metadata['thing_classes']:
        (output_dir / class_name).mkdir()
    if SETTINGS.SAVE_LABELLED_IMAGES:
        labelled_directory = (directory / 'inference_dataset' / 'labelled_images')
        utils.remake_dir(labelled_directory)

    #output_directory = str(output_directory)

    masks = np.empty(0)
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
            class_filename = os.path.splitext(image_filename)[0] + f"_labelled.png"
            class_output_path = os.path.join(output_directory, class_name, class_filename)
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
