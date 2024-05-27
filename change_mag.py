import cv2
import json
import copy
from pathlib import Path


def downsample_image(image, scale_factor):
    """
    Downsample the image by a given scale factor.

    Parameters:
    - image: Input high-resolution image.
    - scale_factor: Factor by which to downsample the image. E.g., 0.5 for half resolution.

    Returns:
    - downsampled_image: The downsampled image.
    - scale_factor: Actual scale factor used for resizing.
    """
    # Calculate new dimensions
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    new_dimensions = (width, height)

    # Resize image
    downsampled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return downsampled_image, scale_factor


def scale_annotations(annotations, scale_factor):
    """
    Scale the segmentation coordinates in the annotations.

    Parameters:
    - annotations: List of COCO annotations.
    - scale_factor: Factor by which to scale the coordinates.

    Returns:
    - scaled_annotations: The annotations with scaled segmentation coordinates.
    """
    scaled_annotations = copy.deepcopy(annotations)
    for annotation in scaled_annotations:
        if 'segmentation' in annotation:
            for i in range(len(annotation['segmentation'])):
                annotation['segmentation'][i] = [coord * scale_factor for coord in annotation['segmentation'][i]]
        if 'bbox' in annotation:
            annotation['bbox'] = [coord * scale_factor for coord in annotation['bbox']]
    return scaled_annotations


def update_coco_json(coco_json, scale_factor):
    """
    Update the COCO JSON structure with scaled image dimensions and annotations.

    Parameters:
    - coco_json: Original COCO JSON data.
    - scale_factor: Factor by which to scale the image and annotations.

    Returns:
    - updated_coco_json: The updated COCO JSON data.
    """
    updated_coco_json = copy.deepcopy(coco_json)

    # Update image dimensions
    for image_info in updated_coco_json['images']:
        image_info['width'] = int(image_info['width'] * scale_factor)
        image_info['height'] = int(image_info['height'] * scale_factor)

    # Update annotations
    updated_coco_json['annotations'] = scale_annotations(coco_json['annotations'], scale_factor)

    return updated_coco_json


def do_resize(im_path, label_path, new_im_path, new_label_path, sf=0.5):
    # Load the original COCO JSON file
    with open(label_path) as f:
        coco_json = json.load(f)

    # Load the original high-resolution image
    original_image = cv2.imread(im_path)

    # Downsample the image to simulate lower magnification
    scale_factor = sf  # Example: reduce resolution by half
    downsampled_image, scale_factor = downsample_image(original_image, scale_factor)

    # Save the downsampled image
    cv2.imwrite(new_im_path, downsampled_image)

    # Update the COCO JSON file
    updated_coco_json = update_coco_json(coco_json, scale_factor)

    # Save the updated COCO JSON file
    with open(new_label_path, 'w') as f:
        json.dump(updated_coco_json, f)

def main():
    for im in Path('RAW_DATA/04/training_dataset/kfold/images').iterdir():
        do_resize(str(im), str(im.parents[1] / 'labels' / ('labels' + im.stem + '.json')), str(Path('RAW_DATA/04/training_dataset/kfold_half') / 'images' / im.name), str(Path('RAW_DATA/04/training_dataset/kfold_half') / 'labels' / ('labels' + im.stem + '.json')))

if __name__ == '__main__':
    main()