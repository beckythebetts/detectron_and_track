import numpy as np
import cv2


def to_instance_mask(mask):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    separated_cells = np.zeros_like(mask)
    for label in range(1, num_labels):
        separated_cells[labels == label] = label
    return separated_cells


def to_masks(image_path, type):
    mask_vals = {"amoeba": 127, "yeast": 254, "proximity": 255}
    seg_mask = cv2.imread(image_path)
    seg_mask = seg_mask[:, 1024:]
    if type == 'proximity':
        return np.where(seg_mask[:, :, 2] == 255, 1, 0)
    else:
        return to_instance_mask(np.where(seg_mask[:, :, 2] == mask_vals[type], 1, 0))


def split_mask(mask_full):
    masks = [np.where(mask_full == i+1, 1, 0) for i in range(0, np.max(mask_full)) if i+1 in mask_full]
    return masks


def circle_equ(x, y, centre, radius):
    return (x-centre[0])**2 + (y-centre[1])**2 <= radius**2


def create_circle(centre, radius, array_shape=(1024, 1024)):
    circle = circle_equ(np.arange(0, array_shape[0], 1), np.arange(0, array_shape[1], 1)[:, np.newaxis], centre, radius)
    return circle


def cal_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    result = np.sum(intersection) / np.sum(union)
    return result

