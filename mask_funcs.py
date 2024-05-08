import numpy as np
import cv2
import torch
import torch.nn.functional as F


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

class SplitMask:

    def __init__(self, mask_full):
        self.mask_full = mask_full
        self.i = 1
        self.max = torch.max(self.mask_full)
    def __iter__(self):
        return self
    def __next__(self):
        if self.i <= self.max:
            while self.i not in self.mask_full:
                self.i += 1
            self.i += 1
            return torch.where(self.mask_full == self.i, 1, 0), self.i
        else:
            raise StopIteration

def split_mask(mask_full, use_torch=False, return_indices=False):
    # Return indices=True only works if use_troch=True
    if use_torch:
        #masks = torch.tensor([torch.where(mask_full == i + 1, 1, 0) for i in range(0, int(torch.max(mask_full))) if i + 1 in mask_full])
        masks = []
        max_val = int(torch.max(mask_full))
        masks_dict = {}
        for i in range(1, max_val + 1):
            if i in mask_full:
                mask = torch.where(mask_full == i, 1, 0)
                if return_indices:
                    masks_dict[i] = mask
                else:
                    masks.append(mask)
        if not return_indices:
            masks = torch.stack(masks)
    else:
        masks = [[np.where(mask_full == i + 1, 1, 0)] for i in range(0, np.max(mask_full)) if i + 1 in mask_full]
    if return_indices:
        return masks_dict
    else:
        return masks


def circle_equ(x, y, centre, radius):
    return (x-centre[0])**2 + (y-centre[1])**2 <= radius**2


def create_circle(centre, radius, array_shape=(1024, 1024)):
    circle = circle_equ(np.arange(0, array_shape[0], 1), np.arange(0, array_shape[1], 1)[:, np.newaxis], centre, radius)
    return circle

def torch_circle(centre, radius, array_shape=(1200, 1200)):
    circle = circle_equ(torch.arange(0, array_shape[0], 1), torch.arange(0, array_shape[1], 1).unsqueeze(0), centre, radius)
    return circle

def cal_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    result = np.sum(intersection) / np.sum(union)
    return result

def mask_outline(mask, thickness):
    expanded_mask = F.max_pool2d(mask.float().unsqueeze(0), kernel_size=2*thickness+1, stride=1, padding=thickness) > 0
    # print(expanded_mask.shape)
    outline = (expanded_mask.byte().squeeze() - mask).bool()
    return outline

def centre(mask):
    coords = torch.nonzero(mask)
    len = coords.shape[0]
    x_mean = torch.sum(coords[:, 1]) / len
    y_mean = torch.sum(coords[:, 0]) / len
    return x_mean, y_mean

