import SETTINGS
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def get_centre(mask):
    x_mesh_grid, y_mesh_grid = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
    area = np.sum(mask)
    x_centre = np.sum(mask*x_mesh_grid)/area
    y_centre = np.sum(mask*y_mesh_grid)/area
    return (x_centre, y_centre)

def dist_between_points(point_1, point_2):
    print(point_1, point_2)
    return np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)


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
        self.i = 0
        self.max = torch.max(self.mask_full)
    def __iter__(self):
        return self
    def __next__(self):
        self.i += 1
        while self.i not in self.mask_full and self.i <= self.max:
            self.i += 1
        if self.i <= self.max:
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

def torch_circle(centre, radius, array_shape=SETTINGS.IMAGE_SIZE):
    circle = circle_equ(torch.arange(0, array_shape[0], 1).cuda().unsqueeze(1), torch.arange(0, array_shape[1], 1).cuda().unsqueeze(0), centre, radius)
    return circle

def cal_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    result = np.sum(intersection) / np.sum(union)
    return result

# def mask_outline(mask, thickness=3):
#     expanded_mask = F.max_pool2d(mask.float().unsqueeze(0), kernel_size=2*thickness+1, stride=1, padding=thickness) > 0
#     outline = (expanded_mask.byte().squeeze() - mask).bool()
#     return outline

def mask_outline(mask, thickness=3):
    with torch.no_grad():
        expanded_mask = F.max_pool2d(mask.float(), kernel_size=2*thickness+1, stride=1, padding=thickness) > 0
        outline = (expanded_mask.squeeze() - mask).bool()
    return outline

# def find_centre(mask):
#     coords = torch.nonzero(mask)
#     len = coords.shape[0]
#     x_mean = torch.sum(coords[:, 1]) / len
#     y_mean = torch.sum(coords[:, 0]) / len
#     return x_mean, y_mean

if __name__ == '__main__':
    #get_centre(np.zeros((5, 5)))
    array_1 = np.array([[0,0], [1,0], [2, 2]])
    array_2 = np.array([[0,1], [2, 2], [3, 4], [2, 1]])
    distances = np.linalg.norm(array_1[:, np.newaxis] - array_2[np.newaxis], axis=2)
    print(distances)

