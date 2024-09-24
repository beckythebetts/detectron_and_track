from cellpose import models, metrics, core, io
import matplotlib.pyplot as plt

import SETTINGS


def cellpose_eval(directory):
    io.logger_setup()
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=SETTINGS.CELLPOSE_MODEL / 'models' / 'model')
    validation_ims = [io.imread(im) for im in directory.iterdir() if 'im' in im.name]
    channels = [0, 0]
    masks, flow, styles = model.eval(validation_ims)
    for mask in masks:
        plt.imshow(masks)


def main():
    cellpose_eval(SETTINGS.MASK_RCNN_MODEL / 'validate')

if __name__ == '__main__'
    main()