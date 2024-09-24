from cellpose import models, metrics, core, io
import matplotlib.pyplot as plt

import SETTINGS


def cellpose_eval(directory):
    use_GPU = core.use_gpu()
    io.logger_setup()
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=str(SETTINGS.CELLPOSE_MODEL / 'models' / 'model'))
    im_names = [im.stem[:2] for im in directory.iterdir() if 'im' in im.stem]
    validation_ims = [io.imread(im) for im in directory.iterdir() if 'im' in im.name]
    channels = [0, 0]
    masks, flow, styles = model.eval(validation_ims)
    for mask, im_name in zip(masks, im_names):
        plt.imsave(str(directory/f'{im_name}pred.png'), mask, cmap='gray')


def main():
    cellpose_eval(SETTINGS.CELLPOSE_MODEL / 'validate')

if __name__ == '__main__':
    main()