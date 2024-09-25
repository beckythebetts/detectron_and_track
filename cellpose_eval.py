from cellpose import models, metrics, core, io, plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import SETTINGS
import view_frame
import utils


def cellpose_eval(directory):
    use_GPU = core.use_gpu()
    io.logger_setup()
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=str(SETTINGS.CELLPOSE_MODEL / 'models' / 'model'))
    im_names = [im.stem[:2] for im in directory.iterdir() if 'im' in im.stem]
    validation_ims = [io.imread(im) for im in directory.iterdir() if 'im' in im.name]
    channels = [0, 0]
    preds, flow, styles = model.eval(validation_ims)
    for pred, im_name in zip(preds, im_names):
        plt.imsave(str(directory/f'{im_name}pred.png'), pred, cmap='gray')
    true_masks = [io.imread(im) for im in directory.iterdir() if 'mask' in im.name]
    # NOTE, this is not AP as defined elsewhere
    thresholds = [0.5, 0.75, 0.9]
    APs, TPs, FPs, FNs = metrics.average_precision(true_masks, preds, threshold=thresholds)
    precisions = TPs / (TPs+FPs)
    recalls = TPs / (TPs+FNs)
    F1s = TPs / (TPs + 0.5*(FPs+FNs))
    for i, im_name in enumerate(im_names):
        df = pd.DataFrame({'Precision': precisions[i],
                         'Recall': recalls[i],
                         'F1': F1s[i]},
                        index=thresholds)
        df.to_csv(str(directory / f'{im_name}_results.txt'), sep='\t')
        #view_frame.show_frame(str(directory / f'{im_name}im.png'), str(directory /f'{im_name}pred.png'), str(directory /f'{im_name}_view.png'))
        plt.imsave(str(directory /f'{im_name}_view.png'), utils.show_segmentation(np.array(validation_ims[i]), np.array(preds[i]).astype(np.int16), np.array(true_masks[i]).astype(np.int16)))

def cellpose_eval_from_ims(directory):
    image = [io.imread(str(directory/'Images'/'snap01.png'))]
    mask = [io.imread(str(directory/'Masks'/'01mask.png'))]
    pred = [io.imread(str(directory/'Masks'/'01pred.png'))]
    thresholds = [0.5, 0.75, 0.9]
    APs, TPs, FPs, FNs = metrics.average_precision(mask, pred, threshold=thresholds)
    precisions = TPs / (TPs + FPs)
    recalls = TPs / (TPs + FNs)
    F1s = TPs / (TPs + 0.5 * (FPs + FNs))

    df = pd.DataFrame({'Precision': precisions[0],
                       'Recall': recalls[0],
                       'F1': F1s[0]},
                      index=thresholds)
    df.to_csv(str(directory / f'{im_name}_results.txt'), sep='\t')
    # view_frame.show_frame(str(directory / f'{im_name}im.png'), str(directory /f'{im_name}pred.png'), str(directory /f'{im_name}_view.png'))
    plt.imsave(str(directory / f'{im_name}_view.png'),
               utils.show_segmentation(np.array(image), np.array(pred).astype(np.int16),
                                       np.array(mask).astype(np.int16)))
# class Results:
#     def __init__(self, model, threshold, files):
#         self.model = models
#         self.threshold = threshold
#         for file in files:
#             results = pd.read_csv(file).loc[threshold]
#         self.precisions = [pd.read_csv(file)]
#
# def plot_results(cellpose_results, rcnn_results):
#     cell
def main():
    #cellpose_eval(SETTINGS.CELLPOSE_MODEL / 'validate')
    cellpose_eval_from_ims(SETTINGS.MASK_RCNN_MODEL / 'Training_Data' / 'validate')
if __name__ == '__main__':
    main()