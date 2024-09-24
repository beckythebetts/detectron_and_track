import SETTINGS
from cellpose import io, models, core, train
import matplotlib.pyplot as plt
import json

def cellpose_train(directory):
    use_GPU = core.use_gpu()
    io.logger_setup()
    images, labels, image_names, test_images, test_labels, image_names_test = io.load_train_test_data(str(directory / 'train'), str(directory / 'validate'), image_filter='im', mask_filter='mask')
    model = models.CellposeModel(gpu=use_GPU, model_type='cyto3')
    model_path, train_losses, test_losses = train.train_seg(model.net,
                                                            train_data=images, train_labels=labels,
                                                            channels=[0, 0], normalize=True,
                                                            test_data=test_images, test_labels=test_labels,
                                                            weight_decay=1e-4, SGD=True, learning_rate=0.1,
                                                            n_epochs=100, save_path=str(directory), model_name='model')
    losses_dict = {'Train Losses': train_losses, 'Validation Losses': test_losses}
    with open(str(directory / 'losses.txt'), 'w') as f:
        json.dump(losses_dict, f)


def main():
    model_directory = SETTINGS.CELLPOSE_MODEL
    cellpose_train(model_directory)
    #cellpose_train(r'C:\Users\php23rjb\Documents\detectron_and_track\cellpose_Models\filters\train', r'C:\Users\php23rjb\Documents\detectron_and_track\cellpose_Models\filters\validate')

if __name__ == '__main__':
    main()