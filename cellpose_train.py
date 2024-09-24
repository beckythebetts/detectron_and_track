from cellpose import io, models, core
from cellpose.train import train_seg



def train(train_directory, test_directory):
    use_GPU = core.use_gpu()
    io.logger_setup()
    images, labels, image_names, test_images, test_labels, image_names_test = io.load_train_test_data(train_directory, test_directory, image_filter='im', mask_filter='mask')
    model = models.CellposeModel(gpu=use_GPU, model_type='cyto3')
    model_path, train_losses, test_losses = train_seg(model.net,
                                                            train_data=images, train_labels=labels,
                                                            channels=[0, 0], normalize=True,
                                                            test_data=test_images, test_labels=test_labels,
                                                            weight_decay=1e-4, SGD=True, learning_rate=0.1,
                                                            n_epochs=100, model_name="01")

def main():
    train('/home/ubuntu/Documents/detectron_and_track/cellpose_Models/filters/train', '/home/ubuntu/Documents/detectron_and_track/cellpose_Models/filters/validate')

if __name__ == '__main__':
    main()