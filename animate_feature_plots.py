import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from pathlib import Path

import SETTINGS
import utils


def init(lines):
    for line in lines:
        line.set_data([], [])
    return lines


def update(frame, lines, data, axs):
    for i, line in enumerate(lines):
        line.set_data(data.index[:frame], data.iloc[:frame, i])
        axs[i].relim()  # Recalculate limits
        axs[i].autoscale_view()
    return lines



def animate_plot_features(cell_number):
    utils.remake_dir(SETTINGS.DIRECTORY / 'features_animated')
    plt.rcParams["font.family"] = 'serif'
    features_path = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(cell_number)+'.txt')
    data = pd.read_csv(features_path, sep='\t')
    fig, axs = plt.subplots(5, sharex=True, figsize=(10, 10))

    lines = []
    ylabels = ['Area', 'Speed', 'Perimeter' , 'Distance to nearest yeast', 'Yeast pixels']
    for i in range(5):
        line = axs[i].plot([], [], color='k')[0]
        lines.append(line)
        #print(data.columns.values.tolist())
        axs[i].set(xlim=[0, 1200], ylabel=ylabels[i])
        axs[i].grid()

    fig.suptitle('Amoeba ' + features_path.stem)
    axs[-1].set(xlabel='frames')



    ani = FuncAnimation(fig, update, fargs=(lines, data, axs), frames=range(1, 1000), init_func=lambda: init(lines), blit=False)

    # Save the animation as a GIF
    animation_path = SETTINGS.DIRECTORY / 'features_animated' / (
            features_path.stem + '.gif')  # Replace with your save directory path
    ani.save(animation_path, writer='pillow', fps=30)

def show_cell(cell_index):
    utils.remake_dir(SETTINGS.DIRECTORY / 'show_cell')
    for frame in (SETTINGS.DIRECTORY / 'phase' / 'inference_dataset'):
        image = torch.tensor(utils.read_tiff(frame)).cuda()
        epi_image = torch.tensor(utils.read_tiff(
            SETTINGS.DIRECTORY / 'inference_dataset' / 'epi' / (frame.stem + '.tif')).astype(
            np.float32)).cuda()
        mask = torch.tensor(utils.read_tiff(
            SETTINGS.DIRECTORY / 'tracked' / 'phase' / (frame.stem + '.tif')).astype(
            np.int16)).cuda()
        outline = mask_funcs.mask_outline(torch.where(mask == int(cell_index), 1, 0), thickness=1)
        epi_image_normalised = (epi_image - epi_image.min()) / (epi_image.max() - epi_image.min()) * 255
        im_rgb = torch.stack((image, image, image), axis=0)

        im_rgb[0] = torch.where(outline, 0, im_rgb[0])
        im_rgb[1] = torch.where(outline, 0, im_rgb[1])
        im_rgb[2] = torch.where(outline, 255, im_rgb[2])

        im_rgb[0] = torch.where(epi_image > SETTINGS.THRESHOLD, epi_image_normalised, im_rgb[0])
        im_rgb[1] = torch.where(epi_image > SETTINGS.THRESHOLD, 0, im_rgb[1])
        im_rgb[2] = torch.where(epi_image > SETTINGS.THRESHOLD, 0, im_rgb[2])

        im_rgb = im_rgb.permute(1, 2, 0)

        utils.save_tiff((im_rgb).cpu().numpy().astype(np.uint8),
                        SETTINGS.DIRECTORY / 'show_cell' ("{0:04}".format(frame) + '.jpg'))


def main():
    #animate_plot_features(17)
    show_cell(17)

if __name__ == '__main__':
    main()

