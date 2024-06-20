import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

import SETTINGS
import utils

def animate_plot_features(cell_number):
    utils.remake_dir(SETTINGS.DIRECTORY / 'features_animated')
    plt.rcParams["font.family"] = 'serif'
    features_path = SETTINGS.DIRECTORY / 'features' / ("{0:04}".format(cell_number)+'.txt')
    data = pd.read_csv(features_path, sep='\t')
    fig, axs = plt.subplots(5, sharex=True, figsize=(10, 10))

    lines = []
    for i in range(5):
        line, = axs[i].plot([], [], color='k')
        lines.append(line)
        #print(data.columns.values.tolist())
        axs[i].set(xlim=[0, 1200], ylabel=data.columns.values.tolist()[i])
        axs[i].grid()

    fig.suptitle('Amoeba ' + features_path.stem)
    axs[-1].set(xlabel='frames')

# Define the animation function
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(data.index[:frame], data.iloc[:frame, i])
        return lines

    ani = FuncAnimation(fig, update, frames=range(1, len(data) + 1), init_func=init, blit=True)
    plt.show()
    # Save the animation as a GIF
    animation_path = SETTINGS.DIRECTORY / 'features_animated' / (
            features_path.stem + '.gif')  # Replace with your save directory path
    ani.save(animation_path, writer='pillow', fps=30)

    plt.close()

def main():
    animate_plot_features(1)

if __name__ == '__main__':
    main()

