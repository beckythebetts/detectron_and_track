import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib import rc

def plot_average_loss_curves(directory):
    plt.rcParams["font.family"] = 'serif'
    all_train_losses = []
    all_val_losses = []
    for folder in directory.iterdir():

        with open(folder / 'model' / 'metrics.json', 'r') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            train_losses = [x['total_loss'] for x in lines if 'total_loss' in x]
            train_iters = [x['iteration'] for x in lines if 'total_loss' in x]

            val_losses = [x['validation_loss'] for x in lines if 'validation_loss' in x]
            val_iters = [x['iteration'] for x in lines if 'validation_loss' in x]

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)

    average_train_losses = np.mean(all_train_losses, axis=0)
    average_val_losses = np.mean(all_val_losses, axis=0)

    std_train_losses = np.std(all_train_losses, axis=0)
    std_val_losses = np.std(all_val_losses, axis=0)

    plt.plot(val_iters, average_val_losses, color='red', label='Validation Loss')
    plt.fill_between(val_iters, average_val_losses-std_val_losses, average_val_losses+std_val_losses, color='salmon')

    plt.plot(train_iters, average_train_losses, color='navy', label='Training Loss')
    plt.fill_between(train_iters, average_train_losses - std_train_losses, average_train_losses + std_train_losses, color='lightsteelblue')
    plt.legend()
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.savefig(directory / 'av_loss_plot.png')

def main():
    plot_average_loss_curves(Path('kfold_test') / 'train7')

if __name__ == '__main__':
    main()