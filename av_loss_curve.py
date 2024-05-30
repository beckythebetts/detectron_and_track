import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def plot_average_loss_curves(directory):
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
    print(train_losses.shape, val_losses.shape)

def main():
    plot_average_loss_curves(Path('kfold_quarter_old') / 'train6')

if __name__ == '__main__':
    main()