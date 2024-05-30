import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def plot_average_loss_curves(directory):

    for folder in directory.iterdir():
        with open(folder / 'model' / 'metrics.json', 'r') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            print(lines)
            #train_losses = [x['total_loss'] for x in lines if 'total_loss' in x]
            train_losses = np.where('total_loss' in lines, lines['total_loss'], np.nan)
            val_losses = np.where('validation_loss' in lines, lines['validation_loss'], np.nan)
            print(train_losses, val_losses)

def main():
    plot_average_loss_curves(Path('kfold_quarter_old') / 'train6')

if __name__ == '__main__':
    main()