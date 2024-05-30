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
            train_losses = [x['total_loss'] for x in lines if 'total_loss' in x]
            train_iters = [x['iteration'] for x in lines if 'total_loss' in x]

            val_losses = [x['validation_loss'] for x in lines if 'validation_loss' in x]
            val_iters = [x['iteration'] for x in lines if 'validation_loss' in x]
            print(train_iters, val_iters)

def main():
    plot_average_loss_curves(Path('kfold_quarter_old') / 'train6')

if __name__ == '__main__':
    main()