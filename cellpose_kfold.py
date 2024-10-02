from pathlib import Path
import numpy as np
import os
import shutil
import pandas as pd
import json
import matplotlib.pyplot as plt

import cellpose_train
import cellpose_eval

class CellposeKfold:
    def __init__(self, directory):
        self.directory = directory
        self.names = np.unique([im.stem[:2] for im in (directory / 'all').iterdir() if 'im' in im.stem])
        self.pairs = [[directory / 'all' / (name+'im'+'.png'), directory / 'all' / (name+'mask'+'.png')] for name in self.names]

    def split_datasets(self):
        for test_pair in self.pairs:
            train_dir = self.directory / f'test_with_{test_pair[0].stem[:2]}' / 'train'
            val_dir = self.directory / f'test_with_{test_pair[0].stem[:2]}' / 'validate'
            train_dir.mkdir(parents=True)
            val_dir.mkdir()
            for item in test_pair:
                shutil.copy(item, val_dir)
            for train_pair in self.pairs:
                if train_pair != test_pair:
                    for item in train_pair:
                        shutil.copy(item, train_dir)

    def train_eval_datasets(self):
        for dataset_dir in self.directory.iterdir():
            if dataset_dir.name != 'all':
                cellpose_train.cellpose_train(dataset_dir)
                cellpose_eval.cellpose_eval(dataset_dir / 'validate', model=dataset_dir / 'models' / 'model')

    def get_results(self):
        results = []
        for dataset_dir in self.directory.iterdir():
            if dataset_dir.name != 'all':
                for file in (dataset_dir / 'validate').iterdir():
                    if file.suffix == '.txt':
                        resultsi = pd.read_csv(file, sep='\t', index_col=0)
                        results.append(resultsi)
        results = pd.concat(results, axis=0)
        self.means, self.stds = results.groupby(level=0).mean(), results.groupby(level=0).std()
        self.means.to_csv(str(self.directory / 'results_means.txt'), sep='\t')
        self.stds.to_csv(str(self.directory / 'results_stds.txt'), sep='\t')

    def plot_results(self):
        metrics = self.means.columns.values
        thresholds = self.means.index.values
        print(thresholds)
        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        for ax, metric in zip(axs, metrics):
            ax.plot(thresholds, self.means[metric], color='red', label='Cellpose')
            ax.fill_between(thresholds, self.means[metric] - self.stds[metric],
                            self.means[metric] + self.stds[metric], color='red', alpha=0.5, edgecolor='none')
            ax.set_xlabel('IOU Threshold')
            ax.set_ylabel(metric)
            ax.grid(True)
        plt.savefig(str(self.directory / 'results.png'))

    def plot_losses(self):
        train_losses = []
        test_losses = []
        file_names = [file / 'losses.txt' for file in self.directory.iterdir() if file.name != 'all']
        for dataset_dir in self.directory.iterdir():
            if dataset_dir.name != 'all' and dataset_dir.is_dir():
                with open(dataset_dir / 'losses.txt') as f:
                    losses = json.load(f)
                    print(type(losses))
                    train_losses.append(f['Train Losses'])
                    test_losses.append(f['Validation Losses'])
        train_losses = np.vstack(train_losses)
        test_losses = np.vstack(test_losses)
        train_means = np.mean(train_losses, axis=1)
        test_means = np.mean(test_losses, axis=1)
        train_stds = np.std(train_losses, axis=1)
        test_stds = np.std(test_losses, axis=1)
        iterations = np.arange(0, len(train_means))
        test_iterations, test_means, test_stds = [[it, mean, std] for it, mean, std in zip(iterations, test_means, test_stds) if mean !=0]
        plt.rcParams["font.family"] = 'serif'

        plt.plot(test_iterations, test_means, color='red', label='Validation loss')
        plf.fill_between(test_iterations, test_means-test_stds, test_means+test_stds, color='red', alpha=0.5, edgecolor='none')

        plt.plot(iterations, train_means, color='red', label='Train loss')
        plf.fill_between(iterations, train_means - train_stds, train_means + train_stds, color='navy', alpha=0.5,
                         edgecolor='none')

        plt.grid()
        plt.legend()
        plt.savefig(str(self.directory / 'av_loss_plot.png'))


def main():
    kfold = CellposeKfold(Path('cellpose_Models') / 'filter_kfold')
    # kfold.split_datasets()
    # kfold.train_eval_datasets()
    # kfold.get_results()
    # kfold.plot_results()
    kfold.plot_losses()

if __name__ == '__main__':
    main()
