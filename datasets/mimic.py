"""Implements dataloaders for generic MIMIC tasks."""
import os
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle
import copy
import pytorch_lightning as pl


class MIMICDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, task=-1, batch_size=32, num_workers=1, train_shuffle=True, filename='im.pk'):
        """Get dataloaders for MIMIC dataset.

        Args:
            data_dir: Directory where the data is stored.
            task (int): Integer between -1 and 19 inclusive, -1 means mortality task, 0-19 means icd9 task.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers to load data in. Defaults to 1.
            train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
            filename (str, optional): Datafile name. Defaults to 'im.pk'.
        """

        super().__init__()
        # turn parameters into attributes
        self.data_dir = data_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.filename = filename
        self.datasets = None
        self.le = None

    def setup(self, stage=None):
        with open(os.path.join(self.data_dir, self.filename), 'rb') as f:
            datafile = pickle.load(f)
        X_t = datafile['ep_tdata']
        X_s = datafile['adm_features_all']

        X_t[np.isinf(X_t)] = 0
        X_t[np.isnan(X_t)] = 0
        X_s[np.isinf(X_s)] = 0
        X_s[np.isnan(X_s)] = 0

        X_s_avg = np.average(X_s, axis=0)
        X_s_std = np.std(X_s, axis=0)
        X_t_avg = np.average(X_t, axis=(0, 1))
        X_t_std = np.std(X_t, axis=(0, 1))

        for i in range(len(X_s)):
            X_s[i] = (X_s[i] - X_s_avg) / X_s_std
            for j in range(len(X_t[0])):
                X_t[i][j] = (X_t[i][j] - X_t_avg) / X_t_std

        if self.task < 0:
            y = datafile['adm_labels_all'][:, 1]
            admlbl = datafile['adm_labels_all']
            le = len(y)
            for i in range(le):
                if admlbl[i][1] > 0:
                    y[i] = 1
                elif admlbl[i][2] > 0:
                    y[i] = 2
                elif admlbl[i][3] > 0:
                    y[i] = 3
                elif admlbl[i][4] > 0:
                    y[i] = 4
                elif admlbl[i][5] > 0:
                    y[i] = 5
                else:
                    y[i] = 0
        else:
            y = datafile['y_icd9'][:, self.task]
            le = len(y)
        self.le = le
        self.datasets = [(X_s[i].astype(np.float32), X_t[i].astype(np.float32), y[i]) for i in range(le)]

        random.seed(10)
        random.shuffle(self.datasets)

    def train_dataloader(self):
        return DataLoader(
            self.datasets[self.le // 5:],
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[: self.le // 10],
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset_robust = copy.deepcopy(self.datasets[self.le // 10:self.le // 5])

        X_s_robust = [dataset_robust[i][0]
                      for i in range(len(self.datasets[self.le // 10:self.le // 5]))]

        X_t_robust = [dataset_robust[i][1]
                      for i in range(len(self.datasets[self.le // 10:self.le // 5]))]
        y_robust = [dataset_robust[i][2] for i in range(len(self.datasets[self.le // 10:self.le // 5]))]
        return DataLoader(
            [
                (X_s_robust[i], X_t_robust[i], y_robust[i])
                for i in range(len(y_robust))
            ],
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )
