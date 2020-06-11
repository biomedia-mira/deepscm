import os

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from deepscm.morphomnist import io


class MNISTLike(TensorDataset):
    def __init__(self, root_dir, train: bool = True):
        self.root_dir = root_dir
        prefix = "train" if train else "t10k"
        images_filename = prefix + "-images-idx3-ubyte.gz"
        labels_filename = prefix + "-labels-idx1-ubyte.gz"
        self.images = torch.from_numpy(io.load_idx(os.path.join(self.root_dir, images_filename)))
        self.labels = torch.from_numpy(io.load_idx(os.path.join(self.root_dir, labels_filename)))
        super().__init__(self.images, self.labels)


class InterleavedDataset(Dataset):
    def __init__(self, datasets, which):
        lengths = [len(dataset) for dataset in datasets]
        if any(length != lengths[0] for length in lengths[1:]):
            raise ValueError(f"Datasets should all have the same length, instead got {lengths}")
        self.datasets = datasets
        self.which = which

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return self.datasets[self.which[idx]][idx]


def get_dataset(data_dirs, weights=None, train: bool = True):
    datasets = [MNISTLike(data_dir, train=train) for data_dir in data_dirs]
    if len(datasets) > 1:
        if weights is not None:
            weights = [float(w) for w in weights]
            weights = np.array(weights) / np.sum(weights)
        which = np.random.choice(len(datasets), len(datasets[0]), p=weights)
        return InterleavedDataset(datasets, which)
    return datasets[0]
