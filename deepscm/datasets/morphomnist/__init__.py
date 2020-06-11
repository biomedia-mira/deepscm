import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from deepscm.morphomnist import io


def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    metrics_filename = prefix + "-morpho.csv"
    images_path = os.path.join(root_dir, images_filename)
    labels_path = os.path.join(root_dir, labels_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)
    return images_path, labels_path, metrics_path


def load_morphomnist_like(root_dir, train: bool = True, columns=None) \
        -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Args:
        root_dir: path to data directory
        train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
        columns: list of morphometrics to load; by default (``None``) loads the image index and
            all available metrics: area, length, thickness, slant, width, and height
    Returns:
        images, labels, metrics
    """
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = io.load_idx(images_path)
    labels = io.load_idx(labels_path)

    if columns is not None and 'index' not in columns:
        usecols = ['index'] + list(columns)
    else:
        usecols = columns
    metrics = pd.read_csv(metrics_path, usecols=usecols, index_col='index')
    return images, labels, metrics


def save_morphomnist_like(images: np.ndarray, labels: np.ndarray, metrics: pd.DataFrame,
                          root_dir, train: bool):
    """
    Args:
        images: array of MNIST-like images
        labels: array of class labels
        metrics: data frame of morphometrics
        root_dir: path to the target data directory
        train: whether to save as the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
    """
    assert len(images) == len(labels)
    assert len(images) == len(metrics)
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    os.makedirs(root_dir, exist_ok=True)
    io.save_idx(images, images_path)
    io.save_idx(labels, labels_path)
    metrics.to_csv(metrics_path, index_label='index')


class MorphoMNISTLike(Dataset):
    def __init__(self, root_dir, train: bool = True, columns=None):
        """
        Args:
            root_dir: path to data directory
            train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
                subset (``False``, ``'t10k-*'`` files)
            columns: list of morphometrics to load; by default (``None``) loads the image index and
                all available metrics: area, length, thickness, slant, width, and height
        """
        self.root_dir = root_dir
        self.train = train
        images, labels, metrics_df = load_morphomnist_like(root_dir, train, columns)
        self.images = torch.as_tensor(images)
        self.labels = torch.as_tensor(labels)
        if columns is None:
            columns = metrics_df.columns
        self.metrics = {col: torch.as_tensor(metrics_df[col]) for col in columns}
        self.columns = columns
        assert len(self.images) == len(self.labels) and len(self.images) == len(metrics_df)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {col: values[idx] for col, values in self.metrics.items()}
        item['image'] = self.images[idx]
        item['label'] = self.labels[idx]
        return item


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # Example usage
    dataset = MorphoMNISTLike(root_dir="/vol/biomedic/users/dc315/mnist/original",
                              columns=['slant', 'thickness'], train=False)
    print(dataset.columns)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch in data_loader:
        print(batch)
        break
