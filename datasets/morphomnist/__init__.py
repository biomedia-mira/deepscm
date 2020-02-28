import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from morphomnist import io


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
        prefix = "train" if train else "t10k"
        images_filename = prefix + "-images-idx3-ubyte.gz"
        labels_filename = prefix + "-labels-idx1-ubyte.gz"
        metrics_filename = prefix + "-morpho.csv"
        self.images = torch.as_tensor(io.load_idx(os.path.join(self.root_dir, images_filename)))
        self.labels = torch.as_tensor(io.load_idx(os.path.join(self.root_dir, labels_filename)))
        metrics_df = pd.read_csv(os.path.join(root_dir, metrics_filename), usecols=columns)
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
