from torch.utils.data.dataset import Dataset
from skimage.io import imread
import numpy as np
import pandas as pd
import os

import torch
import torchvision as tv


class UKBBDataset(Dataset):
    def __init__(self, csv_path, base_path='/vol/biomedic2/bglocker/gemini/UKBB/t0/', crop_type=None, crop_size=(192, 192), downsample: int = None):
        super().__init__()
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)
        self.num_items = len(df)
        self.metrics = {col: torch.as_tensor(df[col]).float() for col in df.columns}
        self.base_path = base_path

        self.crop_type = crop_type
        self.crop_size = crop_size

        self.downsample = downsample

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        item = {col: values[index] for col, values in self.metrics.items()}

        img_path = os.path.join(self.base_path, '{}_T1_unbiased_brain_rigid_to_mni.png'.format(int(item['eid'])))
        img = imread(img_path, as_gray=True)

        transform_list = []
        transform_list += [tv.transforms.ToPILImage()]
        if self.crop_type is not None:
            if self.crop_type == 'center':
                transform_list += [tv.transforms.CenterCrop(self.crop_size)]
            elif self.crop_type == 'random':
                transform_list += [tv.transforms.RandomCrop(self.crop_size)]
            else:
                raise ValueError('unknwon crop type: {}'.format(self.crop_type))

        if self.downsample is not None and self.downsample > 1:
            transform_list += [tv.transforms.Resize(np.array(self.crop_size) // self.downsample)]

        transform_list += [tv.transforms.ToTensor()]

        img = tv.transforms.Compose(transform_list)(img)

        item['image'] = img

        return item
