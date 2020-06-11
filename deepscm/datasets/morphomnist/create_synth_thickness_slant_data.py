import numpy as np
import os
import pandas as pd
import pyro
import torch

from pyro.distributions import Gamma, Normal
from tqdm import tqdm

from deepscm.datasets.morphomnist import load_morphomnist_like, save_morphomnist_like
from deepscm.datasets.morphomnist.transforms import SetThickness, SetSlant, ImageMorphology


def model_(n_samples=None):
    with pyro.plate('observations', n_samples):
        thickness = pyro.sample('thickness', Gamma(10., 5.))

        loc = thickness * 6.
        slant = pyro.sample('slant', Normal(loc, 1.))

    return slant, thickness


def model(n_samples=None):
    with pyro.plate('observations', n_samples):
        thickness = pyro.sample('thickness', Gamma(10., 5.))

        loc = (thickness - 2.5) * 20
        slant = pyro.sample('slant', Normal(loc, 1.))

    return slant, thickness


def gen_dataset(args, train=True):
    pyro.clear_param_store()
    images, labels, _ = load_morphomnist_like(args.data_dir, train=train)
    mask = (labels == args.digit_class)
    images = images[mask]
    labels = labels[mask]

    n_samples = len(images)
    with torch.no_grad():
        slant, thickness = model(n_samples)

    metrics = pd.DataFrame(data={'thickness': thickness, 'slant': slant})

    for n, (slant, thickness) in enumerate(tqdm(zip(slant, thickness), total=n_samples)):
        morph = ImageMorphology(images[n], scale=16)
        tmp_img = np.float32(SetThickness(thickness)(morph))
        tmp_morph = ImageMorphology(tmp_img, scale=1)
        tmp_img = np.float32(SetSlant(np.deg2rad(slant))(tmp_morph))
        images[n] = morph.downscale(tmp_img)

    save_morphomnist_like(images, labels, metrics, args.out_dir, train=train)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/vol/biomedic/users/dc315/mnist/original/', help="Path to MNIST (default: %(default)s)")
    parser.add_argument('-o', '--out-dir', type=str, help="Path to store new dataset")
    parser.add_argument('-d', '--digit-class', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="digit class to select")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.txt'), 'w') as f:
        print(f'Generated data for:\n {args.__dict__}', file=f)

    print('Generating Training Set')
    print('#######################')
    gen_dataset(args, True)

    print('Generating Test Set')
    print('###################')
    gen_dataset(args, False)
