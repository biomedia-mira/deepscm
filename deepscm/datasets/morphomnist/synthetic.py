import multiprocessing

import numpy as np
import pandas as pd
from deepscm.morphomnist.measure import measure_image, Morphometrics
from deepscm.morphomnist.morpho import ImageMorphology

from deepscm.datasets.morphomnist import load_morphomnist_like, save_morphomnist_like, transforms


def subsample(filter_fn, source_dir, train: bool, target_dir):
    images, labels, metrics = load_morphomnist_like(source_dir, train, source_dir)
    idx = filter_fn(metrics)
    save_morphomnist_like(images[idx], labels[idx], metrics[idx], train, target_dir)


class _PackedArgsFn:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return self.fn(*args)


def _run_batched(fn, args, num: int, pool: multiprocessing.Pool = None, chunksize: int = 100):
    fn = _PackedArgsFn(fn)
    if pool is None:
        gen = map(fn, args)
    else:
        gen = pool.imap(fn, args, chunksize=chunksize)

    try:
        import tqdm
        gen = tqdm.tqdm(gen, total=num, unit='img', ascii=True)
    except ImportError:
        def plain_progress(g):
            print(f"\rProcessing images: {0}/{num}", end='')
            for i, res in enumerate(g):
                print(f"\rProcessing images: {i + 1}/{num}", end='')
                yield res
            print()

        gen = plain_progress(gen)

    results = list(gen)
    return results


def apply_conditional_transformation(fn, images, labels, metrics,
                                     pool: multiprocessing.Pool = None, chunksize: int = 100):
    args = ((image, label, Morphometrics(*metrics_row[1]))
            for image, label, metrics_row
            in zip(images, labels, metrics.iterrows()))

    results = _run_batched(fn, args, len(images), pool, chunksize)
    trf_images, trf_labels, trf_metrics = zip(*results)
    trf_images = np.asarray(trf_images)
    trf_labels = np.asarray(trf_labels)
    trf_metrics = pd.DataFrame(trf_metrics)
    print(trf_metrics)
    return trf_images, trf_labels, trf_metrics


def example_fn(image, label, metrics: Morphometrics):
    morph = ImageMorphology(image, scale=4)
    # target_slant = np.deg2rad(30 if label % 2 == 0 else -30)
    # trf = transforms.SetSlant(target_slant)
    # target_thickness = 6 if label % 2 == 0 else 1.5
    # trf = transforms.SetThickness(target_thickness)
    target_thickness = max(1, 2.5 * np.exp(2*metrics.slant))
    trf = transforms.SetThickness(target_thickness)
    trf_image = morph.downscale(trf(morph))
    trf_metrics = measure_image(trf_image, scale=4, verbose=False)
    return trf_image, label, trf_metrics


if __name__ == '__main__':
    import os

    root_dir = "/vol/biomedic/users/dc315/mnist"
    source_dir = os.path.join(root_dir, "original")
    target_dir = os.path.join(root_dir, "sub_th3_sl0")
    images, labels, metrics = load_morphomnist_like(source_dir, train=False)
    # with multiprocessing.Pool() as pool:
    pool = None
    nrow, ncol = 9, 12
    trf_images, trf_labels, trf_metrics = apply_conditional_transformation(
        example_fn, images[:nrow * ncol], labels, metrics, pool)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrow, ncol)
    for i, ax in enumerate(axs.flat):
        if i >= len(trf_images):
            break
        ax.imshow(trf_images[i], cmap='gray_r')
        # ax.set_title(f"label: {trf_labels[i]}")
        ax.axis('off')
    plt.show()
    # for x in metrics.iterrows():
    #     print(x)
    # subsample(lambda metrics: (metrics.thickness > 3.) & (metrics.slant < 0.),
    #           source_dir, True, target_dir)
