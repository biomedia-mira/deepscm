import numpy as np
from skimage import morphology, transform

from morphomnist.morpho import ImageMorphology
from morphomnist.perturb import Perturbation


def _get_disk(radius: int, scale: int):
    mag_radius = scale * radius
    mag_disk = morphology.disk(mag_radius, dtype=np.float64)
    disk = transform.pyramid_reduce(mag_disk, downscale=scale, order=1, multichannel=False)
    return disk


class SetThickness(Perturbation):
    _disk_cache = {}

    def __init__(self, target_thickness: float):
        self.target_thickness = target_thickness

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        delta = self.target_thickness - morph.mean_thickness
        radius = int(morph.scale * abs(delta) / 2.)
        if radius in self._disk_cache:
            disk = self._disk_cache[radius]
        else:
            disk = _get_disk(radius, scale=16)
            self._disk_cache[radius] = disk
        img = morph.binary_image
        if delta >= 0:
            return morphology.dilation(img, disk)
        else:
            return morphology.erosion(img, disk)


if __name__ == '__main__':
    from morphomnist import io, measure

    image_path = "/vol/biomedic/users/dc315/mnist/original/t10k-images-idx3-ubyte.gz"
    images = io.load_idx(image_path)

    pert = SetThickness(3.5)
    for n in range(20):
        print(n, "before:")
        measure.measure_image(images[n], verbose=True)

        morph = ImageMorphology(images[n], scale=16)
        pert_hires_image = pert(morph)
        pert_image = morph.downscale(pert_hires_image)
        # pert_morph = ImageMorphology(pert_image, scale=4)
        # print(f"[{n}] before: {morph.mean_thickness:.3f}; after: {pert_morph.mean_thickness:.3f}")

        print(n, "after:")
        measure.measure_image(pert_image, verbose=True)
        print()
