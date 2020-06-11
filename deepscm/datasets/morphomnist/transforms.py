import numpy as np
from skimage import morphology, transform

from deepscm.morphomnist.morpho import ImageMoments, ImageMorphology, bounding_parallelogram
from deepscm.morphomnist.perturb import Deformation, Perturbation


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


class LinearDeformation(Deformation):
    def _get_matrix(self, moments: ImageMoments, morph: ImageMorphology) -> np.ndarray:
        raise NotImplementedError

    def warp(self, xy: np.ndarray, morph: ImageMorphology) -> np.ndarray:
        moments = ImageMoments(morph.binary_image)
        centroid = np.array(moments.centroid)
        matrix = self._get_matrix(moments, morph)
        xy_ = (xy - centroid) @ matrix.T + centroid
        return xy_


class SetSlant(LinearDeformation):
    def __init__(self, target_slant_rad: float):
        self.target_shear = -np.tan(target_slant_rad)

    def _get_matrix(self, moments: ImageMoments, morph: ImageMorphology) -> np.ndarray:
        source_shear = moments.horizontal_shear
        delta = self.target_shear - source_shear
        return np.array([[1., -delta], [0., 1.]])


def _measure_width(morph: ImageMorphology, frac=.02, moments: ImageMoments = None):
    top_left, top_right = bounding_parallelogram(morph.hires_image,
                                                 frac=frac, moments=moments)[:2]
    return (top_right[0] - top_left[0]) / morph.scale


class SetWidth(LinearDeformation):
    _tolerance = 1.

    def __init__(self, target_width: float, validate=False):
        self.target_width = target_width
        self._validate = validate

    def _get_matrix(self, moments: ImageMoments, morph: ImageMorphology) -> np.ndarray:
        source_width = _measure_width(morph, moments=moments)
        factor = source_width / self.target_width
        shear = moments.horizontal_shear
        return np.array([[factor, shear * (1. - factor)], [0., 1.]])

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        pert_hires_image = super().__call__(morph)
        pert_image = morph.downscale(pert_hires_image)
        if self._validate:
            pert_morph = ImageMorphology(pert_image, threshold=morph.threshold, scale=morph.scale)
            width = _measure_width(pert_morph)
            if abs(width - self.target_width) > self._tolerance:
                print(f"!!! Incorrect width after transformation: {width:.1f}, "
                      f"expected {self.target_width:.1f}.")
                pert_hires_image = self(pert_morph)
        return pert_hires_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from morphomnist import io, measure

    image_path = "/vol/biomedic/users/dc315/mnist/original/t10k-images-idx3-ubyte.gz"
    images = io.load_idx(image_path)

    # pert = SetThickness(3.5)
    # pert = SetSlant(np.deg2rad(30))
    pert = SetWidth(8, validate=True)
    for n in range(20):
        print(n, "before:")
        measure.measure_image(images[n], verbose=True)

        morph = ImageMorphology(images[n], scale=16)
        pert_hires_image = pert(morph)
        pert_image = morph.downscale(pert_hires_image)

        print(n, "after:")
        measure.measure_image(pert_image, verbose=True)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(images[n], cmap='gray_r')
        ax2.imshow(pert_image, cmap='gray_r')
        ax1.axis('off')
        ax2.axis('off')
        plt.show()
        print()
