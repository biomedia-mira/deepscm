import numpy as np
import torch


def make_grid_image(size, spacing: int, dtype=torch.double, device=None):
    img = torch.zeros([1, 1, *size], dtype=dtype, device=device)
    img[:, :, ::spacing, :] = 1
    img[:, :, :, ::spacing] = 1
    return img


def vector_field_to_hsv(u, dim=-3, mag_max=None):
    """Maps a vector field to hue, saturation, and value (HSV) for visualisation.

    Hue corresponds to vector orientation, saturation, to magnitude, and value is set to 1.
    Use :func:`hsv_to_rgb` or :func:`skimage.color.hsv2rgb` (if available) to map to RGB colours.

    Args:
        u (torch.Tensor): vector field.
        dim (int): index of ``u``'s dimension representing the vector components
            (default: ``-3``, for shape ``[..., 2, height, width]``).
        mag_max (float): maximum value to be used when normalising the vector field's magnitudes
            (default: ``None``, uses computed maximum).

    Returns:
        (torch.Tensor): hue, saturation, and value components, shaped as
            ``[..., height, width, 3]`` and all in [0, 1].
    """
    dy, dx = torch.unbind(u, dim=dim)
    h = (.5 * torch.atan2(dy, dx) / np.pi) % 1.  # map orientation to [0, 1]
    mag = torch.norm(u, dim=dim)
    if mag_max is None:
        mag_max = mag.max()
    s = torch.clamp(mag / mag_max, 0., 1.)
    v = torch.ones_like(dx)
    return torch.stack([h, s, v], -1)


def hsv_to_rgb(hsv):
    """Adapted from :func:`skimage.color.hsv2rgb`.

    Args:
        hsv (torch.Tensor): HSV tensor of shape ``[..., height, width, 3]``, with values in [0, 1].

    Returns:
        (torch.Tensor): RGB tensor of shape ``[..., height, width, 3]``, with values in [0, 1].
    """
    h, s, v = torch.unbind(hsv, -1)
    hi = torch.floor(h * 6.)
    f = h * 6. - hi
    p = v * (1. - s)
    q = v * (1. - f * s)
    t = v * (1. - (1. - f) * s)

    hi = torch.stack([hi, hi, hi], -1).long() % 6

    # np.choose supports torch.Tensor
    out = np.choose(hi, [torch.stack([v, t, p], -1),
                         torch.stack([q, v, p], -1),
                         torch.stack([p, v, t], -1),
                         torch.stack([p, q, v], -1),
                         torch.stack([t, p, v], -1),
                         torch.stack([v, p, q], -1)])
    return out


def vector_field_to_rgb(u, dim=-3, mag_max=None):
    return hsv_to_rgb(vector_field_to_hsv(u, dim, mag_max))
