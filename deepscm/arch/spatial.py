import torch
import torch.nn.functional as F
from torch import nn


def moveaxis(x: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    ndims = x.dim()
    dims = list(range(ndims))
    dims.pop(src)
    if dst < 0:
        dst = ndims + dst
    dims.insert(dst, src)
    return x.permute(dims)


def get_grid(size):
    grids = torch.meshgrid([torch.arange(s) for s in size])
    return torch.stack(grids, 0).float()


def transform(src, new_locs, interpolation='bilinear'):
    size = new_locs.shape[2:]
    ndim = len(size)
    assert ndim == new_locs.shape[1]

    for i in range(ndim):
        new_locs[:, i] = 2. * new_locs[:, i] / (size[i] - 1) - 1.

    new_locs = moveaxis(new_locs, 1, -1)  # [batch, z?, y, x, dims]
    new_locs = new_locs[..., list(range(ndim))[::-1]]  # reverse dims

    return F.grid_sample(src, new_locs, mode=interpolation)


class SpatialTransformer(nn.Module):
    """
    Spatial resampling block that uses a displacement field to preform a grid_sample
    (https://pytorch.org/docs/stable/nn.functional.html#grid-sample)

    Adapted from: https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self, size, interpolation='bilinear'):
        """
        Args:
            size (torch.Size): spatial dimensions ([z?, y, x])
            interpolation (`str`): {'nearest', 'bilinear'}
        """
        super().__init__()
        self.register_buffer('grid', get_grid(size))
        self.interpolation = interpolation

    def forward(self, src, disp):
        """
        Args:
            src (torch.Tensor): original moving image ([batch, channels, z?, y, x])
            disp (torch.Tensor): displacement field ([batch, dims, z?, y, x])
        """
        return transform(src, self.grid + disp, interpolation=self.interpolation)


def integrate_vel(vel, n_steps: int = 7, logdet_jac=False, interpolation='bilinear',
                  transformer: SpatialTransformer = None):
    """Integrates a stationary velocity field via scaling-and-squaring

    Partially adapted from:
    https://github.com/adalca/neuron/blob/master/neuron/utils.py
    """
    if transformer is None:
        transformer = SpatialTransformer(vel.shape[2:], interpolation=interpolation)

    eps = 2 ** -n_steps
    disp = eps * vel

    if logdet_jac:
        jac = jacobian2d(vel)
        ldjac = logdet1p(jac, eps=eps, n_terms=4, dim1=-4, dim2=-3).unsqueeze(-3)
    else:
        ldjac = None

    for _ in range(n_steps):
        disp = disp + transformer(disp, disp)
        if logdet_jac:
            ldjac = ldjac + transformer(ldjac, disp)
    return disp, ldjac


class Integrator(nn.Module):
    def __init__(self, size, n_steps: int = 7, interpolation='bilinear', transformer=None):
        super().__init__()
        if transformer is None:
            transformer = SpatialTransformer(size, interpolation)
        self.transformer = transformer
        self.n_steps = n_steps

    def forward(self, vel, logdet_jac=False):
        """
        Args:
            vel (torch.Tensor):
            logdet_jac (bool):

        Returns:
            (torch.Tensor, torch.Tensor):
        """
        return integrate_vel(vel, self.n_steps, logdet_jac=logdet_jac, transformer=self.transformer)


def _sobel_kernel2d() -> torch.Tensor:
    dx = torch.tensor([[-1.,  0.,  1.],
                       [-2.,  0.,  2.],
                       [-1.,  0.,  1.]])
    dy = torch.tensor([[-1., -2., -1.],
                       [0.,  0.,  0.],
                       [1.,  2.,  1.]])
    return .125 * torch.stack([dy, dx, dy, dx], 0).unsqueeze(1)


_SOBEL_KERNEL2D = _sobel_kernel2d()


def jacobian2d(vel):
    """
    Args:
        vel (torch.Tensor): velocity field ([N, 2, H, W])

    Returns:
        (torch.Tensor): Jacobian field ([N, 2, 2, H, W])
    """
    padded_vel = F.pad(vel, [1, 1, 1, 1], mode='replicate')
    jac = F.conv2d(padded_vel, _SOBEL_KERNEL2D, padding=0, groups=2)
    return jac.reshape(vel.shape[0], 2, 2, *vel.shape[-2:])


def _make_einsum_spec(ndim, dim1, dim2):
    symbols = 'abcdefghijklmnopqrstuvwxyz'
    in1_spec = list(symbols[:ndim])
    in2_spec = list(symbols[:ndim])
    out_spec = list(symbols[:ndim])
    in1_spec[dim2] = in2_spec[dim1] = symbols[ndim]
    return '{},{}->{}'.format(*map(''.join, [in1_spec, in2_spec, out_spec]))


def logdet1p(x, dim1=-2, dim2=-1, n_terms=4, eps=1.):
    ld = 0.
    x_n = None
    einsum_spec = _make_einsum_spec(x.dim(), dim1, dim2)
    for n in range(1, n_terms + 1):
        x_n = x if n == 1 else torch.einsum(einsum_spec, x_n, x)
        trace_x_n = x_n.diagonal(dim1=dim1, dim2=dim2).sum(-1)
        ld = ld - (-eps) ** n * trace_x_n / n
    return ld


class DiffeomorphicTransformer2D(nn.Module):
    def __init__(self, size, n_steps: int = 7, interpolation='bilinear'):
        super().__init__()
        self.transformer = SpatialTransformer(size, interpolation)
        self.integrator = Integrator(n_steps, transformer=self.transformer)

    def forward(self, src, vel, logdet_jac: bool = False):
        disp, ldjac = self.integrator(vel, logdet_jac)
        warped = self.transformer(src, disp)
        return warped, ldjac


if __name__ == '__main__':
    def unet(x):
        return x

    diffeo = DiffeomorphicTransformer2D([28, 28])
    img = torch.randn(100, 2, 28, 28)  # [N, C, H, W]
    vel = unet(img)  # [N, 2, H, W]
    warped, ldjac = diffeo(img, vel, logdet_jac=True)  # [N, C, H, W]
