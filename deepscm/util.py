import torch


def matvec(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return torch.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)


def outer(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    return vec1.unsqueeze(-1) * vec2.unsqueeze(-2)


def eye_like(A: torch.Tensor) -> torch.Tensor:
    return torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A)


def inverse_cholesky(A: torch.Tensor) -> torch.Tensor:
    r"""Computes the Cholesky decomposition of the inverse: :math:`A^{-1} = LL^\top`.

    Adapted from torch.distributions.multivariate_normal, inspired by:

    - https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006
    - https://math.stackexchange.com/q/1434899
    """
    Lf = torch.cholesky(A.flip(-2, -1), upper=False)
    L_inv = Lf.flip(-2, -1).transpose(-2, -1)
    L = torch.triangular_solve(eye_like(A), L_inv, upper=False)[0]
    return L


def triangular_logdet(tri: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(tri, dim1=-2, dim2=-1).log().sum(-1)


def posdef_logdet(A: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    tril = torch.cholesky(A, upper=False)
    return 2. * triangular_logdet(tril), tril


def posdef_solve(b: torch.Tensor, A: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    r"""Computes batched :math:`A^{-1} b`, assuming :math:`A` is positive definite.

    Adapted from: https://github.com/pytorch/pytorch/issues/4669#issuecomment-520919549
    """
    tril = torch.cholesky(A, upper=False)
    x = torch.triangular_solve(b, tril, upper=False)[0]
    return torch.triangular_solve(x, tril, upper=False, transpose=True)[0], tril


def posdef_inverse(A: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    r"""Computes batched :math:`A^{-1}`, assuming :math:`A` is positive definite.

    Adapted from: https://github.com/pytorch/pytorch/issues/4669#issuecomment-520919549
    """
    return posdef_solve(eye_like(A), A)


def cholseky_inverse(tril: torch.Tensor) -> torch.Tensor:
    r"""Computes batched :math:`(LL^\top)^{-1}`, assuming :math:`L` is a lower-triangular Cholesky
    factor.
    """
    b = eye_like(tril)
    x = torch.triangular_solve(b, tril, upper=False)[0]
    return torch.triangular_solve(x, tril, upper=False, transpose=True)[0]


def symmetrise(A):
    return .5 * (A + A.transpose(-1, -2))


def mahalanobis(A, b):
    r"""Computes batched :math:`b^\top A^{-1} b`, assuming :math:`A` is positive definite."""
    tril = torch.cholesky(A, upper=False)
    x = torch.triangular_solve(b.unsqueeze(-1), tril, upper=False)[0].squeeze(-1)
    return (x * x).sum(-1), tril
