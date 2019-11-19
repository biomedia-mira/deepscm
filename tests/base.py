import unittest

import torch


def _check_posdef(matrix: torch.Tensor):
    return (torch.symeig(matrix, eigenvectors=False)[0] <= 0.).sum() == 0


class TensorTestCase(unittest.TestCase):
    dtype = torch.float32
    device = 'cpu'
    rtol = 1e-5
    atol = 1e-6

    @classmethod
    def gen_matrix(cls, shape):
        return torch.randn(shape, dtype=cls.dtype, device=cls.device)

    @classmethod
    def gen_pos_matrix(cls, shape):
        return torch.rand(shape, dtype=cls.dtype, device=cls.device).exp_()

    @classmethod
    def gen_posdef_matrix(cls, shape):
        if shape[-1] != shape[-2]:
            raise ValueError(f"Positive definite matrices must be square; got shape {shape}")
        A = cls.gen_matrix(shape)
        S = A.transpose(-1, -2) @ A + torch.eye(shape[-1], dtype=cls.dtype, device=cls.device)
        assert _check_posdef(S)
        return S

    def assertPositiveDefinite(self, matrix: torch.Tensor, msg=None):
        self.assertTrue(_check_posdef(matrix), msg)

    def assertAllClose(self, tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=None, atol=None, msg=None):
        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol
        self.assertTrue(torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), msg)

    def assertBroadcastableTo(self, tensor: torch.Tensor, shape: torch.Size, msg=None):
        try:
            tensor.expand(shape)
        except RuntimeError:
            self.fail(msg)
