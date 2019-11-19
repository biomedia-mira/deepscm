import unittest

import torch

import util
from tests.base import TensorTestCase

N, I, J = 10, 4, 5


class TestUtil(TensorTestCase):
    shape = (I, J)
    dtype = torch.float32
    device = 'cpu'

    @classmethod
    def setUpClass(cls):
        shape = torch.Size(cls.shape)
        cls.rect_matrix = cls.gen_matrix(shape)  # (..., I, J)
        cls.vector = cls.gen_matrix(shape[:-2] + (shape[-1],))  # (..., J)
        cls.posdef_matrix = cls.gen_posdef_matrix(shape[:-2] + (shape[-1], shape[-1]))  # (..., J, J)
        cls.tril_matrix = torch.cholesky(cls.posdef_matrix, upper=False)  # (..., J, J)

    def test_matvec_output_shape(self):
        self.assertEqual(util.matvec(self.rect_matrix, self.vector).shape,
                         self.rect_matrix.shape[:-1])

    def test_outer_output_shape(self):
        vec1 = self.vector
        vec2 = self.vector[..., :-1]
        self.assertEqual(util.outer(vec1, vec2).shape, vec1.shape + vec2.shape[-1:])

    def test_inverse_cholesky_output_shape(self):
        self.assertEqual(util.inverse_cholesky(self.posdef_matrix).shape,
                         self.posdef_matrix.shape)

    def test_triangular_logdet_output_shape(self):
        self.assertEqual(util.triangular_logdet(self.tril_matrix).shape,
                         self.tril_matrix.shape[:-2])

    def test_posdef_logdet_output_shape(self):
        self.assertEqual(util.posdef_logdet(self.posdef_matrix)[0].shape,
                         self.posdef_matrix.shape[:-2])

    def test_posdef_solve_output_shape(self):
        b = self.vector.unsqueeze(-1)
        sol, tril = util.posdef_solve(b, self.posdef_matrix)
        self.assertEqual(sol.shape, b.shape)
        self.assertEqual(tril.shape, self.posdef_matrix.shape)

        B = self.rect_matrix.transpose(-1, -2)
        sol, tril = util.posdef_solve(B, self.posdef_matrix)
        self.assertEqual(sol.shape, B.shape)
        self.assertEqual(tril.shape, self.posdef_matrix.shape)

    def test_posdef_inverse_output_shape(self):
        inv, tril = util.posdef_inverse(self.posdef_matrix)
        self.assertEqual(inv.shape, self.posdef_matrix.shape)
        self.assertEqual(tril.shape, self.posdef_matrix.shape)

    def test_mahalanobis_output_shape(self):
        maha, tril = util.mahalanobis(self.posdef_matrix, self.vector)
        self.assertEqual(maha.shape, self.vector.shape[:-1])
        self.assertEqual(tril.shape, self.posdef_matrix.shape)

    def test_triangular_logdet_correctness(self):
        self.assertAllClose(util.triangular_logdet(self.tril_matrix),
                            torch.logdet(self.tril_matrix))

    def test_posdef_logdet_correctness(self):
        self.assertAllClose(util.posdef_logdet(self.posdef_matrix)[0],
                            torch.logdet(self.posdef_matrix))

    def test_posdef_inverse_correctness(self):
        inv = util.posdef_inverse(self.posdef_matrix)[0]
        eye = util.eye_like(self.posdef_matrix)
        self.assertAllClose(inv @ self.posdef_matrix, eye)
        self.assertAllClose(self.posdef_matrix @ inv, eye)
        self.assertAllClose(inv, torch.inverse(self.posdef_matrix))

    def test_posdef_solve_correctness(self):
        b = self.vector.unsqueeze(-1)
        sol = util.posdef_solve(b, self.posdef_matrix)[0]
        self.assertAllClose(self.posdef_matrix @ sol, b)
        self.assertAllClose(sol, torch.solve(b, self.posdef_matrix)[0])

        B = self.rect_matrix.transpose(-1, -2)
        sol = util.posdef_solve(B, self.posdef_matrix)[0]
        self.assertAllClose(self.posdef_matrix @ sol, B)
        self.assertAllClose(sol, torch.solve(B, self.posdef_matrix)[0])

    def test_cholesky_inverse_correctness(self):
        self.assertAllClose(self.tril_matrix @ self.tril_matrix.transpose(-2, -1),
                            self.posdef_matrix)
        inv = util.cholseky_inverse(self.tril_matrix)
        eye = util.eye_like(self.posdef_matrix)
        self.assertAllClose(inv @ self.posdef_matrix, eye)
        self.assertAllClose(self.posdef_matrix @ inv, eye)
        self.assertAllClose(inv, torch.inverse(self.posdef_matrix))

    def test_mahalanobis_correctness(self):
        maha = util.mahalanobis(self.posdef_matrix, self.vector)[0]
        maha_explicit = (self.vector * torch.solve(self.vector.unsqueeze(-1),
                                                   self.posdef_matrix)[0].squeeze(-1)).sum(-1)
        self.assertAllClose(maha, maha_explicit)


class TestUtilBatched(TestUtil):
    shape = (N, I, J)


class TestUtilDouble(TestUtil):
    shape = (I, J)
    dtype = torch.float64
    atol = 1e-12


class TestUtilDoubleBatched(TestUtil):
    shape = (N, I, J)
    dtype = torch.float64
    atol = 1e-12


# class TestUtilGPU(TestUtil):
#     device = 'cuda:0'
#
#
# class TestUtilBatchedGPU(TestUtil):
#     device = 'cuda:0'
#     shape = (N, I, J)


if __name__ == '__main__':
    unittest.main()
