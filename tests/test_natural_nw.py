import unittest

import torch

from distributions.natural_nw import NaturalNormalWishart
from tests.base import TensorTestCase
from util import outer

N, K, D = 20, 10, 5


class TestNaturalNormalWishart(TensorTestCase):
    batch_shape = ()
    sample_shape = (N,)

    @classmethod
    def setUpClass(cls):
        batch_shape = torch.Size(cls.batch_shape)
        sample_shape = torch.Size(cls.sample_shape)
        # cls.dof = 2. * D + 1. + cls.gen_matrix(batch_shape).exp_()
        # cls.nu = cls.gen_matrix(batch_shape).exp_()
        # mean = cls.gen_matrix(batch_shape + (D,))
        # cls.lambda1 = cls.nu.unsqueeze(-1) * mean
        # B = cls.gen_posdef_matrix(batch_shape + (D, D))
        # cls.lambda2 = 2. * B + outer(cls.lambda1, mean)
        # cls.data = cls.gen_matrix(sample_shape + (D,))
        mean = cls.gen_matrix(batch_shape + (D,))
        nu = cls.gen_pos_matrix(batch_shape)
        a = .5 * (D - 1) + cls.gen_pos_matrix(batch_shape)
        B = cls.gen_posdef_matrix(batch_shape + (D, D))
        niw = NaturalNormalWishart.from_standard(mean, nu, a, B, validate_args=True)
        cls.dof = niw.dof
        cls.lambda1 = niw.lambda1
        cls.lambda2 = niw.lambda2
        cls.nu = niw.nu

        cls.data = cls.gen_matrix(sample_shape + (D,))

    def setUp(self):
        self.niw = NaturalNormalWishart(self.dof, self.lambda1, self.lambda2, self.nu,
                                        validate_args=True)
        self.assertEqual(self.niw.batch_shape, self.batch_shape)
        self.assertEqual(self.niw.event_shape, (D,))

    def test_standard_conversion(self):
        nat2std = self.niw.to_standard()
        nat2std2nat = NaturalNormalWishart.from_standard(*nat2std, validate_args=True)
        nat2std2nat2std = nat2std2nat.to_standard()

        for param_name in ['dof', 'lambda1', 'lambda2', 'nu']:
            self.assertAllClose(getattr(self.niw, param_name), getattr(nat2std2nat, param_name))
        for (param1, param2) in zip(nat2std, nat2std2nat2std):
            self.assertAllClose(param1, param2)

    def test_expected_stats_shapes(self):
        expec_eta1, expec_eta2, expec_lognorm = self.niw.expected_stats()
        self.assertEqual(expec_eta1.shape, self.batch_shape + (D,))
        self.assertEqual(expec_eta2.shape, self.batch_shape + (D, D))
        self.assertEqual(expec_lognorm.shape, self.batch_shape)


class TestNaturalNormalWishartBatched(TestNaturalNormalWishart):
    batch_shape = (K,)
    sample_shape = (N, K)


if __name__ == '__main__':
    unittest.main()
