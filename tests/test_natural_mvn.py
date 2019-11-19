import unittest

import torch
from torch.distributions import MultivariateNormal

from distributions.natural_mvn import NaturalMultivariateNormal
from tests.base import TensorTestCase

N, K, D = 20, 10, 5


class TestNaturalMultivariateNormal(TensorTestCase):
    batch_shape = ()
    sample_shape = (N,)

    @classmethod
    def setUpClass(cls):
        batch_shape = torch.Size(cls.batch_shape)
        sample_shape = torch.Size(cls.sample_shape)
        cls.eta1 = cls.gen_matrix(batch_shape + (D,))
        cls.eta2 = -cls.gen_posdef_matrix(batch_shape + (D, D))
        cls.data = cls.gen_matrix(sample_shape + (D,))

    def setUp(self):
        self.nmvn = NaturalMultivariateNormal(self.eta1, self.eta2, validate_args=True)
        self.assertEqual(self.nmvn.batch_shape, self.batch_shape)
        self.assertEqual(self.nmvn.event_shape, (D,))

    def test_standard_conversion(self):
        nat2std = self.nmvn.to_standard()
        nat2std2nat = NaturalMultivariateNormal.from_standard(nat2std)
        nat2std2nat2std = nat2std2nat.to_standard()

        self.assertIsInstance(nat2std, MultivariateNormal)
        self.assertIsInstance(nat2std2nat, NaturalMultivariateNormal)
        self.assertIsInstance(nat2std2nat2std, MultivariateNormal)

        self.assertAllClose(self.nmvn.nat_param1, nat2std2nat.nat_param1)
        self.assertAllClose(self.nmvn.nat_param2, nat2std2nat.nat_param2)

        self.assertAllClose(nat2std.loc, nat2std2nat2std.loc)
        self.assertAllClose(nat2std.scale_tril, nat2std2nat2std.scale_tril)

    def test_log_normalizer_shape(self):
        self.assertEqual(self.nmvn.log_normalizer.shape, self.batch_shape)

    def test_log_prob_shape(self):
        self.assertEqual(self.nmvn.log_prob(self.data).shape, self.data.shape[:-1])

    def test_entropy_shape(self):
        self.assertEqual(self.nmvn.entropy().shape, self.batch_shape)

    def test_derived_params_shapes(self):
        vec_shape = self.nmvn.batch_shape + self.nmvn.event_shape
        mat_shape = vec_shape + self.nmvn.event_shape
        self.assertEqual(self.nmvn.precision_matrix.shape, mat_shape)
        self.assertEqual(self.nmvn.scale_tril.shape, mat_shape)
        self.assertEqual(self.nmvn.covariance_matrix.shape, mat_shape)
        self.assertEqual(self.nmvn.mean.shape, vec_shape)

    def test_rsample_shape(self):
        sample_shape = torch.Size(self.sample_shape)[:-len(self.batch_shape)]
        self.assertEqual(self.nmvn.sample(sample_shape).shape,
                         sample_shape + self.nmvn.batch_shape + self.nmvn.event_shape)

    def test_log_prob_correctness(self):
        std = self.nmvn.to_standard()
        self.assertAllClose(self.nmvn.log_prob(self.data), std.log_prob(self.data))

    def test_entropy_correctness(self):
        std = self.nmvn.to_standard()
        self.assertAllClose(self.nmvn.entropy(), std.entropy())

    def test_expand(self):
        expanded_batch_shape = (K + 1,) + self.nmvn.batch_shape
        expanded_nmvn = self.nmvn.expand(expanded_batch_shape)
        vec_shape = expanded_batch_shape + self.nmvn.event_shape
        mat_shape = vec_shape + self.nmvn.event_shape
        self.assertEqual(expanded_nmvn.nat_param1.shape, vec_shape)
        self.assertEqual(expanded_nmvn.nat_param2.shape, mat_shape)
        self.assertEqual(expanded_nmvn.batch_shape, expanded_batch_shape)
        self.assertEqual(expanded_nmvn.event_shape, self.nmvn.event_shape)


class TestNaturalMultivariateNormalBatched(TestNaturalMultivariateNormal):
    batch_shape = (K,)
    sample_shape = (N, K)


if __name__ == '__main__':
    unittest.main()
