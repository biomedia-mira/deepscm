import unittest

import torch
import scipy.stats as ss

from distributions.wishart import Wishart
from tests.base import TensorTestCase

N, K, D = 20, 10, 5


def _eval_ss(fcn, x=None):
    if x is None:
        return torch.as_tensor(fcn())
    return torch.as_tensor([fcn(x_) for x_ in x.numpy()])


class TestWishartBatched(TensorTestCase):
    batch_shape = (K,)
    sample_shape = (N, K)

    @classmethod
    def setUpClass(cls):
        batch_shape = torch.Size(cls.batch_shape)
        sample_shape = torch.Size(cls.sample_shape)
        cls.a = cls.gen_pos_matrix(batch_shape) + .5 * (D - 1)
        cls.B = cls.gen_posdef_matrix(batch_shape + (D, D))
        cls.data = cls.gen_posdef_matrix(sample_shape + (D, D))

    def setUp(self):
        self.wishart = Wishart(self.a, self.B, validate_args=True)
        # self.ss_wishart = ss.wishart(2. * self.a.numpy(), torch.inverse(2. * self.B).numpy())
        self.assertEqual(self.wishart.batch_shape, self.batch_shape)
        self.assertEqual(self.wishart.event_shape, (D, D))

    def test_log_normalizer_shape(self):
        self.assertEqual(self.wishart.log_normalizer.shape, self.batch_shape)

    def test_log_prob_shape(self):
        self.assertEqual(self.wishart.log_prob(self.data).shape, self.data.shape[:-2])

    def test_entropy_shape(self):
        self.assertEqual(self.wishart.entropy().shape, self.batch_shape)

    def test_derived_params_shapes(self):
        mat_shape = self.wishart.batch_shape + self.wishart.event_shape
        self.assertEqual(self.wishart.scale_tril.shape, mat_shape)
        self.assertEqual(self.wishart.mean.shape, mat_shape)

    def test_sample_shape(self):
        sample_shape = torch.Size(self.sample_shape)[:-len(self.batch_shape)]
        self.assertEqual(self.wishart.sample(sample_shape).shape,
                         sample_shape + self.wishart.batch_shape + self.wishart.event_shape)

    def test_expand(self):
        expanded_batch_shape = (K + 1,) + self.wishart.batch_shape
        expanded_wishart = self.wishart.expand(expanded_batch_shape)
        mat_shape = expanded_batch_shape + self.wishart.event_shape
        self.assertEqual(expanded_wishart.concentration.shape, expanded_batch_shape)
        self.assertEqual(expanded_wishart.scale.shape, mat_shape)
        self.assertEqual(expanded_wishart.batch_shape, expanded_batch_shape)
        self.assertEqual(expanded_wishart.event_shape, self.wishart.event_shape)


class TestWishart(TestWishartBatched):
    batch_shape = ()
    sample_shape = (N,)

    def setUp(self):
        self.wishart = Wishart(self.a, self.B, validate_args=True)
        self.ss_wishart = ss.wishart(2. * self.a.numpy(), torch.inverse(2. * self.B).numpy())
        self.assertEqual(self.wishart.batch_shape, self.batch_shape)
        self.assertEqual(self.wishart.event_shape, (D, D))

    def test_log_prob_correctness(self):
        self.assertAllClose(self.wishart.log_prob(self.data),
                            _eval_ss(self.ss_wishart.logpdf, self.data))

    def test_entropy_correctness(self):
        self.assertAllClose(self.wishart.entropy(),
                            _eval_ss(self.ss_wishart.entropy))


if __name__ == '__main__':
    unittest.main()
