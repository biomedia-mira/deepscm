import math
import unittest

import torch
import scipy.stats as ss
from torch.autograd import gradcheck

from distributions.torch_wishart import Wishart, InverseWishart
from tests.base import TensorTestCase

N, K, D = 10, 3, 4


def set_rng_seed(seed):
    import random
    import numpy
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def _eval_ss(fcn, x=None):
    if x is None:
        return torch.as_tensor(fcn())
    return torch.as_tensor([fcn(x_) for x_ in x.numpy()])


class TestWishartBatched(TensorTestCase):
    dtype = torch.double
    batch_shape = (K,)
    sample_shape = (N, K)

    @classmethod
    def setUpClass(cls):
        # set_rng_seed(0)
        batch_shape = torch.Size(cls.batch_shape)
        sample_shape = torch.Size(cls.sample_shape)
        cls.df = cls.gen_pos_matrix(batch_shape) + D - 1
        cls.scale = cls.gen_posdef_matrix(batch_shape + (D, D))
        # scale_tril = torch.cholesky(cls.scale)
        # data = cls.gen_posdef_matrix(sample_shape + (D, D))
        # cls.data = scale_tril @ data @ scale_tril.transpose(-2, -1) / cls.df[..., None, None]
        cls.data = cls.gen_posdef_matrix(sample_shape + (D, D))

    def setUp(self):
        self.dist = Wishart(self.df, self.scale, validate_args=True)
        self.assertEqual(self.dist.batch_shape, self.batch_shape)
        self.assertEqual(self.dist.event_shape, (D, D))

    def test_log_normalizer_shape(self):
        self.assertEqual(self.dist.log_normalizer.shape, self.batch_shape)

    def test_log_prob_shape(self):
        self.assertEqual(self.dist.log_prob(self.data).shape, self.data.shape[:-2])

    def test_entropy_shape(self):
        self.assertEqual(self.dist.entropy().shape, self.batch_shape)

    def test_derived_params_shapes(self):
        mat_shape = self.dist.batch_shape + self.dist.event_shape
        self.assertEqual(self.dist.scale_tril.shape, mat_shape)
        self.assertEqual(self.dist.mean.shape, mat_shape)

    def test_sample_shape(self):
        sample_shape = torch.Size(self.sample_shape)[:-len(self.batch_shape)]
        self.assertEqual(self.dist.sample(sample_shape).shape,
                         sample_shape + self.dist.batch_shape + self.dist.event_shape)

    def test_expand(self):
        expanded_batch_shape = (K + 1,) + self.dist.batch_shape
        expanded_wishart = self.dist.expand(expanded_batch_shape)
        mat_shape = expanded_batch_shape + self.dist.event_shape
        self.assertEqual(expanded_wishart.df.shape, expanded_batch_shape)
        self.assertEqual(expanded_wishart.scale.shape, mat_shape)
        self.assertEqual(expanded_wishart.scale_tril.shape, mat_shape)
        self.assertEqual(expanded_wishart.batch_shape, expanded_batch_shape)
        self.assertEqual(expanded_wishart.event_shape, self.dist.event_shape)

    def test_expected_suff_stats(self):
        nparams = [p.detach().requires_grad_() for p in self.dist._natural_params]
        lg_normal = self.dist._log_normalizer(*nparams)
        self.assertAllClose(lg_normal, self.dist.log_normalizer)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        self.assertAllClose(gradients[0], self.dist._expected_logdet())
        self.assertAllClose(gradients[1], self.dist.mean)

    def assertSignificantlyClose(self, x, mean, std=None, alpha=1e-3):
        n = x.shape[0]
        avg = x.mean(0)
        if std is None:
            sem = x.std(0) / math.sqrt(n)
            crit = ss.t.isf(alpha, n - 1)
        else:
            sem = std / math.sqrt(n)
            crit = ss.norm.isf(alpha)
        score = (avg - mean) / sem
        self.assertTrue((score.abs() < crit).all())

    def test_statistics(self):
        set_rng_seed(0)
        n_samples = 10000
        samples = self.dist.sample((n_samples,))
        mean = self.dist.mean

        self.assertSignificantlyClose(samples.reshape(samples.shape[:-2] + (-1,)),
                                      mean.reshape(mean.shape[:-2] + (-1,)))

        sample_trils = torch.cholesky(samples)
        sample_logdets = 2. * sample_trils.diagonal(dim1=-1, dim2=-2).log().sum(-1)
        logdet_mean = self.dist._expected_logdet()
        # logdet_std = self.wishart._variance_logdet().sqrt()
        # self.assertSignificantlyClose(sample_logdets, logdet_mean, logdet_std)
        self.assertSignificantlyClose(sample_logdets, logdet_mean)

        sample_logprobs = self.dist.log_prob(samples)
        self.assertSignificantlyClose(-sample_logprobs, self.dist.entropy())

    def test_grad(self):
        scale = self.scale
        scale_tril = torch.cholesky(scale)

        def wishart_log_prob_gradcheck(df, scale=None, scale_tril=None):
            wishart_samples = Wishart(df, scale, scale_tril).sample().requires_grad_()

            def gradcheck_func(samples, df, scale, scale_tril):
                samples = .5 * (samples + samples.transpose(-1, -2))  # Ensure symmetry of samples
                if scale is not None:
                    scale = .5 * (scale + scale.transpose(-1, -2))  # Ensure symmetry of scale
                return Wishart(df, scale, scale_tril).log_prob(samples)

            gradcheck(gradcheck_func, (wishart_samples, df, scale, scale_tril), raise_exception=True)

        wishart_log_prob_gradcheck(self.df, scale, None)
        wishart_log_prob_gradcheck(self.df, None, scale_tril)


class TestWishart(TestWishartBatched):
    batch_shape = ()
    sample_shape = (N,)

    def setUp(self):
        self.dist = Wishart(self.df, self.scale, validate_args=True)
        self.ss_dist = ss.wishart(self.df.item(), self.scale.numpy())
        self.assertEqual(self.dist.batch_shape, self.batch_shape)
        self.assertEqual(self.dist.event_shape, (D, D))

    def test_log_prob_correctness(self):
        self.assertAllClose(self.dist.log_prob(self.data),
                            _eval_ss(self.ss_dist.logpdf, self.data).type(self.dtype))

    def test_entropy_correctness(self):
        self.assertAllClose(self.dist.entropy(),
                            _eval_ss(self.ss_dist.entropy).type(self.dtype))


class TestInverseWishartBatched(TestWishartBatched):
    @classmethod
    def setUpClass(cls):
        batch_shape = torch.Size(cls.batch_shape)
        sample_shape = torch.Size(cls.sample_shape)
        cls.df = cls.gen_pos_matrix(batch_shape) + D + 1
        cls.scale = cls.gen_posdef_matrix(batch_shape + (D, D))
        cls.data = cls.gen_posdef_matrix(sample_shape + (D, D))

    def setUp(self):
        self.dist = InverseWishart(self.df, self.scale, validate_args=True)
        self.assertEqual(self.dist.batch_shape, self.batch_shape)
        self.assertEqual(self.dist.event_shape, (D, D))

    def test_expected_suff_stats(self):
        nparams = [p.detach().requires_grad_() for p in self.dist._natural_params]
        lg_normal = self.dist._log_normalizer(*nparams)
        self.assertAllClose(lg_normal, self.dist.log_normalizer)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        self.assertAllClose(gradients[0], self.dist._expected_logdet())
        self.assertAllClose(gradients[1], self.dist._expected_inverse())

    def test_grad(self):
        scale = self.scale
        scale_tril = torch.cholesky(scale)

        def wishart_log_prob_gradcheck(df, scale=None, scale_tril=None):
            wishart_samples = InverseWishart(df, scale, scale_tril).sample().requires_grad_()

            def gradcheck_func(samples, df, scale, scale_tril):
                samples = .5 * (samples + samples.transpose(-1, -2))  # Ensure symmetry of samples
                if scale is not None:
                    scale = .5 * (scale + scale.transpose(-1, -2))  # Ensure symmetry of scale
                return InverseWishart(df, scale, scale_tril).log_prob(samples)

            gradcheck(gradcheck_func, (wishart_samples, df, scale, scale_tril), raise_exception=True)

        wishart_log_prob_gradcheck(self.df, scale, None)
        wishart_log_prob_gradcheck(self.df, None, scale_tril)


class TestInverseWishart(TestInverseWishartBatched):
    batch_shape = ()
    sample_shape = (N,)

    def setUp(self):
        self.dist = InverseWishart(self.df, self.scale, validate_args=True)
        self.ss_dist = ss.invwishart(self.df.item(), self.scale.numpy())
        self.assertEqual(self.dist.batch_shape, self.batch_shape)
        self.assertEqual(self.dist.event_shape, (D, D))

    def test_log_prob_correctness(self):
        self.assertAllClose(self.dist.log_prob(self.data),
                            _eval_ss(self.ss_dist.logpdf, self.data).type(self.dtype))
        wishart = Wishart(self.df, torch.inverse(self.scale))
        data_logprobs = self.dist.log_prob(self.data)
        data_logdets = 2. * torch.cholesky(self.data).diagonal(dim1=-1, dim2=-2).log().sum(-1)
        logdet_jacobian = (self.data.shape[-1] + 1) * data_logdets
        self.assertAllClose(data_logprobs + logdet_jacobian, wishart.log_prob(torch.inverse(self.data)))

    # def test_statistics(self):
    #     # set_rng_seed(0)
    #     n_samples = 10000
    #     samples = self.wishart.sample((n_samples,))
    #     # ss_samples = self.ss_wishart.rvs(n_samples)
    #     mean = self.wishart.mean
    #
    #     # print(samples.mean(0))
    #     # print(ss_samples.mean(0))
    #     # print(mean)
    #     # print(self.ss_wishart.mean())
    #     self.assertSignificantlyClose(samples.reshape(samples.shape[:-2] + (-1,)),
    #                                   mean.reshape(mean.shape[:-2] + (-1,)))
    #     # sample_trils = torch.cholesky(samples)
    #     # sample_logdets = 2. * sample_trils.diagonal(dim1=-1, dim2=-2).log().sum(-1)
    #     # logdet_mean = self.wishart._expected_logdet()
    #     # logdet_std = self.wishart._variance_logdet().sqrt()
    #     # self.assertSignificantlyClose(sample_logdets, logdet_mean, logdet_std)


if __name__ == '__main__':
    unittest.main()
