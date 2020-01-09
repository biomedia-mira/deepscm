import unittest
from typing import Sequence

import torch
import torch.distributions as td

from distributions.factorised import Factorised
from distributions.multivariate import MultivariateDistribution
from distributions.mvn import MultivariateNormal
from models.mixture import MultivariateMixture


class TestFactorised(unittest.TestCase):
    num_factors = 5
    total_ndims = num_factors * 2
    num_samples = 3

    def setUp(self):
        self.dist = Factorised([td.MultivariateNormal(torch.zeros(2), torch.eye(2))
                                for _ in range(self.num_factors)])
        self.assertEqual(self.dist.event_shape, (self.total_ndims,))
        self.values = self.dist.sample((self.num_samples,))
        self.assertEqual(self.values.shape, (self.num_samples, self.total_ndims))

    def test_num_variables(self):
        self.assertEqual(self.dist.num_variables, len(self.dist.variable_shapes))

    def test_marginalise_single(self):
        var_shapes = self.dist.variable_shapes
        var_index = 0
        marg = self.dist.marginalise(var_index)
        self.assertEqual(marg.batch_shape, self.dist.batch_shape)
        var_dims = var_shapes[var_index]
        expected_event_shape = (var_dims,) if var_dims > 1 else ()
        self.assertEqual(marg.event_shape, expected_event_shape)

    def test_marginalise_multi(self):
        var_shapes = self.dist.variable_shapes
        for var_indices in [(0,), [0], (0, 1), list(range(self.dist.num_variables))]:
            marg = self.dist.marginalise(var_indices)
            self.assertIsInstance(marg, MultivariateDistribution)
            self.assertEqual(marg.batch_shape, self.dist.batch_shape)
            expected_event_shape = sum(var_shapes[i] for i in var_indices),
            self.assertEqual(marg.event_shape, expected_event_shape)

    def test_condition_shape(self):
        var_shapes = self.dist.variable_shapes
        for dims in [(0,), [0], (0, 1), list(range(self.dist.num_variables - 1))]:
            cond = self.dist.condition({dim: self.values[:, dim].unsqueeze(-1) for dim in dims})
            expected_batch_shape = self.values.shape[:-1] + self.dist.batch_shape
            cond_dims = sum(var_shapes[d] for d in dims) if isinstance(dims, Sequence) else \
                var_shapes[dims]
            expected_event_shape = (self.total_ndims - cond_dims,)
            self.assertEqual(cond.batch_shape, expected_batch_shape)
            self.assertEqual(cond.event_shape, expected_event_shape)


class TestMultivariateNormal(TestFactorised):
    total_ndims = 5
    num_samples = 3

    def setUp(self):
        self.dist = MultivariateNormal(torch.zeros(self.total_ndims), torch.eye(self.total_ndims))
        self.assertEqual(self.dist.event_shape, (self.total_ndims,))
        self.values = self.dist.sample((self.num_samples,))
        self.assertEqual(self.values.shape, (self.num_samples, self.total_ndims))


class TestMultivariateMixture(TestFactorised):
    total_ndims = 5
    num_samples = 3
    num_components = 7

    def setUp(self):
        mixing = torch.ones(self.num_components) / self.num_components
        components = MultivariateNormal(torch.zeros(self.num_components, self.total_ndims),
                                        torch.eye(self.total_ndims))
        self.dist = MultivariateMixture(mixing, components)
        self.assertEqual(self.dist.event_shape, (self.total_ndims,))
        self.values = self.dist.sample((self.num_samples,))
        self.assertEqual(self.values.shape, (self.num_samples, self.total_ndims))


if __name__ == '__main__':
    unittest.main()
