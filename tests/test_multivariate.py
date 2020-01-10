import unittest

import torch
import torch.distributions as td

from distributions.factorised import Factorised
from distributions.multivariate import MultivariateDistribution
from distributions.mvn import MultivariateNormal
from distributions.natural_mvn import NaturalMultivariateNormal
from models.mixture import MultivariateMixture


class BaseTests:
    class TestMultivariate(unittest.TestCase):
        total_ndims = -1
        num_samples = 3

        def _get_distribution(self) -> MultivariateDistribution:
            raise NotImplementedError

        def setUp(self):
            self.dist = self._get_distribution()
            self.assertEqual(self.dist.event_shape, (self.total_ndims,))
            self.values = self.dist.sample((self.num_samples,))
            self.assertEqual(self.values.shape, (self.num_samples, self.total_ndims))

        def test_num_variables(self):
            self.assertEqual(self.dist.num_variables, self.total_ndims)

        def test_variable_shapes(self):
            var_shapes = self.dist.variable_shapes
            self.assertEqual(len(var_shapes), self.dist.num_variables)
            self.assertEqual(sum(var_shapes), self.dist.event_shape[0])

        def test_invalid_indices(self):
            dummy_values = self.values[:, 0]
            valid_index = 0
            valid_values = self.values[:, valid_index]
            for invalid_index in [-self.dist.num_variables - 1,
                                  -1,
                                  self.dist.num_variables,
                                  self.dist.num_variables + 1]:
                with self.assertRaises(ValueError):
                    self.dist.marginalise(invalid_index)
                with self.assertRaises(ValueError):
                    self.dist.marginalise([invalid_index, valid_index])

                with self.assertRaises(ValueError):
                    self.dist.condition({invalid_index: dummy_values})
                with self.assertRaises(ValueError):
                    self.dist.condition({invalid_index: dummy_values,
                                         valid_index: valid_values})

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

        def test_squeeze(self):
            with self.assertRaises(RuntimeError):
                self.dist.squeeze()
            var_index = 0
            marg = self.dist.marginalise([var_index])
            squeezed = marg.squeeze()
            self.assertEqual(squeezed.batch_shape, marg.batch_shape)
            # self.assertEqual(squeezed.event_shape, marg.event_shape)

        def test_condition_shape(self):
            var_shapes = self.dist.variable_shapes
            for dims in [(0,), [0], (0, 1), list(range(self.dist.num_variables - 1))]:
                cond_dict = {dim: self.values[:, dim].unsqueeze(-1) for dim in dims}
                cond = self.dist.condition(cond_dict)
                self.assertIsInstance(cond, MultivariateDistribution)
                expected_batch_shape = self.values.shape[:-1] + self.dist.batch_shape
                cond_dims = sum(var_shapes[d] for d in dims)
                expected_event_shape = (self.total_ndims - cond_dims,)
                self.assertEqual(cond.batch_shape, expected_batch_shape)
                self.assertEqual(cond.event_shape, expected_event_shape)

        def test_condition_squeeze(self):
            # Test typical squeeze case
            num_variables = self.dist.num_variables
            dims = list(range(num_variables - 1))
            cond_dict = {dim: self.values[:, dim].unsqueeze(-1) for dim in dims}
            cond = self.dist.condition(cond_dict, squeeze=False)
            self.assertEqual(cond.num_variables, 1)

            # Test consistency with conditional+squeeze and with single marginal
            marg_indices = [i for i in range(num_variables) if i not in cond_dict]
            assert len(marg_indices) == 1
            marg_index = marg_indices[0]
            cond_squeeze = self.dist.condition(cond_dict, squeeze=True)
            squeezed = cond.squeeze()
            marginal = self.dist.marginalise(marg_index)
            self.assertEqual(type(cond_squeeze), type(squeezed))
            self.assertEqual(type(cond_squeeze), type(marginal))
            self.assertEqual(cond_squeeze.batch_shape, squeezed.batch_shape)
            self.assertEqual(cond_squeeze.event_shape, squeezed.event_shape)
            self.assertEqual(cond_squeeze.event_shape, marginal.event_shape)

            # Test 'unsqueezable' conditioning
            dim = 0
            cond_dict = {dim: self.values[:, dim]}
            with self.assertRaises(RuntimeError):
                self.dist.condition(cond_dict, squeeze=True)


class TestFactorised(BaseTests.TestMultivariate):
    num_factors = 5
    total_ndims = num_factors * 2

    def _get_distribution(self):
        return Factorised([td.MultivariateNormal(torch.zeros(2), torch.eye(2))
                           for _ in range(self.num_factors)])

    def test_num_variables(self):
        self.assertEqual(self.dist.num_variables, self.num_factors)


class TestMultivariateNormal(BaseTests.TestMultivariate):
    total_ndims = 5

    def _get_distribution(self):
        return MultivariateNormal(torch.zeros(self.total_ndims), torch.eye(self.total_ndims))


class TestNaturalMultivariateNormal(BaseTests.TestMultivariate):
    total_ndims = 5

    def _get_distribution(self):
        mvn = MultivariateNormal(torch.zeros(self.total_ndims), torch.eye(self.total_ndims))
        return NaturalMultivariateNormal.from_standard(mvn)


class TestMultivariateMixture(BaseTests.TestMultivariate):
    total_ndims = 5
    num_components = 7

    def _get_distribution(self):
        mixing = torch.ones(self.num_components) / self.num_components
        components = MultivariateNormal(torch.zeros(self.num_components, self.total_ndims),
                                        torch.eye(self.total_ndims))
        return MultivariateMixture(mixing, components)

    def test_num_variables(self):
        self.assertEqual(self.dist.num_variables, self.total_ndims)


class TestCorrectness(unittest.TestCase):
    num_factors = 5
    var_shapes = [i + 10 for i in range(num_factors)]
    total_ndims = sum(var_shapes)

    def setUp(self):
        self.dist = Factorised([td.MultivariateNormal(torch.zeros(d), torch.eye(d))
                                for d in self.var_shapes])

    def test_marginalise_single(self):
        for index, dim in enumerate(self.var_shapes):
            marg = self.dist.marginalise(index)
            self.assertEqual(marg.event_shape, (dim,))

    def test_marginalise_multi(self):
        for index1, dim1 in enumerate(self.var_shapes):
            for index2, dim2 in enumerate(self.var_shapes):
                marg = self.dist.marginalise([index1, index2])
                self.assertEqual(tuple(marg.variable_shapes), (dim1, dim2))
                self.assertEqual(marg.event_shape, (dim1 + dim2,))


if __name__ == '__main__':
    unittest.main()
