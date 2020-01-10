import unittest

import torch
import torch.distributions as td

from distributions.factorised import Factorised
from distributions.multivariate import NamedMultivariateDistribution


class TestNamedMultivariate(unittest.TestCase):
    num_samples = 3
    num_factors = 5
    var_shapes = [i + 10 for i in range(num_factors)]
    total_ndims = sum(var_shapes)
    var_names = [chr(i + 97) for i in range(num_factors)]  # ['a', 'b', 'c', ...]
    named_shapes = dict(zip(var_names, var_shapes))

    def setUp(self):
        self.dist = Factorised([td.MultivariateNormal(torch.zeros(d), torch.eye(d))
                                for d in self.var_shapes])
        self.assertEqual(self.dist.event_shape, (self.total_ndims,))
        self.named = NamedMultivariateDistribution(self.dist, self.var_names)
        self.values = self.named.sample((self.num_samples,))
        self.assertEqual(self.values.shape, (self.num_samples, self.total_ndims))
        self.named_values = {name: self.named.marginalise(name).sample((self.num_samples,))
                             for name in self.var_names}

    def test_constructor(self):
        with self.assertRaises(ValueError):
            NamedMultivariateDistribution(self.dist, self.var_names[:-1])

    def test_invalid_names(self):
        invalid_name = 'invalid_name'
        dummy_values = self.values[:, 0]
        valid_name = self.var_names[0]

        with self.assertRaises(KeyError):
            self.named.marginalise(invalid_name)
        with self.assertRaises(KeyError):
            self.named.marginalise([invalid_name, valid_name])

        with self.assertRaises(KeyError):
            self.named.condition({invalid_name: dummy_values})
        with self.assertRaises(KeyError):
            self.named.condition({invalid_name: dummy_values,
                                  valid_name: dummy_values})

    def test_marginalise_single(self):
        name = 'a'
        marg = self.named.marginalise(name)
        expected_var_shape = self.named_shapes[name]
        self.assertEqual(marg.event_shape, (expected_var_shape,))

    def test_marginalise_multi(self):
        for names in [['a'], ['a', 'b'], ['b', 'a']]:
            marg = self.named.marginalise(names)
            self.assertSequenceEqual(marg.var_names, names)
            expected_var_shapes = [self.named_shapes[name] for name in names]
            self.assertSequenceEqual(marg.variable_shapes, expected_var_shapes)

    def test_condition(self):
        for names in [['a'], ['a', 'b'], ['b', 'a']]:
            marg_names = [name for name in self.var_names if name not in names]
            cond_dict = {name: self.named_values[name] for name in names}
            cond = self.named.condition(cond_dict, squeeze=False)
            self.assertSequenceEqual(cond.var_names, marg_names)
            expected_var_shapes = [self.named_shapes[name] for name in marg_names]
            self.assertSequenceEqual(cond.variable_shapes, expected_var_shapes)

    def test_simultaneously_marginalise_and_condition(self):
        for marg_names, cond_names in [(['a'], ['b']),
                                       (['a', 'b'], ['c']),
                                       (['a'], ['b', 'c']),
                                       (['a', 'b'], ['c', 'd'])]:
            cond_dict = {name: self.named_values[name] for name in cond_names}
            cond = self.named(*marg_names, **cond_dict, squeeze=False)
            self.assertSequenceEqual(cond.var_names, marg_names)
            expected_var_shapes = [self.named_shapes[name] for name in marg_names]
            self.assertSequenceEqual(cond.variable_shapes, expected_var_shapes)


if __name__ == '__main__':
    unittest.main()
