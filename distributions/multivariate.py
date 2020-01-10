from typing import Sequence, Union

import numpy as np
from torch.distributions import Distribution


def _is_single(idx):
    return np.ndim(idx) == 0


class MultivariateDistribution(Distribution):
    @property
    def num_variables(self) -> int:
        raise NotImplementedError

    @property
    def variable_shapes(self) -> Sequence[int]:
        raise NotImplementedError

    def _check_index(self, index):
        if index < 0 or index >= self.num_variables:
            raise ValueError(f"Variable index ({index}) must be between 0 and "
                             f"number of variables ({self.num_variables})")

    def _marginalise_single(self, marg_index) -> Distribution:
        raise NotImplementedError

    def _marginalise_multi(self, marg_indices) -> 'MultivariateDistribution':
        raise NotImplementedError

    def marginalise(self, which) -> Union[Distribution, 'MultivariateDistribution']:
        if _is_single(which):
            self._check_index(which)
            return self._marginalise_single(which)
        else:
            for index in which:
                self._check_index(index)
            return self._marginalise_multi(which)

    def _condition(self, marg_indices, cond_indices, cond_values, squeeze) \
            -> Union[Distribution, 'MultivariateDistribution']:
        raise NotImplementedError

    def condition(self, cond_index_value_dict, squeeze=False) -> 'MultivariateDistribution':
        for index in cond_index_value_dict:
            self._check_index(index)
        marg_indices = [i for i in range(self.num_variables) if i not in cond_index_value_dict]
        if squeeze and len(marg_indices) > 1:
            raise RuntimeError(f"Only univariate distributions can be squeezed "
                               f"(num_variables={len(marg_indices)})")
        cond_indices = list(cond_index_value_dict.keys())
        cond_values = list(cond_index_value_dict.values())
        return self._condition(marg_indices, cond_indices, cond_values, squeeze)

    def squeeze(self) -> Distribution:
        if self.num_variables != 1:
            raise RuntimeError(f"Only univariate distributions can be squeezed "
                               f"(num_variables={self.num_variables})")
        return self.marginalise(0)


class NamedMultivariateDistribution(Distribution):
    def __init__(self, base_dist: MultivariateDistribution, var_names: Sequence[str]):
        if len(var_names) != base_dist.num_variables:
            raise ValueError(f"Number of names ({len(var_names)}) must match "
                             f"number of variables ({base_dist.num_variables})")
        self.base_dist = base_dist
        self.var_names = var_names
        self.var_indices = {name: i for i, name in enumerate(var_names)}
    
    def __getattr__(self, name): 
        # Delegate attributes and methods to base_dist
        return getattr(self.base_dist, name)
    
    def _map_indices(self, names):
        return [self.var_indices[name] for name in names]
    
    def marginalise(self, which):
        which_indices = self._map_indices(which)
        marg_dist = self.base_dist.marginalise(which_indices)
        marg_names = which
        return NamedMultiDistribution(marg_dist, marg_names)
    
    def condition(self, **given):
        given_indices = {self.var_indices[name]: value for name, value in given.items()}
        cond_dist = self.base_dist.condition(given_indices)
        cond_names = [name for name in var_names if name not in given]
        return NamedMultiDistribution(cond_dist, cond_names)


if __name__ == '__main__':
    from torch.distributions import Foo

    factor_dist = Factorised([dist1, dist2])
    named = NamedMultivariateDistribution(factor_dist, ['x', 'y'])
    named.condition(z=values1)
    
