from typing import Sequence

import torch
from torch.distributions import Distribution


class MultivariateDistribution(Distribution):
    @abstractproperty
    def num_variables(self):
        raise NotImplementedError

    def marginalise(self, which_indices):
        raise NotImplementedError
    
    def condition(self, cond_index_value_dict):
        raise NotImplementedError


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
    
