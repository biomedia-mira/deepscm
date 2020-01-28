from typing import Any, Generic, Mapping, Sequence, TypeVar, Union

import torch
import torch.distributions as td
from torch import nn

from distributions.factorised import Factorised
from distributions.mixture import Mixture
from distributions.mvn import MultivariateNormal
from distributions.multivariate import MultivariateDistribution

T = TypeVar('T', bound=td.Distribution)
Size = Union[torch.Size, Sequence[int]]


def _broadcastable(shape1, shape2):  # https://stackoverflow.com/a/47244284
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(shape1[::-1], shape2[::-1]))


def _all_broadcastable(*shapes):
    for i, shape1 in enumerate(shapes[:-1]):
        for shape2 in shapes[i + 1:]:
            if not _broadcastable(shape1, shape2):
                return False
    return True


def _broadcast_shapes(*shapes):
    if not _all_broadcastable(*shapes):
        raise ValueError(f"Shapes are not broadcastable: {shapes}")
    return torch.broadcast_tensors(*[torch.empty(shape) for shape in shapes])[0].shape


class DistributionParams(Generic[T], nn.Module):
    def __init__(self, batch_shape: Size = torch.Size()):
        super().__init__()
        self.batch_shape = torch.Size(batch_shape)

    def get_distribution(self) -> T:
        raise NotImplementedError

    @staticmethod
    def from_distribution(dist: T) -> 'DistributionParams[T]':
        raise NotImplementedError

    # Hack to remove linting warnings and allow code completion
    def __call__(self, *input, **kwargs) -> Any:
        return super().__call__(*input, **kwargs)


def _assemble_tril(log_diag: torch.Tensor, tril_vec: torch.Tensor):
    tril = torch.diag_embed(log_diag.exp())
    i, j = torch.tril_indices(*tril.shape[-2:], offset=-1)
    tril[..., i, j] = tril_vec
    return tril


def _disassemble_tril(tril: torch.Tensor):
    log_diag = tril.diagonal(dim1=-2, dim2=-1).log()
    i, j = torch.tril_indices(*tril.shape[-2:], offset=-1)
    tril_vec = tril[..., i, j]
    return log_diag, tril_vec


class MultivariateNormalParams(DistributionParams[MultivariateNormal]):
    def __init__(self, n_dimensions: int, batch_shape: Size = torch.Size(), var_names=None):
        super().__init__(batch_shape=batch_shape)
        cov_low_tri_dim = int((n_dimensions * (n_dimensions - 1)) / 2)

        self.loc = nn.Parameter(torch.randn(*batch_shape, n_dimensions))
        self.log_diag = nn.Parameter(torch.randn(*batch_shape, n_dimensions))
        self.tril_vec = nn.Parameter(torch.randn(*batch_shape, cov_low_tri_dim))
        self.var_names = var_names

    @property
    def scale_tril(self):
        return _assemble_tril(self.log_diag, self.tril_vec)

    def get_distribution(self) -> MultivariateNormal:
        return MultivariateNormal(self.loc, scale_tril=self.scale_tril, var_names=self.var_names)

    @staticmethod
    def from_distribution(dist: MultivariateNormal) -> 'MultivariateNormalParams':
        new = MultivariateNormalParams.__new__(MultivariateNormalParams)
        super(MultivariateNormalParams, new).__init__(batch_shape=dist.batch_shape)
        log_diag, tril_vec = _disassemble_tril(dist.scale_tril)
        new.loc = nn.Parameter(dist.loc)
        new.log_diag = nn.Parameter(log_diag)
        new.tril_vec = nn.Parameter(tril_vec)
        new.var_names = dist.variable_names
        return new

    def extra_repr(self):
        s = f"{self.loc.shape[-1]}"
        if self.batch_shape:
            s += f", batch_shape={self.batch_shape}"
        if self.var_names is not None:
            s += f", var_names={self.var_names}"
        return s


class CategoricalParams(DistributionParams[td.Categorical]):
    def __init__(self, n_categories, batch_shape: Size = torch.Size()):
        super().__init__(batch_shape=torch.Size(batch_shape))
        self.logits = nn.Parameter(torch.randn(*batch_shape, n_categories))

    def get_distribution(self) -> td.Categorical:
        return td.Categorical(logits=self.logits)

    @staticmethod
    def from_distribution(dist: td.Categorical):
        new = CategoricalParams.__new__(CategoricalParams)
        super(CategoricalParams, new).__init__(batch_shape=dist.batch_shape)
        new.logits = nn.Parameter(dist.logits)
        return new

    def extra_repr(self):
        s = f"{self.logits.shape[-1]}"
        batch_shape = self.logits.shape[:-1]
        if batch_shape:
            s += f", batch_shape={batch_shape}"
        return s


class FactorisedParams(DistributionParams[Factorised]):
    def __init__(self, factors: Union[Sequence[DistributionParams],
                                      Mapping[str, DistributionParams]],
                 var_names=None):
        if isinstance(factors, Mapping):
            if var_names is not None:
                raise ValueError("var_names should not be given alongside a factor dictionary")
            var_names = list(factors.keys())
            factors = list(factors.values())

        batch_shapes = [factor.batch_shape for factor in factors]
        batch_shape = _broadcast_shapes(*batch_shapes)
        super().__init__(batch_shape=batch_shape)

        if var_names is not None:
            if len(factors) != len(var_names):
                raise ValueError(f"Number of names ({len(var_names)}) must match "
                                 f"number of factors ({len(factors)})")
            self.factors = nn.ModuleDict(dict(zip(var_names, factors)))
        else:
            self.factors = nn.ModuleList(factors)
        self.var_names = var_names

    def get_distribution(self) -> Factorised:
        factors = self.factors.values() if isinstance(self.factors, nn.ModuleDict) else self.factors
        dist_factors = [factor.get_distribution() for factor in factors]
        return Factorised(dist_factors, var_names=self.var_names)


class MixtureParams(DistributionParams[Mixture]):
    def __init__(self, mixing: CategoricalParams, components: DistributionParams, var_names=None):
        batch_shape = _broadcast_shapes(mixing.logits.shape, components.batch_shape)
        batch_shape = batch_shape[:-1]  # Discard component dimension
        super().__init__(batch_shape=batch_shape)
        self.mixing = mixing
        self.components = components
        self.var_names = var_names

    def get_distribution(self) -> Mixture:
        mixing = self.mixing.get_distribution()
        components = self.components.get_distribution()
        if isinstance(components, MultivariateDistribution) and self.var_names is not None:
            components.rename(self.var_names)
        return Mixture(mixing, components)


if __name__ == '__main__':
    prior = FactorisedParams({
        'factorised': FactorisedParams([
            MultivariateNormalParams(2),
            MultivariateNormalParams(3)
        ]),
        'mixture': MixtureParams(
            CategoricalParams(5),
            MultivariateNormalParams(3, (5,))
        ),
        'mixture_of_factorised': MixtureParams(
            CategoricalParams(7),
            FactorisedParams([
                MultivariateNormalParams(2, (7,)),
                MultivariateNormalParams(8, (7,))
            ])
        ),
        'mixture_of_mixtures': MixtureParams(
            CategoricalParams(4),
            MixtureParams(
                CategoricalParams(5, (4,)),
                MultivariateNormalParams(2, (4, 5), var_names=['x', 'y']),
            )
        )
    })
    print(prior)

    def print_state_dict(module: nn.Module):
        print('{ ' + ',\n  '.join(f"{key}: {tuple(val.shape)}, {val.dtype}, {val.device}"
              for key, val in module.state_dict().items()) + ' }')

    print_state_dict(prior)
    prior.double()
    print_state_dict(prior)

    prior_dist = prior.get_distribution()
    print(prior_dist)
    print(prior_dist.variable_names)
    print(prior_dist.log_prob(prior_dist.sample((2,))))

    mvn = prior.factors['mixture'].components.get_distribution()
    print(mvn)
    mvn.rename([chr(ord('a') + i) for i in range(mvn.num_variables)])
    print(mvn)
