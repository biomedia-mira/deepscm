from typing import Generic, TypeVar, Union

import torch
import torch.distributions as td

from distributions.natural_mvn import NaturalMultivariateNormal
from distributions.products import product
from util import posdef_solve

T = TypeVar('T', bound=td.Distribution)


class Mixture(td.Distribution, Generic[T]):
    def __init__(self, proportions: Union[td.Categorical, torch.Tensor], components: T):
        if isinstance(proportions, torch.Tensor):
            proportions = td.Categorical(proportions)
        if proportions._num_events != components.batch_shape[-1]:
            raise ValueError(f"Length of proportions vector ({proportions._num_events}) "
                             f"must match number of components ({components.batch_shape[-1]}).")
        self.mixing = proportions
        self.components = components
        super().__init__(components.batch_shape[:-1], components.event_shape)

    @property
    def num_components(self) -> int:
        return self.components.batch_shape[-1]

    def rsample(self, sample_shape=torch.Size()):
        assignments = self.mixing.sample(sample_shape)
        samples = self.components.rsample(sample_shape)
        batch_shape = self.batch_shape
        full_shape = [*sample_shape, *batch_shape, 1, *self.event_shape]
        thin_shape = [*sample_shape, *batch_shape, 1] + [1] * len(self.event_shape)
        sdim = len(sample_shape) + len(batch_shape)
        assignments = assignments.view(thin_shape).expand(full_shape)
        return samples.gather(sdim, assignments, sparse_grad=True).squeeze(sdim)

    def _broadcast(self, sample: torch.Tensor) -> torch.Tensor:
        sample_shape = torch.Size(sample.shape[:-1])
        batch_shape = self.components.batch_shape
        empty_batch_shape = torch.Size([1] * len(batch_shape))
        thin_shape = sample_shape + empty_batch_shape + self.event_shape
        full_shape = sample_shape + batch_shape + self.event_shape
        return sample.view(thin_shape).expand(full_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_liks = self.components.log_prob(self._broadcast(value))
        return torch.logsumexp(self.mixing.logits + log_liks, dim=-1)

    def posterior(self, potentials: T) -> 'Mixture':
        post_components, post_lognorm = product(potentials, self.components, expand=True)
        post_logits = self.mixing.logits.unsqueeze(-1) + post_lognorm
        post_mixing = td.Categorical(logits=post_logits)
        return Mixture(post_mixing, post_components)


class MultivariateNormalMixture(Mixture[td.MultivariateNormal]):
    def posterior(self, potentials: td.MultivariateNormal) -> 'MultivariateNormalMixture':
        means = potentials.mean.unsqueeze(1)  # (N, 1, D)
        precs = potentials.precision_matrix.unsqueeze(1)  # (N, 1, D, D)
        covs = potentials.covariance_matrix.unsqueeze(1)  # (N, 1, D, D)

        prior_means = self.components.mean.unsqueeze(0)  # (1, K, D)
        prior_precs = self.components.precision_matrix.unsqueeze(0)  # (1, K, D, D)
        prior_covs = self.components.covariance_matrix.unsqueeze(0)  # (1, K, D, D)

        post_precs = precs + prior_precs
        post_means = posdef_solve(precs @ means[..., None] + prior_precs @ prior_means[..., None],
                                  post_precs)[0].squeeze(-1)
        post_components = td.MultivariateNormal(post_means, precision_matrix=post_precs)

        post_lognorm = td.MultivariateNormal(prior_means, covs + prior_covs).log_prob(means)
        post_logits = self.mixing.logits + post_lognorm

        return MultivariateNormalMixture(td.Categorical(logits=post_logits), post_components)


class NaturalMultivariateNormalMixture(Mixture[NaturalMultivariateNormal]):
    def posterior(self, potentials: Union[NaturalMultivariateNormal, td.MultivariateNormal]) \
            -> 'NaturalMultivariateNormalMixture':
        if isinstance(potentials, td.MultivariateNormal):
            potentials = NaturalMultivariateNormal.from_standard(potentials)

        eta1 = potentials.nat_param1.unsqueeze(1)  # (N, 1, D)
        eta2 = potentials.nat_param2.unsqueeze(1)  # (N, 1, D, D)

        prior_eta1 = self.components.nat_param1.unsqueeze(0)  # (1, K, D)
        prior_eta2 = self.components.nat_param2.unsqueeze(0)  # (1, K, D, D)

        post_eta1 = eta1 + prior_eta1
        post_eta2 = eta2 + prior_eta2
        post_components = NaturalMultivariateNormal(post_eta1, post_eta2)

        post_lognorm = post_components.log_normalizer - self.components.log_normalizer
        post_logits = self.mixing.logits + post_lognorm

        return NaturalMultivariateNormalMixture(td.Categorical(logits=post_logits), post_components)


def eval_grid(xx, yy, fcn):
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return fcn(xy).reshape_as(xx)


if __name__ == '__main__':
    N, K, D = 200, 4, 2
    props = td.Dirichlet(5*torch.ones(K)).sample()
    mean = torch.arange(K).float().view(K, 1).expand(K, D)
    var = .1 * torch.eye(D).expand(K, -1, -1)
    mixing = td.Categorical(props)
    components = td.MultivariateNormal(mean, var)
    print("mixing", mixing.batch_shape, mixing.event_shape)
    print("components", components.batch_shape, components.event_shape)
    # mixture = MultivariateNormalMixture(mixing, components)
    # mixture = NaturalMultivariateNormalMixture(mixing, NaturalMultivariateNormal.from_standard(components))
    mixture = Mixture(mixing, NaturalMultivariateNormal.from_standard(components))
    print("mixture", mixture.batch_shape, mixture.event_shape)
    probe = td.MultivariateNormal(mean[:3]+1*torch.tensor([1., -1.]), .2 * var[:3])
    post_mixture = mixture.posterior(probe)
    samples = mixture.sample([N])
    n = 1
    post_samples = post_mixture.sample([N])[:, n]
    print("sample", samples.shape)

    x = torch.linspace(-2, K - 1 + 2, 200)
    y = torch.linspace(-2, K - 1 + 2, 200)
    xx, yy = torch.meshgrid(x, y)
    xy = torch.stack([xx, yy], -1)
    zz = mixture.log_prob(xy)
    post_zz = post_mixture.log_prob(xy)[:, :, 1]
    probe_zz = probe.log_prob(xy.unsqueeze(-2))[:, :, 1]

    import matplotlib.pyplot as plt
    plt.imshow(zz.exp().T, interpolation='bilinear', origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]])
    plt.scatter(*samples.T, c='k', s=4)
    plt.contour(xx, yy, probe_zz.exp(), cmap='inferno')
    plt.show()

    plt.imshow(post_zz.exp().T, interpolation='bilinear', origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]])
    plt.scatter(*post_samples.T, c='k', s=2)
    plt.contour(xx, yy, zz.exp(), cmap='inferno')
    plt.contour(xx, yy, probe_zz.exp(), cmap='inferno')
    plt.show()
    # mixture.posterior(td.Normal(mean, std))
