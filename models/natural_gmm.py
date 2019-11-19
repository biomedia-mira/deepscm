import torch
import torch.distributions as td
from torch import nn

from distributions.natural_nw import NaturalNormalWishart
from models.mixture import NaturalMultivariateNormalMixture


class NaturalGMMPrior(nn.Module):
    def __init__(self, alpham1, dof, lambda1, lambda2, nu, requires_grad=False):
        self.alpham1 = nn.Parameter(alpham1, requires_grad=requires_grad)
        self.dof = nn.Parameter(dof, requires_grad=requires_grad)
        self.lambda1 = nn.Parameter(lambda1, requires_grad=requires_grad)
        self.lambda2 = nn.Parameter(lambda2, requires_grad=requires_grad)
        self.nu = nn.Parameter(nu, requires_grad=requires_grad)
        super().__init__()

    @property
    def mixing_prior(self) -> td.Dirichlet:
        return td.Dirichlet(self.alpham1 + 1.)

    @property
    def component_prior(self) -> NaturalNormalWishart:
        return NaturalNormalWishart(self.dof, self.lambda1, self.lambda2, self.nu)

    def rsample(self, sample_shape=torch.Size()):
        mixing = self.mixing_prior.rsample(sample_shape)
        components = self.component_prior.rsample(sample_shape)
        return NaturalMultivariateNormalMixture(mixing, components)

    @staticmethod
    def from_priors(mixing_prior: td.Dirichlet, component_prior: NaturalNormalWishart) \
            -> 'NaturalGMMPrior':
        return NaturalGMMPrior(mixing_prior.concentration - 1., component_prior.dof,
                               component_prior.lambda1, component_prior.lambda2, component_prior.nu)


def init(n_components, latent_dims, alpha_scale=.1, nu_scale=1e-5, dof_init=10.,
         mean_scale=1., cov_scale=10., requires_grad=False, device='cuda:0') -> NaturalGMMPrior:
    alpha = torch.full((n_components,), alpha_scale, device=device)
    means = mean_scale * torch.randn(n_components, latent_dims, device=device)
    nu = torch.full((n_components,), nu_scale, device=device)
    a = .5 * torch.full((n_components,), float(latent_dims + dof_init), device=device)
    B = .5 * cov_scale * torch.eye(latent_dims, device=device).expand(n_components, -1, -1)

    # transform to natural parameters
    prior = NaturalNormalWishart.from_standard(means, nu, a, B)
    alpham1 = alpha - 1.

    return NaturalGMMPrior(alpham1, prior.dof, prior.lambda1, prior.lambda2, prior.nu,
                           requires_grad=requires_grad).to(device)
