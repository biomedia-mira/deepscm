import torch
import torch.distributions as td
from torch import nn

from distributions.mixture import NaturalMultivariateNormalMixture
from distributions.natural_nw import NaturalNormalWishart
from distributions.natural_mvn import NaturalMultivariateNormal


class MultivariateNGMM(nn.Module):
    def __init__(self, n_components, n_dimensions):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(n_components))
        self.locs = nn.Parameter(torch.randn(n_components, n_dimensions))

        cov_low_tri_dim = int((n_dimensions * (n_dimensions - 1)) / 2)
        self.diag = nn.Parameter(torch.randn(n_components, n_dimensions))
        self.tril_vec = nn.Parameter(torch.randn(n_components, cov_low_tri_dim))

        self.distribution = self._get_distribution()

    def tril(self, diag: torch.Tensor, tril_vec: torch.Tensor):
        dim = diag.shape[-1]
        L = torch.diag_embed(torch.exp(diag))  # L is lower-triangular
        L = L.to(diag.device)
        i, j = torch.tril_indices(dim, dim, offset=-1)
        L[..., i, j] = tril_vec

        return L

    def _get_distribution(self):
        mixing = td.Categorical(logits=self.logits)
        tril = self.tril(self.diag, self.tril_vec)
        precisions = tril @ tril.transpose(-1, -2)
        components = NaturalMultivariateNormal(self.locs, -precisions)
        return NaturalMultivariateNormalMixture(mixing, components)

    def forward(self, data):
        return self._get_distribution().log_prob(data)


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
