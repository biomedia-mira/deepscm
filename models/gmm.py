"""Adapted from: https://github.com/emtiyaz/vmp-for-svae/blob/master/models/gmm.py"""

import torch
import torch.distributions as td
from torch import nn
<<<<<<< HEAD
from torch.nn import functional as F
=======
>>>>>>> a9f508a7fc1299dae410157e1a6b8060e53b5957

from distributions.natural_nw import NaturalNormalWishart
from models import natural_gmm, mixture
from util import triangular_logdet, mahalanobis, outer

"""
Variational Mixture of Gaussians, according to:
  Pattern Matching and Machine Learning (Chapter 10.2)
  Christopher M. Bishop.
  Springer, 2006.
"""


def _compute_expec_log_lik(x, component_posteriors: NaturalNormalWishart):
    # Bishop eq 10.64
    # output shape: (N, K)
    expec_eta1, expec_eta2, expec_log_norm = component_posteriors.expected_stats()

    prod1 = torch.einsum('nd,kd->nk', x, expec_eta1)
    prod2 = torch.einsum('nd,kde,ne->nk', x, expec_eta2, x)

    return prod1 + prod2 - expec_log_norm


def _compute_expct_log_pi(mixing_posterior: td.Dirichlet):
    # Bishop eq 10.66
    alpha_k = mixing_posterior.concentration
    return torch.digamma(alpha_k) - torch.digamma(alpha_k.sum())


def e_step(x, mixing_posterior, component_posteriors):
    """
    Variational E-update: update local parameters
    Args:
        x: data
        mixing_posterior
        component_posteriors

    Returns:
        responsibilities
    """
    expec_log_lik = _compute_expec_log_lik(x, component_posteriors)
    expct_log_pi = _compute_expct_log_pi(mixing_posterior)  # Bishop eq 10.66
    r_nk = (expct_log_pi + expec_log_lik).softmax(-1)
    return r_nk


def m_step(x, r_nk, mixing_prior: td.Dirichlet, component_prior: NaturalNormalWishart):
    """
    Variational M-update: Update global parameters
    Args:
        x: data
        r_nk: responsibilities
        mixing_prior
        component_prior

    Returns:
        posterior parameters
    """
    N_k = r_nk.sum(0)
    mixing_posterior = td.Dirichlet(mixing_prior.concentration + N_k)  # Bishop eq 10.58
    component_posteriors = component_prior.posterior(x, r_nk)

    return mixing_posterior, component_posteriors


def run_vem(x, K, n_iter: int):
    N, D = x.shape

    r_nk = td.Dirichlet(torch.ones(K)).sample((N,))
    gmm = natural_gmm.init(K, D, alpha_scale=0.05 / K, nu_scale=0.5, mean_scale=0,
                           cov_scale=D + 0.5, dof_init=D + 0.5)  # type: natural_gmm.NaturalGMMPrior

    for i in range(n_iter):
        mixing_posterior, component_posteriors = m_step(x, r_nk, gmm.mixing_prior, gmm.component_prior)
        r_nk = e_step(x, mixing_posterior, component_posteriors)

    return natural_gmm.NaturalGMMPrior.from_priors(mixing_posterior, component_posteriors)


def inference(x, K):
    """
    Args:
        x: data; shape = N, D
        K: number of components
    """
    N, D = x.shape

    r_nk = td.Dirichlet(torch.ones(K)).sample((N,))

    gmm = natural_gmm.init(K, D, alpha_scale=0.05 / K, nu_scale=0.5, mean_scale=0,
                           cov_scale=D + 0.5, dof_init=D + 0.5)  # type: natural_gmm.NaturalGMMPrior

    mixing_posterior, component_posteriors = m_step(x, r_nk, gmm.mixing_prior, gmm.component_prior)

    r_nk_new, pi = e_step(x, mixing_posterior, component_posteriors)


class MultivariateGMM(nn.Module):
    def __init__(self, n_components, n_dimensions):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(n_components))
        self.means = nn.Parameter(torch.randn(n_components, n_dimensions))

        cov_low_tri_dim = int((n_dimensions * (n_dimensions - 1)) / 2)

        self.diag = nn.Parameter(torch.rand(n_components, n_dimensions))
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
        components = td.MultivariateNormal(self.means, scale_tril=tril)
        return mixture.MultivariateNormalMixture(mixing, components)

    def forward(self, data):
        return self._get_distribution().log_prob(data)
