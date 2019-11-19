"""Adapted from: https://github.com/emtiyaz/vmp-for-svae/blob/master/distributions/gaussian.py"""
import numpy as np
import torch


def standard_to_natural(mu, sigma):
    eta2 = -0.5 * torch.inverse(sigma)  # shape = (nb_components, latent_dim, latent_dim)
    eta1 = -2 * eta2 @ mu  # shape = (nb_components, latent_dim)
    return eta1, eta2


def natural_to_standard(eta1, eta2):
    sigma = torch.inverse(-2 * eta2)
    mu = sigma @ eta1
    return mu, sigma


def log_probability_nat(x: torch.Tensor, eta1: torch.Tensor, eta2: torch.Tensor, weights=None):
    N, D = x.shape

    logprob = eta1 @ x
    logprob += x @ eta2 @ x
    logprob -= D/2. * torch.log(2. * np.pi)

    # eta1: (K, D), eta2: (K, D, D)
    # add dimension for further computations
    eta1 = eta1.unsqueeze(3)  # (K, D, 1, 1)
    logprob += .25 * torch.einsum('nkdi,nkdi->nk', torch.solve(eta1, eta2)[0], eta1)

    logprob += 0.5 * torch.logdet(-2. * eta2 + 1e-20 * torch.eye(D))

    if weights is not None:
        logprob += torch.log(weights).unsqueeze(0)

    normalizer = torch.logsumexp(logprob, dim=1, keepdim=True)

    return logprob - normalizer


def log_probability_nat_per_samp(x_samps, eta1, eta2):
    """
    Args:
        x_samps: matrix of shape (N, K, S, D)
        eta1: 1st natural parameter for Gaussian distr; shape: (N, K, D)
        eta2: 2nd natural parameter for Gaussian distr; shape: (N, K, D, D)
    Returns:
        1/S sum^S_{s=1} log N(x^(s)|eta1, eta2) of shape (N, K, S)
    """
    # same as above, but x consists of S samples for K components: x.shape = (N, K, S, D)
    # todo: merge with above function (above is the same but normalised)
    N, K, S, D = x_samps.shape
    assert eta1.shape == (N, K, D)
    assert eta2.shape == (N, K, D, D)

    # -1\2 (sigma^(-1) * x) * x + sigma^(-1)*mu*x
    log_normal = torch.einsum('nksi,nkij,nksj->nks', x_samps, eta2, x_samps),
    log_normal += torch.einsum('nki,nksi->nks', eta1, x_samps)

    # 1/4 (-2 * sigma * (sigma^(-1) * mu)) sigma^(-1) * mu = -1/2 mu sigma^(-1) mu; shape = N, K, 1
    log_normal += .25 * torch.einsum('nkdi,nkd->nki', torch.solve(eta1, eta2)[0], eta1)
    log_normal -= D/2. * np.log(2. * np.pi)

    # + 1/2 log |sigma^(-1)|
    log_normal += .5 * torch.logdet(-2.0 * eta2 + 1e-20 * torch.eye(D)).unsqueeze(2)

    return log_normal
