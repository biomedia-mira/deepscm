"""Adapted from: https://github.com/emtiyaz/vmp-for-svae/blob/master/models/gmm.py"""

import torch
import torch.distributions as td

from . import niw_san, dirichlet_san

"""
Variational Mixture of Gaussians, according to:
  Pattern Matching and Machine Learning (Chapter 10.2)
  Christopher M. Bishop.
  Springer, 2006.
"""


def update_Nk(r_nk):
    # Bishop eq 10.51
    return r_nk.sum(0)


def update_xk(x, r_nk, N_k):
    # Bishop eq 10.52; output shape = (K, D)
    x_k = torch.einsum('nk,nd->kd', r_nk, x)
    x_k_normed = x_k / N_k[:, None]
    # remove nan values (if N_k == 0)
    return torch.where(N_k > 0, x_k_normed, x_k)


def update_Sk(x, r_nk, N_k, x_k):
    # Bishop eq 10.53
    x_xk = x[:, None] - x_k[None, :]
    S = torch.einsum('nk,nkd,nke->kde', r_nk, x_xk, x_xk)
    S_normed = S / N_k[..., None, None]
    # remove nan values (if N_k == 0)
    return torch.where(N_k > 0, S_normed, S)


def update_alphak(alpha_0, N_k):
    # Bishop eq 10.58
    return alpha_0 + N_k


def update_betak(beta_0, N_k):
    # Bishop eq 10.60
    return beta_0 + N_k


def update_mk(beta_0, m_0, N_k, x_k, beta_k):
    # Bishop eq 10.61
    if len(beta_0.shape) == 1:
        beta_0 = beta_0.unsqueeze(1)

    Nk_xk = N_k[:, None] * x_k
    beta0_m0 = beta_0 * m_0
    return (beta0_m0 + Nk_xk) / beta_k[:, None]


def update_Ck(C_0, x_k, N_k, m_0, beta_0, beta_k, S_k):
    # Bishop eq 10.62
    C = C_0 + N_k[:, None, None] * S_k
    Q0 = x_k - m_0
    q = Q0[:, :, None] * Q0[:, None, :]
    return C + (beta_0 * N_k / beta_k)[:, None, None] * q


def update_vk(v_0, N_k):
    # Bishop eq 10.63
    return v_0 + N_k + 1


def compute_expct_mahalanobis_dist(x, beta_k, m_k, P_k, v_k):
    # Bishop eq 10.64
    # output shape: (N, K)
    D = x.shape[1]

    dist = x[:, None] - m_k[None, :]  # shape=(N, K, D)
    m = torch.einsum('k,nkd,kde,nke->nk', v_k, dist, P_k, dist)
    return m + (D / beta_k)[None, :]  # shape=(N, K)


def compute_dev_missing_data(x, beta_k, m_k, P_k, v_k, missing_data_mask):
    # Bishop eq 10.64; ignoring missing data
    # output shape: (N, K)
    D = x.shape[1]

    dist = x[:, None] - m_k[None, :]  # shape=(N, K, D)

    # exclude missing data: set 'missing' values to zero
    av_data_mask = (~missing_data_mask).float()[:, None]
    dist = dist * av_data_mask

    m = torch.einsum('k,nkd,kde,nke->nk', v_k, dist, P_k, dist)
    return m + (D / beta_k)[None, :]  # shape=(N, K)


def compute_expct_log_det_prec(v_k, P_k):
    # Bishop eq 10.65
    log_det_P = torch.logdet(P_k)
    # log_det_W = tf.log()  # shape=(5,)

    K, D, _ = P_k.shape
    D_log_2 = D * torch.log(2.)

    i = torch.arange(D, dtype=torch.float32)[None, :]
    sum_digamma = torch.digamma(0.5 * (v_k[:, None] + 1. + i)).sum(1)

    return sum_digamma + D_log_2 + log_det_P


def compute_log_pi(alpha_k):
    # Bishop eq 10.66
    return torch.digamma(alpha_k) - torch.digamma(alpha_k.sum())


def compute_rnk(expct_log_pi, expct_log_det_cov, expct_dev):
    # Bishop eq 10.49
    log_rho_nk = expct_log_pi + .5 * (expct_log_det_cov - expct_dev)
    return torch.softmax(log_rho_nk, dim=1)


def e_step(x, alpha_k, beta_k, m_k, P_k, v_k, missing_data_mask=None):
    """
    Variational E-update: update local parameters
    Args:
        x: data
        alpha_k: Dirichlet parameter
        beta_k: NW param, variance of mean
        m_k: NW param, mean
        P_k: NW param, precision
        v_k: NW param, degrees of freedom
        missing_data_mask: binary matrix of shape (N, D) indicating missing values

    Returns:
        responsibilities and mixture coefficients
    """
    if missing_data_mask is None:
        expct_dev = compute_expct_mahalanobis_dist(x, beta_k, m_k, P_k, v_k)  # Bishop eq 10.64
    else:
        expct_dev = compute_dev_missing_data(x, beta_k, m_k, P_k, v_k, missing_data_mask)
    expct_log_det_cov = compute_expct_log_det_prec(v_k, P_k)  # Bishop eq 10.65
    expct_log_pi = compute_log_pi(alpha_k)  # Bishop eq 10.66
    r_nk = compute_rnk(expct_log_pi, expct_log_det_cov, expct_dev)  # Bishop eq 10.49

    return r_nk, torch.exp(expct_log_pi)


def m_step(x, r_nk, alpha_0, beta_0, m_0, C_0, v_0):
    """
    Variational M-update: Update global parameters
    Args:
        x: data
        r_nk: responsibilities
        alpha_0: prior Dirichlet parameters
        beta_0: prior NiW; controls variance of mean
        m_0: prior of mean
        C_0: prior Covariance
        v_0: prior degrees of freedom

    Returns:
        posterior parameters as well as data statistics
    """
    N_k = update_Nk(r_nk)  # Bishop eq 10.51
    x_k = update_xk(x, r_nk, N_k)  # Bishop eq 10.52
    S_k = update_Sk(x, r_nk, N_k, x_k)  # Bishop eq 10.53

    alpha_k = update_alphak(alpha_0, N_k)  # Bishop eq 10.58
    beta_k = update_betak(beta_0, N_k)  # Bishop eq 10.60
    m_k = update_mk(beta_0, m_0, N_k, x_k, beta_k)  # Bishop eq 10.61
    C_k = update_Ck(C_0, x_k, N_k, m_0, beta_0, beta_k, S_k)  # Bishop eq 10.62
    v_k = update_vk(v_0, N_k)  # Bishop eq 10.63

    return alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k


def inference(x, K, seed):
    """

    Args:
        x: data; shape = N, D
        K: number of components
        seed: random seed

    Returns:

    """
    N, D = x.shape

    r_nk = td.Dirichlet(torch.ones(K)).sample((N,))

    alpha, A, b, beta, v_hat = init_mm_params(K, D, alpha_scale=0.05 / K, beta_scale=0.5, m_scale=0,
                                              C_scale=D + 0.5, v_init=D + 0.5, trainable=False)
    beta_0, m_0, C_0, v_0 = niw_san.natural_to_standard(A, b, beta, v_hat)
    alpha_0 = dirichlet_san.natural_to_standard(alpha)

    alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = m_step(x, r_nk, alpha_0, beta_0, m_0, C_0, v_0)
    P_k = torch.inverse(C_k)
    r_nk_new, pi = e_step(x, alpha_k, beta_k, m_k, P_k, v_k)

    step = r_nk.assign(r_nk_new)

    theta = alpha_k, beta_k, m_k, C_k, v_k

    log_r_nk = torch.log(r_nk_new)

    return step, log_r_nk, theta, (x_k, S_k, pi)


def init_mm_params(nb_components, latent_dims, alpha_scale=.1, beta_scale=1e-5, v_init=10.,
                   m_scale=1., C_scale=10., trainable=False, device='cuda'):
    alpha_init = alpha_scale * torch.ones((nb_components,), device=device)
    beta_init = beta_scale * torch.ones((nb_components,), device=device)
    v_init = torch.full((nb_components,), float(latent_dims + v_init), device=device)
    means_init = m_scale * torch.empty(nb_components, latent_dims, device=device).uniform_(-1., 1.)
    covariance_init = C_scale * torch.eye(latent_dims, device=device).unsqueeze(0).repeat(
        nb_components, 1, 1)

    # transform to natural parameters
    A, b, beta, v_hat = niw_san.standard_to_natural(beta_init, means_init, covariance_init, v_init)
    alpha = dirichlet_san.standard_to_natural(alpha_init)

    params = alpha, A, b, beta, v_hat

    if trainable:
        for param in params:
            param.requires_grad_()

    return params
