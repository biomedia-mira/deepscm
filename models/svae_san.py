import torch

from distributions import dirichlet_san, gaussian_san, niw_san
from models import vae, gmm_san


def e_step(phi_enc, phi_gmm, nb_samples):
    """
    Args:
        phi_enc: encoded data; Gaussian natural parameters
        phi_gmm: paramters of recognition GMM (eta1_phi2, eta2_phi2, pi_phi2)
        nb_samples: number of times to sample from q(x|z, y)
        seed: random seed

    Returns:

    """
    eta1_phi1, eta2_phi1_diag = phi_enc
    eta2_phi1 = torch.diagflat(eta2_phi1_diag)

    # get gaussian natparams and dirichlet natparam for recognition GMM
    eta1_phi2, eta2_phi2, pi_phi2 = unpack_recognition_gmm(phi_gmm)

    # compute log q(z|y, phi)
    log_z_given_y_phi, dbg = compute_log_z_given_y(eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2,
                                                   pi_phi2)

    # compute parameters phi_tilde (corresponds to mu_tilde and sigma_tilde in the paper)
    # eta1_phi_tilde.shape = (N, K, D, 1); eta2_phi_tilde.shape = (N, K, D, D)
    eta1_phi_tilde = (eta1_phi1.unsqueeze(1) + eta1_phi2.unsqueeze(0)).unsqueeze(-1)
    eta2_phi_tilde = eta2_phi1.unsqueeze(1) + eta2_phi2.unsqueeze(0)
    phi_tilde = eta1_phi_tilde, eta2_phi_tilde

    # sample x for each of the K components
    # x_samps.shape = size_minibatch, nb_components, nb_samples, latent_dim
    x_k_samples = sample_x_per_comp(eta1_phi_tilde, eta2_phi_tilde, nb_samples)

    return x_k_samples, log_z_given_y_phi, phi_tilde, dbg


def compute_log_z_given_y(eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2, pi_phi2):
    """
    Args:
        eta1_phi1: encoder output; shape = N, K, L
        eta2_phi1: encoder output; shape = N, K, L, L
        eta1_phi2: GMM-EM parameter; shape = K, L
        eta2_phi2: GMM-EM parameter; shape = K, L, L

    Returns:
        log q(z|y, phi)
    """
    N, L = eta1_phi1.shape
    assert eta2_phi1.shape == (N, L, L)
    K, L2 = eta1_phi2.shape
    assert L2 == L
    assert eta2_phi2.shape == (K, L, L)

    # combine eta2_phi1 and eta2_phi2
    eta2_phi_tilde = eta2_phi1.unsqueeze(1) + eta2_phi2.unsqueeze(0)

    # w_eta2 = -0.5 * inv(sigma_phi1 + sigma_phi2)
    solved = torch.solve(eta2_phi2.unsqueeze(0).expand(N, -1, -1, -1), eta2_phi_tilde)[0]
    w_eta2 = torch.einsum('nju,nkui->nkij', eta2_phi1, solved)

    # for numerical stability...
    w_eta2 = .5 * (w_eta2 + w_eta2.transpose(-1, -2))

    # w_eta1 = inv(sigma_phi1 + sigma_phi2) * mu_phi2
    w_eta1 = torch.einsum('nuj,nkuv->nkj',
                          eta2_phi1,
                          torch.solve(eta1_phi2.unsqueeze(0).unsqueeze(-1).expand(N, -1, -1, -1),
                                      eta2_phi_tilde)[0]  # shape inside solve= N, K, D, 1
                          )  # w_eta1.shape = N, K, D

    # compute means
    mu_phi1, _ = gaussian_san.natural_to_standard(eta1_phi1, eta2_phi1)

    # compute log_z_given_y_phi
    return gaussian_san.log_probability_nat(mu_phi1, w_eta1, w_eta2, pi_phi2), (w_eta1, w_eta2)


def sample_x_per_comp(eta1, eta2, nb_samples):
    """
    Args:
        eta1: 1st Gaussian natural parameter, shape = N, K, L, 1
        eta2: 2nd Gaussian natural parameter, shape = N, K, L, L
        nb_samples: nb of samples to generate for each of the K components

    Returns:
        x ~ N(x|eta1[k], eta2[k]), nb_samples times for each of the K components.
    """
    inv_sigma = -2 * eta2
    N, K, _, D = eta2.shape

    # cholesky decomposition and adding noise (raw_noise is of dimension (DxB),
    # where B is the size of MC samples)
    L = torch.cholesky(inv_sigma)
    # sample_shape = (D, nb_samples)
    raw_noise = torch.randn(N, K, D, nb_samples)
    noise = torch.solve(raw_noise, L.transpose(-1, -2))[0]

    # reparam-trick-sampling: x_samps = mu_tilde + noise: shape = N, K, S, D
    x_k_samps = (torch.solve(eta1, inv_sigma)[0] + noise).permute(0, 1, 3, 2)
    return x_k_samps


def subsample_x(x_k_samples, log_q_z_given_y):
    """
    Given S samples for each of the K components for N datapoints (x_k_samples) and q(z_n=k|y),
    subsample S samples for each data point
    Args:
        x_k_samples: sample matrix of shape (N, K, S, L)
        log_q_z_given_y: probability q(z_n=k|y_n, phi)
    Returns:
        x_samples: a sample matrix of shape (N, S, L)
    """
    N, K, S, L = x_k_samples.shape

    # prepare indices for N and S dimension
    n_idx = torch.arange(N).unsqueeze(1).expand(N, S)
    s_idx = torch.arange(S).unsqueeze(0).expand(N, S)

    # sample S times z ~ q(z|y, phi) for each N.
    z_samps = torch.multinomial(log_q_z_given_y.exp(), num_samples=S)

    # tensor of shape (N, S, 3), containing indices of all chosen samples
    choices = torch.stack([n_idx, z_samps, s_idx], dim=2)

    return torch.gather(x_k_samples, 1, choices)


def m_step(gmm_prior, x_samples, r_nk):
    """
    Args:
        gmm_prior: Dirichlet+NiW prior for Gaussian mixture model
        x_samples: samples of shape (N, S, L)
        r_nk: responsibilities of shape (N, K)

    Returns:
        Dirichlet+NiW parameters obtained by executing Bishop's M-step in the VEM algorithm for GMMs
    """
    # execute GMM-EM m-step
    beta_0, m_0, C_0, v_0 = niw_san.natural_to_standard(*gmm_prior[1:])
    alpha_0 = dirichlet_san.natural_to_standard(gmm_prior[0])

    alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = gmm_san.m_step(x_samples, r_nk, alpha_0, beta_0, m_0,
                                                              C_0, v_0)

    A, b, beta, v_hat = niw_san.standard_to_natural(beta_k, m_k, C_k, v_k)
    alpha = dirichlet_san.standard_to_natural(alpha_k)

    return alpha, A, b, beta, v_hat


def compute_elbo(y, reconstructions, theta, phi_tilde, x_k_samps, log_z_given_y_phi, decoder_type):
    # ELBO for latent GMM
    # unpack phi_gmm and compute expected theta
    beta_k, m_k, C_k, v_k = niw_san.natural_to_standard(*theta[1:])
    mu, sigma = niw_san.expected_values((beta_k, m_k, C_k, v_k))
    eta1_theta, eta2_theta = gaussian_san.standard_to_natural(mu, sigma)
    alpha_k = dirichlet_san.natural_to_standard(theta[0])
    expected_log_pi_theta = dirichlet_san.expected_log_pi(alpha_k)

    # do not backpropagate through GMM
    eta1_theta = eta1_theta.detach()
    eta2_theta = eta2_theta.detach()
    expected_log_pi_theta = expected_log_pi_theta.detach()

    r_nk = log_z_given_y_phi.exp()

    # compute negative reconstruction error; sum over minibatch (use VAE function)
    means, out_2 = reconstructions  # out_2 is either gaussian variances or bernoulli logits.
    if decoder_type == 'standard':
        neg_reconstruction_error = vae.expected_diagonal_gaussian_loglike(y, means, out_2, weights=r_nk)
    elif decoder_type == 'bernoulli':
        neg_reconstruction_error = vae.expected_bernoulli_loglike(y, out_2, r_nk=r_nk)
    else:
        raise NotImplementedError

    # compute E[log q_phi(x,z=k|y)]
    eta1_phi_tilde, eta2_phi_tilde = phi_tilde
    N, K, L, _ = eta2_phi_tilde.shape
    eta1_phi_tilde = eta1_phi_tilde.view(N, K, L)

    N, K, S, L = x_k_samps.shape

    log_N_x_given_phi = gaussian_san.log_probability_nat_per_samp(x_k_samps, eta1_phi_tilde,
                                                                  eta2_phi_tilde)
    log_numerator = log_N_x_given_phi + log_z_given_y_phi.unsqueeze(2)

    log_N_x_given_theta = gaussian_san.log_probability_nat_per_samp(x_k_samps, eta1_theta, eta2_theta)
    log_denominator = log_N_x_given_theta + expected_log_pi_theta.view(1, -1, 1)

    regularizer_term = torch.mean(
        torch.sum(
            torch.sum(
                r_nk.unsqueeze(2) * (log_numerator - log_denominator),
                dim=1),  # weighted average over components
            dim=0)  # sum over minibatch
    )  # mean over samples

    elbo = neg_reconstruction_error - regularizer_term

    details = (neg_reconstruction_error,
               torch.sum(r_nk * log_numerator.mean(-1)),
               torch.sum(r_nk * log_denominator.mean(-1)),
               regularizer_term)

    return elbo, details


def unpack_recognition_gmm(phi_gmm):
    eta1, L_k_raw, pi_k_raw = phi_gmm

    # make sure that L is a valid Cholesky decomposition and compute precision
    L_k = torch.tril(L_k_raw)
    L_k = torch.matrix_set_diag(L_k, tf.nn.softplus(torch.diagonal(L_k)))
    P = torch.matmul(L_k, L_k.transpose(-1, -2))

    eta2 = -.5 * P

    # make sure that log_pi_k are valid mixture coefficients
    pi_k = torch.softmax(pi_k_raw, dim=-1)

    return eta1, eta2, pi_k


def update_gmm_params(current_gmm_params, gmm_params_star, step_size):
    """
    Computes convex combination between current and updated parameters.
    Args:
        current_gmm_params: current parameters
        gmm_params_star: parameters received by GMM-EM algorithm
        step_size: step size for convex combination

    Returns:
    """
    new_gmm_params = [(1. - step_size) * curr_param + step_size * param_star
                      for curr_param, param_star in zip(current_gmm_params, gmm_params_star)]
    return tuple(new_gmm_params)


def predict(y, phi_gmm, encoder_layers, decoder_layers):
    """
    Args:
        y: data to cluster and reconstruct
        phi_gmm: latent phi param
        encoder_layers: encoder NN architecture
        decoder_layers: encoder NN architecture

    Returns:
        reconstructed y and most probable cluster allocation
    """
    # encode (reusing current encoder parameters)
    nb_samples = 1
    phi_enc = vae.make_encoder(y, layerspecs=encoder_layers)

    # predict cluster allocation and sample latent variables (e-step)
    x_k_samples, log_r_nk, _, _ = e_step(phi_enc, phi_gmm, nb_samples)
    x_samples = subsample_x(x_k_samples, log_r_nk)[:, 0, :]

    # decode (reusing current decoder parameters)
    y_mean, _ = vae.make_decoder(x_samples, layerspecs=decoder_layers)

    return y_mean, log_r_nk.argmax(1)


def init_mm_params(nb_components, latent_dims, alpha_scale=.1, beta_scale=1e-5, v_init=10.,
                   m_scale=1., C_scale=10., trainable=False, device='cuda:0'):
    alpha_init = torch.full((nb_components,), alpha_scale)
    beta_init = torch.full((nb_components,), beta_scale)
    v_init = torch.full((nb_components,), float(latent_dims + v_init))
    means_init = m_scale * (2. * torch.rand(nb_components, latent_dims) - 1.)
    covariance_init = C_scale * torch.eye(latent_dims).unsqueeze(0).expand(nb_components, -1, -1)

    # transform to natural parameters
    A, b, beta, v_hat = niw_san.standard_to_natural(beta_init, means_init, covariance_init, v_init)
    alpha = dirichlet_san.standard_to_natural(alpha_init)

    # init variable
    if as_variables:
        for param in [alpha, A, b, beta, v_hat]:
            param.to(device).requires_grad_(trainable)

    params = alpha, A, b, beta, v_hat

    return params


def init_mm(nb_components, latent_dims, param_device='cuda:0'):
    # prior parameters are always tf.constant.
    theta_prior = init_mm_params(nb_components, latent_dims, alpha_scale=0.05 / nb_components,
                                 beta_scale=0.5,
                                 m_scale=0, C_scale=latent_dims + 0.5, v_init=latent_dims + 0.5,
                                 trainable=False, device=param_device)

    theta = init_mm_params(nb_components, latent_dims, alpha_scale=1., beta_scale=1.,
                           m_scale=5., C_scale=2. * latent_dims, v_init=latent_dims + 1.,
                           trainable=False)
    return theta_prior, theta


def make_loc_scale_variables(theta, param_device='cuda:0'):
    # create location/scale variables for point estimations
    with torch.no_grad():
        theta_copied = niw_san.natural_to_standard(*theta[1:])
    mu_k_init, sigma_k = niw_san.expected_values(theta_copied)
    L_k_init = torch.cholesky(sigma_k)

    mu_k = mu_k_init.to(param_device).requires_grad_()
    L_k = L_k_init.to(param_device).requires_grad_()

    return mu_k, L_k


def init_recognition_params(theta, nb_components, param_device='cuda:0'):
    # make parameters for PGM part of recognition network
    pi_k_init = torch.softmax(torch.randn(nb_components), dim=-1)

    mu_k, L_k = make_loc_scale_variables(theta, param_device)
    pi_k = pi_k_init.to(param_device).requires_grad_()
    return mu_k, L_k, pi_k


def inference(y, phi_gmm, encoder_layers, decoder_layers, nb_samples=10, stddev_init_nn=0.01,
              param_device='cuda:0'):
    # Use VAE encoder
    x_given_y_phi = vae.make_encoder(y, layerspecs=encoder_layers, stddev_init=stddev_init_nn,
                                     param_device=param_device)

    # execute E-step (update/sample local variables)
    x_k_samples, log_z_given_y_phi, phi_tilde, w_eta_12 = e_step(x_given_y_phi, phi_gmm, nb_samples)

    # compute reconstruction
    y_reconstruction = vae.make_decoder(x_k_samples, layerspecs=decoder_layers,
                                        stddev_init=stddev_init_nn, param_device=param_device)

    x_samples = subsample_x(x_k_samples, log_z_given_y_phi)[:, 0, :]

    return y_reconstruction, x_given_y_phi, x_k_samples, x_samples, log_z_given_y_phi, phi_gmm, phi_tilde


def identity_transform(input, nb_components, nb_samples, type='standard'):
    # debugging: freeze neural net: output Gaussian natparams corresponding to
    # mu_n=x_n and sigma_n = I * nn_var
    nn_var = 1e-1
    mu = input
    sigma = nn_var * torch.eye(2).unsqueeze(0).expand(mu.shape[0], -1, -1)

    if type == 'natparam':
        eta1, eta2 = gaussian_san.standard_to_natural(mu, sigma)
        eta2 = torch.diagonal(eta2)
        return eta1, eta2
    else:
        sigma = torch.diagonal(sigma)
        if sigma.shape != input.shape:
            N, D = sigma.shape
            sigma = sigma.view(N, 1, 1, D).expand(-1, nb_components, nb_samples, -1)
        return mu, sigma
