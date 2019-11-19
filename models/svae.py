import torch

from . import natural_gmm, vae
from .mixture import NaturalMultivariateNormalMixture

# TODO: Implement something like a TensorBundle that forwards operations to all elements?


def inference(data: torch.Tensor, var_mixture: NaturalMultivariateNormalMixture,
              encoder: vae.Encoder, decoder: vae.Decoder, n_samples=10):
    potentials = encoder.posterior(data)
    posteriors = var_mixture.posterior(potentials)
    latent_samples = posteriors.sample((n_samples,))
    reconstruction = decoder.likelihood(latent_samples)
    return potentials, posteriors, reconstruction, latent_samples


def init_mm(n_components, latent_dims, device='cuda:0'):
    theta_prior = natural_gmm.init(n_components, latent_dims, alpha_scale=0.05 / n_components,
                                   nu_scale=0.5, mean_scale=0, cov_scale=latent_dims + 0.5,
                                   dof_init=latent_dims + 0.5, requires_grad=False, device=device)

    theta = natural_gmm.init(n_components, latent_dims, alpha_scale=1., nu_scale=1., mean_scale=5.,
                             cov_scale=2. * latent_dims, dof_init=latent_dims + 1., requires_grad=False)
    return theta_prior, theta


def init_recognition_params(theta: natural_gmm.NaturalGMMPrior, n_components, device='cuda:0'):
    # make parameters for PGM part of recognition network
    pi_k_init = torch.softmax(torch.randn(n_components), dim=-1)

    # create location/scale variables for point estimations
    expected_component = theta.component_prior.mean.to_standard()
    mu_k_init, L_k_init = expected_component.loc, expected_component.scale_tril

    mu_k = mu_k_init.to(device).requires_grad_()
    L_k = L_k_init.to(device).requires_grad_()
    pi_k = pi_k_init.to(device).requires_grad_()

    return mu_k, L_k, pi_k
