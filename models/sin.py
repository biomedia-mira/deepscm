import torch
from torch import nn

from .mixture import Mixture


class MixtureSIN(nn.Module):
    def __init__(self, prior_mixture: Mixture, encoder, decoder, var_mixture: Mixture):
        super().__init__()
        self.prior_mixture = prior_mixture
        self.dec = decoder
        self.enc = encoder
        self.var_mixture = var_mixture

    def forward(self, data):
        potentials = self.enc.posterior(data)
        posteriors = self.var_mixture.posterior(potentials)
        latents = posteriors.rsample()
        likelihoods = self.dec.likelihood(latents)
        return potentials, posteriors, latents, likelihoods


class Trainer(object):
    def __init__(self, model: MixtureSIN, lr: float = 1e-4):
        super().__init__()
        self.model = model

        params = list(self.model.enc.parameters()) + list(self.model.dec.parameters()) + list(self.model.var_mixture.parameters())
        self.opt = torch.optim.Adam(params, lr=lr, betas=(.5, .99), eps=1e-5)

    def step(self, data, verbose=False):
        potentials, posteriors, latents, likelihoods = self.model(data)

        log_likelihood = likelihoods.log_prob(data).mean()
        log_prior = self.model.prior_mixture.log_prob(latents).mean()
        log_var_posterior = posteriors.log_prob(latents).mean()

        # Loss = E[log p(x|z) + log p(z) - log q(z|x)]
        elbo = log_likelihood + log_prior - log_var_posterior

        self.opt.zero_grad()
        (-elbo).backward()
        self.opt.step()

    def forward(self, real_data, verbose: bool = False):
        self.step(real_data, verbose)
