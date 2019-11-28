from torch import nn

from .mixture import Mixture


class MixtureSIN(nn.Module):
    def __init__(self, prior_mixture: Mixture, decoder, encoder, var_mixture: Mixture):
        self.prior_mixture = prior_mixture
        self.decoder = decoder
        self.encoder = encoder
        self.var_mixture = var_mixture
        super().__init__()

    def forward(self, data):
        potentials = self.encoder.posterior(data)
        posteriors = self.var_mixture.posterior(potentials)
        latents = posteriors.rsample()
        likelihoods = self.decoder.likelihood(latents)
        return potentials, posteriors, latents, likelihoods


class SINTrainer(nn.Module):
    def __init__(self, model: MixtureSIN, opt):
        super().__init__()
        self.model = model
        self.opt = opt

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
