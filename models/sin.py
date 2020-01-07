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

    def forward(self, data, sample_all_components=False):
        potentials = self.enc.posterior(data)

        _ = torch.cholesky(potentials.covariance_matrix.cpu())
        posteriors = self.var_mixture.posterior(potentials)
        if sample_all_components:
            latents = posteriors.components.rsample()  # batch shape: [N, K, D]
            likelihoods = self.dec.likelihood(latents)  # batch shape: [N, K]
        else:
            latents = posteriors.rsample()  # batch shape:  [N, D]
            likelihoods = self.dec.likelihood(latents)  # batch shape: [N, ]
        return potentials, posteriors, latents, likelihoods


class Trainer(object):
    def __init__(self, model: MixtureSIN, lr: float = 1e-4, sample_all_components: bool = False):
        super().__init__()
        self.model = model
        self.sample_all_components = sample_all_components

        params = list(self.model.enc.parameters()) + list(self.model.dec.parameters()) + list(self.model.var_mixture.parameters())
        self.opt = torch.optim.Adam(params, lr=lr, betas=(.5, .99), eps=1e-5)

    def step(self, data, verbose=False):
        potentials, posteriors, latents, likelihoods = self.model(data, sample_all_components=self.sample_all_components)

        if self.sample_all_components:
            # data - [N, D_img]
            # likelihoods batch: [N, K], event: [D_img]
            # posterior batch: [N], event: [D]
            # prior: [], [D]
            # latents [N, K, D]
            log_likelihood = likelihoods.log_prob(data.unsqueeze(1))  # [N, K]
            lat_trans = latents.transpose(1, 0)
            log_prior = self.model.prior_mixture.log_prob(latents)  # [N, K]
            log_var_posterior = posteriors.log_prob(lat_trans).transpose(1, 0)  # [N, K]

            # Loss = E[log p(x|z) + log p(z) - log q(z|x)]
            elbo = log_likelihood + log_prior - log_var_posterior

            elbo = (elbo * torch.softmax(log_var_posterior, -1)).sum(-1).mean()
        else:
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
