import torch
from torch import nn
import torchvision

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

        posteriors = self.var_mixture.posterior(potentials)
        if sample_all_components:
            latents = posteriors.components.rsample()  # batch shape: [N, K, D]
            likelihoods = self.dec.likelihood(latents)  # batch shape: [N, K]
        else:
            latents = posteriors.rsample()  # batch shape:  [N, D]
            likelihoods = self.dec.likelihood(latents)  # batch shape: [N, ]
        return potentials, posteriors, latents, likelihoods


class Trainer(object):
    def __init__(self, model: MixtureSIN, lr: float = 1e-4, sample_all_components: bool = False, verbose: bool = False):
        super().__init__()
        self.model = model
        self.sample_all_components = sample_all_components

        params = list(self.model.enc.parameters()) + list(self.model.dec.parameters()) + list(self.model.var_mixture.parameters())
        self.opt = torch.optim.Adam(params, lr=lr, betas=(.5, .99), eps=1e-5)
        self.losses = None

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

        if verbose:
            print(f"elbo = {elbo.item():6g}")

        losses = {'elbo': elbo}
        if self.losses is None:
            self.losses = losses
        else:
            for key, value in losses.items():
                self.losses[key] += value

    def forward(self, real_data, verbose: bool = False):
        self.step(real_data, verbose)

    def get_and_reset_losses(self):
        losses = self.losses.copy()
        self.losses = None
        return losses


class Tester(object):
    def __init__(self, model: MixtureSIN, device: torch.device, use_double: bool):
        self.model = model
        self.device = device
        self.use_double = use_double

    def step(self, real_data):
        self.model.eval()
        with torch.no_grad():
            real_data = real_data.to(self.device).unsqueeze(1)
            if self.use_double:
                real_data = real_data.double() / 255.
            else:
                real_data = real_data.float() / 255.
            recon_data = self.model(real_data)[-1]
            samples = self.model.dec(self.model.prior_mixture.sample((len(real_data),)))
            stacked = torchvision.utils.make_grid(torch.cat((real_data, recon_data.mean, recon_data.stddev)),
                                                  nrow=len(real_data))
            return {'data/real': real_data,
                    'data/recon_mean': recon_data.mean,
                    'data/recon_stddev': recon_data.stddev,
                    'data_stacked': stacked,
                    'samples': samples}