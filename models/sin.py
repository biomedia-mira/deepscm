import torch.distributions as td
from torch import nn

from .mixture import NaturalMultivariateNormalMixture
from distributions.natural_mvn import NaturalMultivariateNormal


class MixtureSIN(nn.Module):
    def __init__(self, prior_mixture: NaturalMultivariateNormalMixture, decoder, encoder,
                 var_mixture: NaturalMultivariateNormalMixture):
        self.prior_mixture = prior_mixture
        self.decoder = decoder
        self.encoder = encoder
        self.var_mixture = var_mixture
        super().__init__()

    def forward(self, data):
        potentials = self.encoder(data)  # type: NaturalMultivariateNormal
        posteriors = self.var_mixture.posterior(potentials)
        latents = posteriors.rsample()


class SINTrainer(nn.Module):
    def __init__(self, model: MixtureSIN):
        self.model = model
        super().__init__()

    def step(self, data, verbose=False):
        pass

    def forward(self, real_data, verbose: bool = False):
        self.step(real_data, verbose)
