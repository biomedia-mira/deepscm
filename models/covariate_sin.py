import torch
from torch import nn
from torch.distributions import Categorical

from ..distributions.factorised import Factorised
from .mixture import Mixture
from .sin import MixtureSIN


def _condition_factorised_mixture(mixture: Mixture[Factorised], values, which: int):
    cond_factor = mixture.components.factors[which]
    other_factors = [factor for k, factor in mixture.components.factors if k != which]
    cond_logits = mixture.mixing.logits + cond_factor.log_prob(values)
    cond_components = Factorised(other_factors) if len(other_factors) > 1 else other_factors[0]
    return Mixture(Categorical(logits=cond_logits), cond_components.expand(cond_logits.shape))


def _marginalise_factorised_mixture(mixture: Mixture[Factorised], which: int):
    return Mixture(mixture.mixing, mixture.components.factors[which])


def _rsample_factorised_mixture(mixture: Mixture[Factorised], sample_shape=torch.Size()):
    concat_samples = mixture.rsample(sample_shape)
    factor_samples = mixture.components.partition_dimensions(concat_samples)
    return factor_samples


class ConditionalDecoder(nn.Module):
    """A wrapper around a decoder with multiple inputs that holds one of its inputs fixed"""
    def __init__(self, decoder: nn.Module, values: torch.Tensor, which: int):
        self.decoder = decoder
        self.values = values
        self.which = which
        super().__init__()

    def forward(self, *inputs):
        inputs = list(inputs)
        inputs.insert(self.which, self.values)
        return self.decoder(*inputs)


class MixtureCovariateSIN(nn.Module):
    def __init__(self, prior_mixture: Mixture[Factorised], decoder, encoder, var_mixture: Mixture[Factorised], **fixed):
        self.prior_mixture = prior_mixture
        self.decoder = decoder
        self.encoder = encoder
        self.var_mixture = var_mixture
        self.fixed = fixed
        super().__init__()

    def forward(self, data):
        potentials = self.encoder.posterior(data)
        posteriors = self.var_mixture.posterior(potentials)
        covariates, latents = _rsample_factorised_mixture(posteriors)
        likelihoods = self.decoder.likelihood(covariates, latents)
        return potentials, posteriors, covariates, latents, likelihoods

    def rsample(self, sample_shape=torch.Size()):
        covariates, latents = _rsample_factorised_mixture(self.prior_mixture, sample_shape)
        data = self.decoder(covariates, latents)
        return data, covariates, latents

    def condition(self, *, data=None, covariates=None, latents=None):  # kwargs only
        if all(x is None for x in [data, covariates, latents]):
            raise ValueError
        if all(x is not None for x in [data, covariates, latents]):
            raise ValueError
        if data is None:
            if covariates is not None and latents is not None:  # p(data | covariates, latents)
                likelihoods = self.decoder.likelihood(covariates, latents)
                return likelihoods
            else:
                if covariates is None:  # p(data, covariates | latents)
                    values, which = latents, 1
                else:  # latents is None; p(data, latents | covariates)
                    values, which = covariates, 0
                cond_prior_mixture = _condition_factorised_mixture(self.prior_mixture, values, which=which)
                cond_var_mixture = _condition_factorised_mixture(self.var_mixture, values, which=which)
                cond_decoder = ConditionalDecoder(self.decoder, values, which=which)
                cond_sin = MixtureSIN(cond_prior_mixture, cond_decoder, self.encoder, cond_var_mixture)
                return cond_sin
        else:
            potentials = self.encoder.posterior(data)
            posteriors = self.var_mixture.posterior(potentials)  # p(covariates, latents | data)
            if covariates is not None:  # p(latents | data, covariates)
                cond_latents = _condition_factorised_mixture(posteriors, covariates, which=0)
                return cond_latents
            elif latents is not None:  # p(covariates | data, latents)
                cond_covariates = _condition_factorised_mixture(posteriors, latents, which=1)
                return cond_covariates
            else:  # p(covariates, latents | data)
                return posteriors
