from typing import Mapping, Sequence

import torch
from torch import nn
from torch.distributions import Distribution

from distributions.params import MultivariateParams, MixtureParams
from distributions.multivariate import MultivariateDistribution


def _partition(value, ndims: Sequence[int]):
    def generator(x):
        for ndim in ndims:
            yield x[..., :ndim]
            x = x[..., ndim:]
    return list(generator(value))


def _interleave_parts(x_parts, x_indices, y_parts, y_indices):
    full_len = len(x_parts) + len(y_parts)
    assert all(i in x_indices + y_indices for i in range(full_len))
    parts_dict = dict(zip(x_indices, x_parts))
    parts_dict.update(dict(zip(y_indices, y_parts)))
    all_parts = [parts_dict[i] for i in range(full_len)]
    return all_parts


class Interleaver:
    def __init__(self, cond_dict: Mapping[str, torch.Tensor], var_names: Sequence[str],
                 var_dims: Sequence[int]):
        self.cond_dict = cond_dict
        self.var_dims = var_dims
        self.cond_indices = [i for i, name in enumerate(var_names) if name in cond_dict]
        self.input_indices = [i for i, name in enumerate(var_names) if name not in cond_dict]
        self.input_dims = [var_dims[i] for i in self.input_indices]
        self.cond_parts = [cond_dict[var_names[i]] for i in self.cond_indices]

    def __call__(self, inputs):
        input_parts = _partition(inputs, self.input_dims)
        cond_parts = [part.expand(inputs.shape[:-len(part.shape)] + part.shape)
                      for part in self.cond_parts]
        all_parts = _interleave_parts(input_parts, self.input_indices, cond_parts, self.cond_indices)
        return torch.cat(all_parts, dim=-1)


class ProbabilisticEncoder:
    def posterior(self, data: torch.Tensor) -> MultivariateDistribution:
        raise NotImplementedError


class ProbabilisticDecoder:
    def likelihood(self, latents: torch.Tensor) -> Distribution:
        raise NotImplementedError


class ConditionalEncoder(ProbabilisticEncoder):
    def __init__(self, encoder: ProbabilisticEncoder, cond_dict, squeeze):
        self._encoder = encoder
        self._cond_dict = cond_dict
        self._squeeze = squeeze

    def posterior(self, data):
        posterior = self._encoder.posterior(data)
        return posterior.condition(self._cond_dict, self._squeeze)


class ConditionalDecoder(ProbabilisticDecoder):
    def __init__(self, decoder: ProbabilisticDecoder, interleaver: Interleaver):
        self._decoder = decoder
        self._interleaver = interleaver

    def likelihood(self, latents):
        full_latents = self._interleaver(latents)
        return self._decoder.likelihood(full_latents)


class MixtureSIN(ProbabilisticEncoder, nn.Module):
    def __init__(self, encoder: ProbabilisticEncoder, mixture_params: MixtureParams):
        super().__init__()
        self.encoder = encoder
        self.mixture_params = mixture_params

    def posterior(self, data):
        potentials = self.encoder.posterior(data)
        mixture = self.mixture_params.get_distribution()
        posteriors = mixture.posterior(potentials)  # q(latents | data)
        return posteriors


class SVAE(MultivariateDistribution, nn.Module):
    def __init__(self, prior_params: MultivariateParams,  # p(latents)
                 decoder: ProbabilisticDecoder,  # p(data | latents)
                 encoder: ProbabilisticEncoder):  # q(latents | data)
        var_names = ['data'] + prior_params.variable_names
        self._num_variables = len(var_names)
        super().__init__(var_names=var_names)
        self.prior_params = prior_params
        self.decoder = decoder
        self.encoder = encoder

    @property
    def num_variables(self):
        return self._num_variables

    def encode(self, data):
        return self.encoder.posterior(data)

    def decode(self, latents):
        return self.decoder.likelihood(latents)

    def forward(self, data):
        posteriors = self.encode(data)
        latents = posteriors.rsample()
        likelihoods = self.decode(latents)
        return posteriors, latents, likelihoods

    def rsample(self, sample_shape=torch.Size()):
        prior = self.prior_params.get_distribution()
        latents = prior.rsample(sample_shape)
        likelihoods = self.decode(latents)
        data = likelihoods.mean
        return data, latents

    def marginalise(self, which):
        import numpy as np

        if (np.ndim(which) == 1 and which == 'data') or 'data' in which:
            raise NotImplementedError("Marginals involving the data are intractable")
        else:
            return self.prior_params.marginalise(which)

    def condition(self, cond_dict, squeeze=False):
        if 'data' in cond_dict:
            data = cond_dict.pop('data')  # Removes 'data' from dictionary
            posteriors = self.encode(data)  # q(latents | data)
            return posteriors.condition(cond_dict, squeeze)
        else:
            prior = self.prior_params.get_distribution()
            var_names = prior.variable_names
            cond_all_variables = set(cond_dict.keys()) == set(var_names)
            if cond_all_variables:  # p(data | latents)
                decoder_inputs = torch.cat([cond_dict[name] for name in var_names], -1)
                return self.decode(decoder_inputs)
            else:  # p(data, some_latents | other_latents)
                var_dims = prior.variable_shapes
                interleaver = Interleaver(cond_dict, var_names, var_dims)

                cond_prior = prior.condition(cond_dict)
                cond_prior_params = self.prior_params.from_distribution(cond_prior)
                cond_decoder = ConditionalDecoder(self.decoder, interleaver)
                cond_encoder = ConditionalEncoder(self.encoder, cond_dict, squeeze)
                return SVAE(cond_prior_params, cond_decoder, cond_encoder)


class Trainer(object):
    def __init__(self, model: SVAE, lr: float = 1e-4, sample_all_components: bool = False,
                 verbose: bool = False):
        super().__init__()
        self.model = model
        self.sample_all_components = sample_all_components

        # TODO: Different optimisers/learning rates for encoder/decoder/prior?
        params = list(self.model.parameters())
        self.opt = torch.optim.Adam(params, lr=lr, betas=(.5, .99), eps=1e-5)
        self.losses = None

    def step(self, data, verbose=False):
        # posteriors, latents, likelihoods = self.model(data, sample_all_components=self.sample_all_components)

        prior = self.model.prior_params.get_distribution()  # batch shape: []
        posteriors = self.model.encode(data)  # batch shape: [N]

        if self.sample_all_components:
            # data - [N, D_img]
            # likelihoods batch: [N, K], event: [D_img]
            # posterior batch: [N], event: [D]
            # prior: [], [D]
            # latents [N, K, D]

            latents = posteriors.components.rsample()  # [N, K, D]
            likelihoods = self.model.decode(latents)  # batch shape: [N, K]

            log_likelihood = likelihoods.log_prob(data.unsqueeze(1))  # [N, K]
            log_prior = prior.log_prob(latents)  # [N, K]
            log_var_posterior = posteriors.log_prob(latents.transpose(1, 0)).transpose(1, 0)  # [N, K]
            weights = torch.softmax(log_var_posterior, -1)  # [N, K]

            # Loss = E[log p(x|z) + log p(z) - log q(z|x)]
            elbo = log_likelihood + log_prior - log_var_posterior

            # Loss = E_q(c|x)[E_q(z|x,c)[log p(x|z,c) + log p(z|c) - log q(z|x,c)]]
            elbo = (elbo * weights).sum(-1).mean()
        else:
            latents = posteriors.rsample()  # [N, D]
            likelihoods = self.model.decode(latents)  # batch shape: [N]

            log_likelihood = likelihoods.log_prob(data).mean()
            log_prior = prior.log_prob(latents).mean()
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
