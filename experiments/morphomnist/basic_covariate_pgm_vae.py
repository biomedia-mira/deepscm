import torch
import pyro

from arch.mnist import Decoder, Encoder
from distributions.deep import DeepBernoulli, DeepIndepNormal, DeepIndepGamma, Conv2dIndepBeta

from pyro.nn import PyroParam, PyroModule, pyro_method
from pyro.distributions import Normal, Gamma
from torch.distributions import constraints

from experiments.morphomnist.base_experiment import BaseCovariateExperiment


class CovariatePGMVAE(PyroModule):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # TODO: This could be handled by passing a product distribution?

        # priors
        self.register_buffer('z_loc', torch.zeros([self.latent_dim], requires_grad=False))
        self.register_buffer('z_scale', torch.ones([self.latent_dim], requires_grad=False))

        # decoder parts
        self.decoder = Conv2dIndepBeta(Decoder(latent_dim + 2))

        self.thickness_conc = PyroParam(torch.ones([1]), constraint=constraints.positive)
        self.thickness_rate = PyroParam(torch.ones([1]), constraint=constraints.positive)

        self.slant_loc_mult = PyroParam(torch.zeros([1]))
        self.slant_scale = PyroParam(torch.ones([1]), constraint=constraints.positive)

        # encoder parts
        self.encoder = Encoder(hidden_dim)

        # TODO: do we need to replicate the PGM here to be able to run conterfactuals? oO
        self.latent_encoder = DeepIndepNormal(torch.nn.Identity(), hidden_dim, latent_dim)
        self.thickness_encoder = DeepIndepGamma(torch.nn.Identity(), hidden_dim, 1)
        self.slant_encoder = DeepIndepNormal(torch.nn.Identity(), hidden_dim, 1)

    @pyro_method
    def model(self):
        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        # TODO: How can we modularise this PGM?
        thickness = pyro.sample('thickness', Gamma(self.thickness_conc, self.thickness_rate).to_event(1))

        slant = pyro.sample('slant', Normal(thickness * self.slant_loc_mult, self.slant_scale).to_event(1))

        latent = torch.cat([z, thickness, slant], 1)

        x = pyro.sample('x', self.decoder.predict(latent))

        return x, z, thickness, slant

    def guide_encoder(self, x):
        hidden = self.encoder(x)
        latent_dist = self.latent_encoder.predict(hidden)
        thickness_dist = self.thickness_encoder.predict(hidden)
        slant_dist = self.slant_encoder.predict(hidden)

        return latent_dist, thickness_dist, slant_dist

    @pyro_method
    def guide(self, x):
        with pyro.plate('observations', x.shape[0]):
            latent_dist, thickness_dist, slant_dist = self.guide_encoder(x)

            z = pyro.sample('z', latent_dist)

            thickness = pyro.sample('thickness', thickness_dist)

            slant = pyro.sample('slant', slant_dist)
        return z, thickness, slant

    @pyro_method
    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            x, z, thickness, slant = self.model()

        return x, z, thickness, slant

    @pyro_method
    def svi_guide(self, x, thickness, slant):
        self.guide(x)

    @pyro_method
    def svi_model(self, x, thickness, slant):
        with pyro.plate('observations', x.shape[0]):
            pyro.condition(self.model, data={'x': x, 'thickness': thickness, 'slant': slant})()

    @pyro_method
    def reconstruct(self, x):
        guide_trace = pyro.poutine.trace(self.guide).get_trace(x)
        # z, thickness, slant = guide_trace.nodes['z']['value'], guide_trace.nodes['thickness']['value'], guide_trace.nodes['slant']['value']

        with pyro.plate('observations', x.shape[0]):
            # this works because the condition is on the input to the guide
            model_trace = pyro.poutine.trace(pyro.poutine.replay(self.model, trace=guide_trace)).get_trace()

        dist = model_trace.nodes['x']['fn'].base_dist

        return dist.mean

    def beta_mode(dist):
        conc0 = dist.concentration0
        conc1 = dist.concentration1

        mode = torch.zeros_like(conc0)

        conc0_geq_0 = (conc0 >= 0)
        conc1_geq_0 = (conc1 >= 0)

        mask = (conc0_geq_0 & conc1_geq_0)
        mode[mask] = (conc0[mask] - 1) / (conc0[mask] + conc1[mask] - 2)

        mode[~conc0_geq_0] = 0
        mode[~conc1_geq_0] = 1

        return mode

    @pyro_method
    def encode(self, x):
        z, thickness, slant = self.guide(x)

        return z, thickness, slant


class Experiment(BaseCovariateExperiment):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.pyro_model = CovariatePGMVAE(hidden_dim=hparams.hidden_dim, latent_dim=hparams.latent_dim)
        self._build_svi()


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(logger=True, checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    experiment_group.add_argument('--latent_dim', default=10, type=int, help="latent dimension of model (defaults to 10)")
    experiment_group.add_argument('--hidden_dim', default=100, type=int, help="hidden dimension of model (defaults to 100)")
    experiment_group.add_argument('--lr', default=1e-4, type=float, help="latent dimension of model (defaults to 1e-4)")
    experiment_group.add_argument('--validate', default=False, action='store_true', help="latent dimension of model (defaults to False)")

    args = parser.parse_args()

    # TODO: push to lightning
    args.gradient_clip_val = float(args.gradient_clip_val)

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)

    lightning_args = groups['lightning_options']
    hparams = groups['experiment']

    trainer = Trainer.from_argparse_args(lightning_args)

    experiment = Experiment(hparams)

    trainer.fit(experiment)
