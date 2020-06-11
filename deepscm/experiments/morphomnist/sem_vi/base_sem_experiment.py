import pyro

from typing import Mapping

from pyro.infer import SVI, TraceGraph_ELBO
from pyro.nn import pyro_method
from pyro.optim import Adam
from torch.distributions import Independent
from deepscm.arch.mnist import Decoder, Encoder
from pyro.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal, TransformedDistribution
from deepscm.distributions.transforms.reshape import ReshapeTransform
from deepscm.distributions.transforms.affine import LowerCholeskyAffine

from pyro.distributions.transforms import ComposeTransform, AffineTransform

from deepscm.distributions.deep import DeepMultivariateNormal, DeepIndepNormal, Conv2dIndepNormal, DeepLowRankMultivariateNormal

import torch

import numpy as np

from deepscm.experiments.morphomnist.base_experiment import BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401


class CustomELBO(TraceGraph_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace

        return model_trace, guide_trace


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class BaseVISEM(BaseSEM):
    context_dim = 0
    img_shape = (1, 28, 28)

    def __init__(self, hidden_dim: int, latent_dim: int, logstd_init: float = -5, decoder_type: str = 'fixed_var', decoder_cov_rank: int = 10, **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.logstd_init = logstd_init
        # TODO: This could be handled by passing a product distribution?

        # priors
        self.register_buffer('thickness_base_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('thickness_base_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('intensity_base_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('intensity_base_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('z_loc', torch.zeros([latent_dim, ], requires_grad=False))
        self.register_buffer('z_scale', torch.ones([latent_dim, ], requires_grad=False))

        self.register_buffer('x_base_loc', torch.zeros([1, 28, 28], requires_grad=False))
        self.register_buffer('x_base_scale', torch.ones([1, 28, 28], requires_grad=False))

        self.decoder_type = decoder_type
        self.decoder_cov_rank = decoder_cov_rank
        # 'fixed_var', 'learned_var', 'independent_gaussian', 'multivariate_gaussian'
        # TODO: This could be handled by passing a product distribution?

        # decoder parts
        decoder = self.decoder = Decoder(self.latent_dim + self.context_dim)

        if self.decoder_type == 'fixed_var':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)

            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = False

            torch.nn.init.constant_(self.decoder.logvar_head.bias, self.logstd_init)
            self.decoder.logvar_head.bias.requires_grad = False
        elif self.decoder_type == 'learned_var':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)

            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = False

            torch.nn.init.constant_(self.decoder.logvar_head.bias, self.logstd_init)
            self.decoder.logvar_head.bias.requires_grad = True
        elif self.decoder_type == 'independent_gaussian':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)

            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = True

            torch.nn.init.normal_(self.decoder.logvar_head.bias, self.logstd_init, 1e-1)
            self.decoder.logvar_head.bias.requires_grad = True
        elif self.decoder_type == 'multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape))
        elif self.decoder_type == 'sharedvar_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape))

            torch.nn.init.zeros_(self.decoder.logdiag_head.weight)
            self.decoder.logdiag_head.weight.requires_grad = False

            torch.nn.init.zeros_(self.decoder.lower_head.weight)
            self.decoder.lower_head.weight.requires_grad = False

            torch.nn.init.normal_(self.decoder.logdiag_head.bias, self.logstd_init, 1e-1)
            self.decoder.logdiag_head.bias.requires_grad = True
        elif self.decoder_type == 'lowrank_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepLowRankMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape), decoder_cov_rank)
        elif self.decoder_type == 'sharedvar_lowrank_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepLowRankMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape), decoder_cov_rank)

            torch.nn.init.zeros_(self.decoder.logdiag_head.weight)
            self.decoder.logdiag_head.weight.requires_grad = False

            torch.nn.init.zeros_(self.decoder.factor_head.weight)
            self.decoder.factor_head.weight.requires_grad = False

            torch.nn.init.normal_(self.decoder.logdiag_head.bias, self.logstd_init, 1e-1)
            self.decoder.logdiag_head.bias.requires_grad = True
        else:
            raise ValueError('unknown decoder type: {}'.format(self.decoder_type))

        # encoder parts
        self.encoder = Encoder(self.hidden_dim)

        # TODO: do we need to replicate the PGM here to be able to run conterfactuals? oO
        latent_layers = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim + self.context_dim, self.hidden_dim), torch.nn.ReLU())
        self.latent_encoder = DeepIndepNormal(latent_layers, self.hidden_dim, self.latent_dim)

    def _get_preprocess_transforms(self):
        return super()._get_preprocess_transforms().inv

    def _get_transformed_x_dist(self, latent):
        x_pred_dist = self.decoder.predict(latent)
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)

        preprocess_transform = self._get_preprocess_transforms()

        if isinstance(x_pred_dist, MultivariateNormal) or isinstance(x_pred_dist, LowRankMultivariateNormal):
            chol_transform = LowerCholeskyAffine(x_pred_dist.loc, x_pred_dist.scale_tril)
            reshape_transform = ReshapeTransform(self.img_shape, (np.prod(self.img_shape), ))
            x_reparam_transform = ComposeTransform([reshape_transform, chol_transform, reshape_transform.inv])
        elif isinstance(x_pred_dist, Independent):
            x_pred_dist = x_pred_dist.base_dist
            x_reparam_transform = AffineTransform(x_pred_dist.loc, x_pred_dist.scale, 3)

        return TransformedDistribution(x_base_dist, ComposeTransform([x_reparam_transform, preprocess_transform]))

    @pyro_method
    def guide(self, x, thickness, intensity):
        raise NotImplementedError()

    @pyro_method
    def svi_guide(self, x, thickness, intensity):
        self.guide(x, thickness, intensity)

    @pyro_method
    def counterfactual_guide(self, x, thickness, intensity, counterfactual_type=-1):
        if counterfactual_type == -1:
            counterfactual_type = np.random.randint(0, 3)

        num_samples = x.shape[0]

        with pyro.poutine.block(hide_all=True):
            # prepare conditioning
            obs = {'x': x, 'thickness': thickness, 'intensity': intensity}

            if counterfactual_type == 0:
                condition = {'intensity': intensity[torch.randperm(num_samples)]}
            elif counterfactual_type == 1:
                condition = {'thickness': thickness[torch.randperm(num_samples)]}
            elif counterfactual_type == 2:
                condition = {'thickness': thickness[torch.randperm(num_samples)], 'intensity': intensity[torch.randperm(num_samples)]}
            else:
                raise ValueError('counterfactual_type needs to be in [0, 1, 2] but got {}'.format(counterfactual_type))

            # get counterfactual
            counterfactual = self.counterfactual(obs=obs, condition=condition, num_particles=1)

        # run normal guide
        counterfactual.pop('z', None)

        return self.guide(**counterfactual)

    @pyro_method
    def svi_model(self, x, thickness, intensity):
        with pyro.plate('observations', x.shape[0]):
            pyro.condition(self.model, data={'x': x, 'thickness': thickness, 'intensity': intensity})()

    @pyro_method
    def infer_z(self, *args, **kwargs):
        return self.guide(*args, **kwargs)

    @pyro_method
    def infer(self, **obs):
        _required_data = ('x', 'thickness', 'intensity')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        z = self.infer_z(**obs)

        exogeneous = self.infer_exogeneous(z=z, **obs)
        exogeneous['z'] = z

        return exogeneous

    @pyro_method
    def reconstruct(self, x, thickness, intensity, num_particles: int = 1):
        obs = {'x': x, 'thickness': thickness, 'intensity': intensity}
        z_dist = pyro.poutine.trace(self.guide).get_trace(**obs).nodes['z']['fn']

        recons = []
        for _ in range(num_particles):
            z = pyro.sample('z', z_dist)
            recon, *_ = pyro.poutine.condition(self.sample, data={'thickness': thickness, 'intensity': intensity, 'z': z})(x.shape[0])
            recons += [recon]
        return torch.stack(recons).mean(0)

    @pyro_method
    def counterfactual(self, obs: Mapping, condition: Mapping = None, num_particles: int = 1):
        _required_data = ('x', 'thickness', 'intensity')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        z_dist = pyro.poutine.trace(self.guide).get_trace(**obs).nodes['z']['fn']

        counterfactuals = []
        for _ in range(num_particles):
            z = pyro.sample('z', z_dist)

            exogeneous = self.infer_exogeneous(z=z, **obs)
            exogeneous = {k: v.detach() for k, v in exogeneous.items()}
            exogeneous['z'] = z
            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=condition)(obs['x'].shape[0])
            counterfactuals += [counter]
        return {k: v for k, v in zip(('x', 'z', 'thickness', 'intensity'), (torch.stack(c).mean(0) for c in zip(*counterfactuals)))}

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--latent_dim', default=16, type=int, help="latent dimension of model (default: %(default)s)")
        parser.add_argument('--hidden_dim', default=100, type=int, help="hidden dimension of model (default: %(default)s)")
        parser.add_argument('--logstd_init', default=-5, type=float, help="init of logstd (default: %(default)s)")
        parser.add_argument(
            '--decoder_type', default='fixed_var', help="var type (default: %(default)s)",
            choices=['fixed_var', 'learned_var', 'independent_gaussian', 'sharedvar_multivariate_gaussian', 'multivariate_gaussian',
                     'sharedvar_lowrank_multivariate_gaussian', 'lowrank_multivariate_gaussian'])
        parser.add_argument('--decoder_cov_rank', default=10, type=int, help="rank for lowrank cov approximation (requires lowrank decoder) (default: %(default)s)")  # noqa: E501

        return parser


class SVIExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__(hparams, pyro_model)

        self.svi_loss = CustomELBO(num_particles=hparams.num_svi_particles)

        self._build_svi()

    def _build_svi(self, loss=None):
        def per_param_callable(module_name, param_name):
            params = {'eps': 1e-5, 'amsgrad': self.hparams.use_amsgrad, 'weight_decay': self.hparams.l2}
            if module_name == 'intensity_flow_components' or module_name == 'thickness_flow_components':
                params['lr'] = self.hparams.pgm_lr
                return params
            else:
                params['lr'] = self.hparams.lr
                return params

        if loss is None:
            loss = self.svi_loss

        if self.hparams.use_cf_guide:
            def guide(*args, **kwargs):
                return self.pyro_model.counterfactual_guide(*args, **kwargs, counterfactual_type=self.hparams.cf_elbo_type)
            self.svi = SVI(self.pyro_model.svi_model, guide, Adam(per_param_callable), loss)
        else:
            self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, Adam(per_param_callable), loss)
        self.svi.loss_class = loss

    def backward(self, *args, **kwargs):
        pass  # No loss to backpropagate since we're using Pyro's optimisation machinery

    def print_trace_updates(self, batch):
        print('Traces:\n' + ('#' * 10))

        guide_trace = pyro.poutine.trace(self.pyro_model.svi_guide).get_trace(**batch)
        model_trace = pyro.poutine.trace(pyro.poutine.replay(self.pyro_model.svi_model, trace=guide_trace)).get_trace(**batch)

        guide_trace = pyro.poutine.util.prune_subsample_sites(guide_trace)
        model_trace = pyro.poutine.util.prune_subsample_sites(model_trace)

        model_trace.compute_log_prob()
        guide_trace.compute_score_parts()

        print(f'model: {model_trace.nodes.keys()}')
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                fn = site['fn']
                if isinstance(fn, Independent):
                    fn = fn.base_dist
                print(f'{name}: {fn} - {fn.support}')
                log_prob_sum = site["log_prob_sum"]
                is_obs = site["is_observed"]
                print(f'model - log p({name}) = {log_prob_sum} | obs={is_obs}')
                if torch.isnan(log_prob_sum):
                    value = site['value'][0]
                    conc0 = fn.concentration0
                    conc1 = fn.concentration1

                    print(f'got:\n{value}\n{conc0}\n{conc1}')

                    raise Exception()

        print(f'guide: {guide_trace.nodes.keys()}')

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                fn = site['fn']
                if isinstance(fn, Independent):
                    fn = fn.base_dist
                print(f'{name}: {fn} - {fn.support}')
                entropy = site["score_parts"].entropy_term.sum()
                is_obs = site["is_observed"]
                print(f'guide - log q({name}) = {entropy} | obs={is_obs}')

    def get_trace_metrics(self, batch):
        metrics = {}

        model = self.svi.loss_class.trace_storage['model']
        guide = self.svi.loss_class.trace_storage['guide']

        metrics['log p(x)'] = model.nodes['x']['log_prob'].mean()
        metrics['log p(intensity)'] = model.nodes['intensity']['log_prob'].mean()
        metrics['log p(thickness)'] = model.nodes['thickness']['log_prob'].mean()
        metrics['p(z)'] = model.nodes['z']['log_prob'].mean()
        metrics['q(z)'] = guide.nodes['z']['log_prob'].mean()
        metrics['log p(z) - log q(z)'] = metrics['p(z)'] - metrics['q(z)']

        return metrics

    def prep_batch(self, batch):
        x = batch['image']
        thickness = batch['thickness'].unsqueeze(1).float()
        intensity = batch['intensity'].unsqueeze(1).float()

        x = x.float()

        if self.training:
            x += torch.rand_like(x)

        x = x.unsqueeze(1)

        return {'x': x, 'thickness': thickness, 'intensity': intensity}

    def training_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        if self.hparams.validate:
            print('Validation:')
            self.print_trace_updates(batch)

        loss = self.svi.step(**batch)

        metrics = self.get_trace_metrics(batch)

        if np.isnan(loss):
            self.logger.experiment.add_text('nan', f'nand at {self.current_epoch}:\n{metrics}')
            raise ValueError('loss went to nan with metrics:\n{}'.format(metrics))

        tensorboard_logs = {('train/' + k): v for k, v in metrics.items()}
        tensorboard_logs['train/loss'] = loss

        return {'loss': torch.Tensor([loss]), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(**batch)

        metrics = self.get_trace_metrics(batch)

        return {'loss': loss, **metrics}

    def test_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(**batch)

        metrics = self.get_trace_metrics(batch)

        samples = self.build_test_samples(batch)

        return {'loss': loss, **metrics, 'samples': samples}

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--num_svi_particles', default=4, type=int, help="number of particles to use for ELBO (default: %(default)s)")
        parser.add_argument('--num_sample_particles', default=32, type=int, help="number of particles to use for MC sampling (default: %(default)s)")
        parser.add_argument('--use_cf_guide', default=False, action='store_true', help="whether to use counterfactual guide (default: %(default)s)")
        parser.add_argument(
            '--cf_elbo_type', default=-1, choices=[-1, 0, 1, 2],
            help="-1: randomly select per batch, 0: shuffle thickness, 1: shuffle intensity, 2: shuffle both (default: %(default)s)")

        return parser


EXPERIMENT_REGISTRY[SVIExperiment.__name__] = SVIExperiment
