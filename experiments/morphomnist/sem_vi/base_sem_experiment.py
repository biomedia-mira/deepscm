import pyro

from typing import Mapping

from pyro.infer import SVI, TraceGraph_ELBO
from pyro.nn import pyro_method
from pyro.optim import Adam
from torch.distributions import Independent

import torch

import numpy as np

from experiments.morphomnist.base_experiment import BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401


class CustomELBO(TraceGraph_ELBO):
    # just do one step of regular elbo
    # condition on data (both guide and model) and change https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/tracegraph_elbo.py#L162-L169 from  - to +
    # ^ or simply go through traces and multiply by -1 if node is observed....!!
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = model_trace

        return model_trace, guide_trace


class BaseVISEM(BaseSEM):
    def __init__(self, hidden_dim: int, latent_dim: int, logstd_init: float = -5, **kwargs):
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

    def _get_preprocess_transforms(self):
        return super()._get_preprocess_transforms().inv

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
            exogeneous['z'] = z
            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=condition)(obs['x'].shape[0])
            counterfactuals += [counter]
        return {k: v for k, v in zip(('x', 'z', 'thickness', 'intensity'), (torch.stack(c).mean(0) for c in zip(*counterfactuals)))}

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--latent_dim', default=10, type=int, help="latent dimension of model (default: %(default)s)")
        parser.add_argument('--hidden_dim', default=100, type=int, help="hidden dimension of model (default: %(default)s)")
        parser.add_argument('--logstd_init', default=-5, type=float, help="init of logstd (default: %(default)s)")

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

        metrics['log p(x)'] = model.nodes['x']['log_prob_sum']
        metrics['log p(intensity)'] = model.nodes['intensity']['log_prob_sum']
        metrics['log p(thickness)'] = model.nodes['thickness']['log_prob_sum']
        metrics['log p(z) - log q(z)'] = model.nodes['z']['log_prob_sum'] - guide.nodes['z']['log_prob_sum']
        metrics['p(z)'] = model.nodes['z']['log_prob_sum']
        metrics['q(z)'] = guide.nodes['z']['log_prob_sum']

        return metrics

    def prep_batch(self, batch):
        x = batch['image']
        thickness = batch['thickness'].unsqueeze(1).float()
        intensity = batch['intensity'].unsqueeze(1).float()

        x = x.float()

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

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_scalar('decoder/decoder_logstd', self.pyro_model.decoder_logstd,  self.current_epoch)

        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(**batch)

        metrics = self.get_trace_metrics(batch)

        return {'loss': loss, **metrics}

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
