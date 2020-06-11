import torch
import pyro

import numpy as np

from pyro.nn import pyro_method

from deepscm.experiments.morphomnist.base_experiment import BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401

from typing import Mapping


class BaseFlowSEM(BaseSEM):
    def __init__(self, num_scales: int = 4, flows_per_scale: int = 2, hidden_channels: int = 256,
                 use_actnorm: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.num_scales = num_scales
        self.flows_per_scale = flows_per_scale
        self.hidden_channels = hidden_channels
        self.use_actnorm = use_actnorm

        # priors
        self.register_buffer('thickness_base_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('thickness_base_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('intensity_base_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('intensity_base_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('x_base_loc', torch.zeros([1, 32, 32], requires_grad=False))
        self.register_buffer('x_base_scale', torch.ones([1, 32, 32], requires_grad=False))

    @pyro_method
    def infer(self, **obs):
        return self.infer_exogeneous(**obs)

    @pyro_method
    def counterfactual(self, obs: Mapping, condition: Mapping = None):
        _required_data = ('x', 'thickness', 'intensity')
        assert set(obs.keys()) == set(_required_data)

        exogeneous = self.infer(**obs)

        counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=condition)(obs['x'].shape[0])
        return {k: v for k, v in zip(('x', 'thickness', 'intensity'), counter)}

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--num_scales', default=4, type=int, help="number of scales (default: %(default)s)")
        parser.add_argument('--flows_per_scale', default=10, type=int, help="number of flows per scale (default: %(default)s)")
        parser.add_argument('--hidden_channels', default=256, type=int, help="number of hidden channels in convnet (default: %(default)s)")
        parser.add_argument('--use_actnorm', default=False, action='store_true', help="whether to use activation norm (default: %(default)s)")

        return parser


class NormalisingFlowsExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model: BaseSEM):
        hparams.latent_dim = 32 * 32

        super().__init__(hparams, pyro_model)

    def configure_optimizers(self):
        thickness_params = self.pyro_model.thickness_flow_components.parameters()
        intensity_params = self.pyro_model.intensity_flow_components.parameters()

        x_params = self.pyro_model.trans_modules.parameters()

        return torch.optim.Adam([
            {'params': x_params, 'lr': self.hparams.lr},
            {'params': thickness_params, 'lr': self.hparams.pgm_lr},
            {'params': intensity_params, 'lr': self.hparams.pgm_lr},
        ], lr=self.hparams.lr, eps=1e-5, amsgrad=self.hparams.use_amsgrad, weight_decay=self.hparams.l2)

    def prepare_data(self):
        super().prepare_data()

        self.z_range = self.z_range.reshape((9, 1, 32, 32))

    def get_logprobs(self, **obs):
        _required_data = ('x', 'thickness', 'intensity')
        assert set(obs.keys()) == set(_required_data)

        cond_model = pyro.condition(self.pyro_model.sample, data=obs)
        model_trace = pyro.poutine.trace(cond_model).get_trace(obs['x'].shape[0])
        model_trace.compute_log_prob()

        log_probs = {}
        nats_per_dim = {}
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                log_probs[name] = site["log_prob"].mean()
                log_prob_shape = site["log_prob"].shape
                value_shape = site["value"].shape
                if len(log_prob_shape) < len(value_shape):
                    dims = np.prod(value_shape[len(log_prob_shape):])
                else:
                    dims = 1.
                nats_per_dim[name] = -site["log_prob"].mean() / dims
                if self.hparams.validate:
                    print(f'at site {name} with dim {dims} and nats: {nats_per_dim[name]} and logprob: {log_probs[name]}')
                    if torch.any(torch.isnan(nats_per_dim[name])):
                        raise ValueError('got nan')

        return log_probs, nats_per_dim

    def prep_batch(self, batch):
        x = batch['image'].float()
        thickness = batch['thickness'].unsqueeze(1).float()
        intensity = batch['intensity'].unsqueeze(1).float()

        x = torch.nn.functional.pad(x, (2, 2, 2, 2))
        x += torch.rand_like(x)

        x = x.reshape(-1, 1, 32, 32)

        return {'x': x, 'thickness': thickness, 'intensity': intensity}

    def training_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(**batch)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        if torch.isnan(loss):
            self.logger.experiment.add_text('nan', f'nand at {self.current_epoch}')
            raise ValueError('loss went to nan')

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {('train/' + k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        tensorboard_logs = {'train/loss': loss, **nats_per_dim, **lls}

        return {'loss': loss, 'log': tensorboard_logs, **lls}

    def validation_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(**batch)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {(k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        return {'loss': loss, **lls, **nats_per_dim}

    def test_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(**batch)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {(k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        return {'loss': loss, **lls, **nats_per_dim}


EXPERIMENT_REGISTRY[NormalisingFlowsExperiment.__name__] = NormalisingFlowsExperiment
