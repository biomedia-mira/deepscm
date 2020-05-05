import torch
import pyro

import numpy as np

from pyro.nn import pyro_method
from torch.distributions import Independent
from pyro.distributions.transforms import ComposeTransform

from experiments.morphomnist.base_experiment import BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401
from experiments import PyroExperiment


class BaseFlowSEM(BaseSEM):
    def __init__(self, num_scales: int = 4, flows_per_scale: int = 2, hidden_channels: int = 256,
                 use_actnorm: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.num_scales = num_scales
        self.flows_per_scale = flows_per_scale
        self.hidden_channels = hidden_channels
        self.use_actnorm = use_actnorm

        # priors
        self.register_buffer('e_t_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_t_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_s_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_s_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_x_loc', torch.zeros([1, 32, 32], requires_grad=False))
        self.register_buffer('e_x_scale', torch.ones([1, 32, 32], requires_grad=False))

    @pyro_method
    def infer(self, x, thickness, slant):
        # assuming that we use transformed distributions for everything:
        cond_sample = pyro.condition(self.sample, data={'x': x, 'thickness': thickness, 'slant': slant})
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(x.shape[0])

        output = {}
        for (node, short) in [('thickness', 'e_t'), ('slant', 'e_s'), ('x', 'e_x')]:
            fn = cond_trace.nodes[node]['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            output[short] = ComposeTransform(fn.transforms).inv(cond_trace.nodes[node]['value'])

        return output

    @pyro_method
    def counterfactual(self, x, thickness, slant, data=None):
        exogeneous = self.infer(x, thickness, slant)

        counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=data)(x.shape[0])
        return (*counter,)

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--num_scales', default=4, type=int, help="number of scales (default: %(default)s)")
        parser.add_argument('--flows_per_scale', default=10, type=int, help="number of flows per scale (default: %(default)s)")
        parser.add_argument('--hidden_channels', default=256, type=int, help="number of hidden channels in convnet (default: %(default)s)")
        parser.add_argument('--use_actnorm', default=False, action='store_true', help="whether to use activation norm (default: %(default)s)")

        return parser


class NormalisingFlowsExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model):
        hparams.latent_dim = 32 * 32

        super().__init__(hparams, pyro_model)

    def configure_optimizers(self):
        thickness_params = self.pyro_model.t_flow_components.parameters()
        slant_params = self.pyro_model.s_flow_components.parameters()

        x_params = self.pyro_model.trans_modules.parameters()

        return torch.optim.Adam([
            {'params': x_params, 'lr': self.hparams.lr},
            {'params': thickness_params, 'lr': self.hparams.pgm_lr},
            {'params': slant_params, 'lr': self.hparams.pgm_lr},
        ], lr=self.hparams.lr, eps=1e-5, amsgrad=self.hparams.use_amsgrad, weight_decay=self.hparams.l2)

    def _get_parameters(self, *args, **kwargs):
        return super(PyroExperiment, self)._get_parameters(*args, **kwargs)

    def backward(self, *args, **kwargs):
        return super(PyroExperiment, self).backward(*args, **kwargs)

    def prepare_data(self):
        super().prepare_data()

        self.z_range = self.z_range.reshape((9, 1, 32, 32))

    def get_logprobs(self, x, thickness, slant):
        data = {'x': x, 'thickness': thickness, 'slant': slant}
        cond_model = pyro.condition(self.pyro_model.sample, data=data)
        model_trace = pyro.poutine.trace(cond_model).get_trace(x.shape[0])
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
        slant = batch['slant'].unsqueeze(1).float()

        x = torch.nn.functional.pad(x, (2, 2, 2, 2))
        x += torch.rand_like(x)

        x = x.reshape(-1, 1, 32, 32)

        return x, thickness, slant

    def training_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(x, thickness, slant)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        if torch.isnan(loss):
            self.logger.experiment.add_text('nan', f'nand at {self.current_epoch}')
            raise ValueError('loss went to nan')

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {('train/' + k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        tensorboard_logs = {'train/loss': loss, **nats_per_dim, **lls}

        return {'loss': loss, 'log': tensorboard_logs, **lls}

    def validation_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(x, thickness, slant)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {(k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        return {'loss': loss, **lls, **nats_per_dim}

    def test_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(x, thickness, slant)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {(k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        return {'loss': loss, **lls, **nats_per_dim}

    def build_counterfactual(self, tag, x, thickness, slant, conditions, absolute=None):
        imgs = [x]
        if absolute == 'thickness':
            sampled_kdes = {'orig': {'slant': slant}}
        elif absolute == 'slant':
            sampled_kdes = {'orig': {'thickness': thickness}}
        else:
            sampled_kdes = {'orig': {'thickness': thickness, 'slant': slant}}
        measured_kdes = {'orig': {'thickness': thickness, 'slant': slant}}

        for name, data in conditions.items():
            counter, sampled_thickness, sampled_slant = self.pyro_model.counterfactual(
                x, thickness, slant, data=data)

            measured_thickness, measured_slant = self.measure_image(counter.cpu())

            imgs.append(counter)
            if absolute == 'thickness':
                sampled_kdes[name] = {'slant': sampled_slant}
            elif absolute == 'slant':
                sampled_kdes[name] = {'thickness': sampled_thickness}
            else:
                sampled_kdes[name] = {'thickness': sampled_thickness, 'slant': sampled_slant}
            measured_kdes[name] = {'thickness': measured_thickness, 'slant': measured_slant}

            self.logger.experiment.add_scalar(
                f'{tag}/{name}/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar(
                f'{tag}/{name}/slant_mae', torch.mean(torch.abs(sampled_slant.cpu() - measured_slant)), self.current_epoch)

        self.log_img_grid(tag, torch.cat(imgs, 0))
        self.log_kdes(f'{tag}_sampled', sampled_kdes, save_img=True)
        self.log_kdes(f'{tag}_measured', measured_kdes, save_img=True)

    def sample_images(self):
        with torch.no_grad():
            samples, sampled_thickness, sampled_slant = self.pyro_model.sample(128)
            self.log_img_grid('samples', samples.data[:8])

            measured_thickness, measured_slant = self.measure_image(samples)
            self.logger.experiment.add_scalar('samples/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('samples/slant_mae', torch.mean(torch.abs(sampled_slant.cpu() - measured_slant)), self.current_epoch)

            samples, *_ = pyro.condition(self.pyro_model.sample, data={'thickness': self.thickness_range, 'slant': self.slant_range, 'z': self.z_range})(9)
            self.log_img_grid('cond_samples', samples.data, nrow=3)

            x, thickness, slant = self.prep_batch(self.get_batch(self.val_loader))

            kde_data = {
                'batch': {'thickness': thickness, 'slant': slant},
                'sampled': {'thickness': sampled_thickness, 'slant': sampled_slant}
            }
            self.log_kdes('sample_kde', kde_data, save_img=True)

            exogeneous = self.pyro_model.infer(x, thickness, slant)

            self.logger.experiment.add_histogram('e_t', exogeneous['e_t'], self.current_epoch)
            self.logger.experiment.add_histogram('e_s', exogeneous['e_s'], self.current_epoch)
            self.logger.experiment.add_histogram('e_x', exogeneous['e_x'], self.current_epoch)

            x = x[:8]
            thickness = thickness[:8]
            slant = slant[:8]

            self.log_img_grid('input', x, save_img=True)

            vals = [x, thickness, slant]

            conditions = {
                '+1': {'thickness': thickness + 1},
                '+2': {'thickness': thickness + 2},
                '+3': {'thickness': thickness + 3}
            }
            self.build_counterfactual('do(thickess=+x)', *vals, conditions=conditions)

            conditions = {
                '1': {'thickness': torch.ones_like(thickness)},
                '2': {'thickness': torch.ones_like(thickness) * 2},
                '3': {'thickness': torch.ones_like(thickness) * 3}
            }
            self.build_counterfactual('do(thickess=x)', *vals, conditions=conditions, absolute='thickness')

            conditions = {
                '-45': {'slant': slant - 45},
                '-25': {'slant': slant - 25},
                '+25': {'slant': slant + 25},
                '+45': {'slant': slant + 45}
            }
            self.build_counterfactual('do(slant=+x)', *vals, conditions=conditions)

            conditions = {
                '-45': {'slant': torch.zeros_like(slant) - 45},
                '-25': {'slant': torch.zeros_like(slant) - 25},
                '0': {'slant': torch.zeros_like(slant)},
                '+25': {'slant': torch.zeros_like(slant) + 25},
                '+45': {'slant': torch.zeros_like(slant) + 45}
            }
            self.build_counterfactual('do(slant=x)', *vals, conditions=conditions, absolute='slant')


EXPERIMENT_REGISTRY[NormalisingFlowsExperiment.__name__] = NormalisingFlowsExperiment
