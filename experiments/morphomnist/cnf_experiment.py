import torch
import pyro

from pyro.nn import pyro_method

from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
from pyro.distributions.score_parts import ScoreParts

from experiments.morphomnist.basic_covariate_pgm_vae import Experiment, CovariatePGMVAE
from pyro.distributions.util import is_identically_zero
import warnings
from experiments.morphomnist.base_experiment import BaseCovariateExperiment
from pyro.nn import PyroParam, PyroModule, pyro_method

from pyro.distributions import Gamma, Normal, TransformedDistribution
from pyro.distributions.torch_transform import TransformModule, ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.distributions.transforms import (
    Spline, ExpTransform, ComposeTransform, ConditionalAffineCoupling,
    Permute, SigmoidTransform, GeneralizedChannelPermute, AffineTransform
)
import torchvision
from experiments import PyroExperiment
from distributions.transforms.reshape import ReshapeTransform, SqueezeTransform, TransposeTransform
from distributions.transforms.affine import ConditionalAffineTransform
from arch.mnist import BasicFlowConvNet
from pyro.nn import DenseNN
from distributions.transforms.normalisation import ActNorm
import numpy as np


class FlowModel(PyroModule):
    def __init__(self, num_scales: int = 4, flows_per_scale: int = 2, preprocessing: str = 'realnvp', hidden_channels: int = 256, use_actnorm: bool = False):
        super().__init__()
        self.num_scales = num_scales
        self.flows_per_scale = flows_per_scale
        self.preprocessing = preprocessing
        self.hidden_channels = hidden_channels
        self.use_actnorm = use_actnorm

        # TODO: This could be handled by passing a product distribution?

        # priors
        self.register_buffer('e_t_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_t_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_s_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_s_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_x_loc', torch.zeros([1, 32, 32], requires_grad=False))
        self.register_buffer('e_x_scale', torch.ones([1, 32, 32], requires_grad=False))

        # decoder parts
        # TODO:
        # Flow for modelling t Gamma
        self.t_flow_components = ComposeTransformModule([Spline(1)])
        self.t_flow_transforms = ComposeTransform([self.t_flow_components, ExpTransform()])

        # affine flow for s normal
        slant_net = DenseNN(1, [1], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.s_flow_components = ConditionalAffineTransform(context_nn=slant_net, event_dim=0)
        self.s_flow_transforms = [self.s_flow_components]
        # build flow as s_affine_w * t * e_s + b -> depends on t though

        # realnvp or so for x
        self._build_image_flow()

    def _build_image_flow(self):
        alpha = 0.05
        num_bits = 8

        self.trans_modules = ComposeTransformModule([])

        self.x_transforms = []

        if self.preprocessing == 'glow':
            # Map to [-0.5,0.5]
            a1 = AffineTransform(-0.5, (1. / 2 ** num_bits))
            preprocess_transform = [a1]
        elif self.preprocessing == 'realnvp':
            # Map to [0,1]
            a1 = AffineTransform(0., (1. / 2 ** num_bits))

            # Map into unconstrained space as done in RealNVP
            a2 = AffineTransform(alpha, (1 - alpha))

            s = SigmoidTransform()

            preprocess_transform = [a1, a2, s.inv]

        self.x_transforms += preprocess_transform

        c = 1
        for scale in range(self.num_scales):
            self.x_transforms.append(SqueezeTransform())
            c *= 4

            for _ in range(self.flows_per_scale):
                if self.use_actnorm:
                    actnorm = ActNorm(c)
                    self.trans_modules.append(actnorm)
                    self.x_transforms.append(actnorm)

                gcp = GeneralizedChannelPermute(channels=c)
                self.trans_modules.append(gcp)
                self.x_transforms.append(gcp)

                self.x_transforms.append(TransposeTransform(torch.tensor((1, 2, 0))))

                ac = ConditionalAffineCoupling(c // 2, BasicFlowConvNet(c // 2, self.hidden_channels, (c // 2, c // 2), 2))
                self.trans_modules.append(ac)
                self.x_transforms.append(ac)

                self.x_transforms.append(TransposeTransform(torch.tensor((2, 0, 1))))

            gcp = GeneralizedChannelPermute(channels=c)
            self.trans_modules.append(gcp)
            self.x_transforms.append(gcp)

        self.x_transforms += [
            ReshapeTransform((4**self.num_scales, 32 // 2**self.num_scales, 32 // 2**self.num_scales), (1, 32, 32))
        ]

        affine_net = DenseNN(2, [16, 16], param_dims=[1, 1])
        affine_trans = ConditionalAffineTransform(context_nn=affine_net, event_dim=3)

        self.trans_modules.append(affine_trans)
        self.x_transforms.append(affine_trans)

    @pyro_method
    def model(self):
        # TODO: disentangle PGM from images
        t_bd = Normal(self.e_t_loc, self.e_t_scale)
        t_dist = TransformedDistribution(t_bd, self.t_flow_transforms)

        thickness = pyro.sample('thickness', t_dist.to_event(1))

        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        s_dist = ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness)

        slant = pyro.sample('slant', s_dist.to_event(1))

        context = torch.cat([thickness / 3., slant / 30.], 1)

        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms).inv
        cond_x_dist = TransformedDistribution(x_bd, cond_x_transforms)

        x = pyro.sample('x', cond_x_dist)

        return x, thickness, slant

    @pyro_method
    def scm(self):
        t_bd = Normal(self.e_t_loc, self.e_t_scale)
        e_t = pyro.sample('e_t', t_bd)

        thickness = self.t_flow_transforms(e_t)
        thickness = pyro.deterministic('thickness', thickness)

        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        e_s = pyro.sample('e_s', s_bd)

        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness).transforms)

        slant = cond_s_transforms(e_s)
        slant = pyro.deterministic('slant', slant)

        context = torch.cat([thickness / 3., slant / 30.], 1)

        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)
        e_x = pyro.sample('e_x', x_bd)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms).inv

        x = cond_x_transforms(e_x)
        x = pyro.deterministic('x', x)

        return x, thickness, slant

    @pyro_method
    def infer_e_t(self, t):
        return self.t_flow_transforms.inv(t)

    @pyro_method
    def infer_e_s(self, t, s):
        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(t).transforms)
        return cond_s_transforms.inv(s)

    @pyro_method
    def infer_e_x(self, t, s, x):
        x_bd = Normal(self.e_x_loc, self.e_x_scale)
        context = torch.cat([t / 3., s / 30.], 1)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms)
        return cond_x_transforms(x)

    def infer(self, t, s, x):
        e_t = self.infer_e_t(t)
        e_s = self.infer_e_s(t, s)
        e_x = self.infer_e_x(t, s, x)
        return e_t, e_s, e_x

    @pyro_method
    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            x, thickness, slant = self.model()

        return x, thickness, slant

    @pyro_method
    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            x, thickness, slant = self.scm()

        return x, thickness, slant


class CNFExperiment(BaseCovariateExperiment):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size

        torch.autograd.set_detect_anomaly(self.hparams.validate)
        if hparams.validate:
            import random
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

        self.pyro_model = FlowModel(
            num_scales=hparams.num_scales, flows_per_scale=hparams.flows_per_scale,
            preprocessing=hparams.preprocessing, hidden_channels=hparams.hidden_channels,
            use_actnorm=hparams.use_actnorm)

    def configure_optimizers(self):
        thickness_params = self.pyro_model.t_flow_components.parameters()
        slant_params = self.pyro_model.s_flow_components.parameters()

        x_params = self.pyro_model.trans_modules.parameters()

        return torch.optim.Adam([
                {'params': x_params, 'lr': self.hparams.lr},
                {'params': thickness_params, 'lr': 5e-2},
                {'params': slant_params, 'lr': 5e-2},
            ], lr=self.hparams.lr)

    def _get_parameters(self, *args, **kwargs):
        return super(PyroExperiment, self)._get_parameters(*args, **kwargs)

    def backward(self, *args, **kwargs):
        return super(PyroExperiment, self).backward(*args, **kwargs)

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

        lls = {('train/' + k + '_ll'): v for k, v in log_probs.items()}
        nats_per_dim = {('train/' + k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        tensorboard_logs = {'train/loss': loss, **nats_per_dim, **lls}

        return {'loss': loss, 'log': tensorboard_logs, **lls}

    def validation_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(x, thickness, slant)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        lls = {(k + '_ll'): v for k, v in log_probs.items()}
        nats_per_dim = {(k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        return {'loss': loss, **lls, **nats_per_dim}

    def test_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(x, thickness, slant)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        lls = {(k + '_ll'): v for k, v in log_probs.items()}
        nats_per_dim = {(k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        return {'loss': loss, **lls, **nats_per_dim}

    def sample_images(self):
        if self.current_epoch < 10:
            return

        with torch.no_grad():
            samples, thickness, slant = self.pyro_model.sample(128)
            samples = samples.reshape(-1, 1, 32, 32)
            grid = torchvision.utils.make_grid(samples.data[:8], normalize=True)
            self.logger.experiment.add_image('samples', grid, self.current_epoch)

            measured_thickness, measured_slant = self.measure_image(samples)
            self.logger.experiment.add_scalar('samples/thickness_mae', torch.mean(torch.abs(thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('samples/slant_mae', torch.mean(torch.abs(slant.cpu() - measured_slant)), self.current_epoch)

            thicknesses = 1. + torch.arange(3, device=samples.device, dtype=torch.float)
            thicknesses = thicknesses.repeat(3).unsqueeze(1)
            slants = 10 * (torch.arange(3, device=samples.device, dtype=torch.float) - 1)
            slants = slants.repeat_interleave(3).unsqueeze(1)

            samples, *_ = pyro.condition(self.pyro_model.sample, data={'thickness': thicknesses, 'slant': slants})(9)
            samples = samples.reshape(-1, 1, 32, 32)
            grid = torchvision.utils.make_grid(samples.data, normalize=True, nrow=3)
            self.logger.experiment.add_image('cond_samples', grid, self.current_epoch)

            x, thickness, slant = self.prep_batch(next(iter(self.val_loader)))
            x = x[:8]
            thickness = thickness[:8]
            slant = slant[:8]
            x_ = x.reshape(-1, 1, 32, 32)

            grid = torchvision.utils.make_grid(x_, normalize=True)
            self.logger.experiment.add_image('input', grid, self.current_epoch)
            x = x.to(samples.device)
            thickness = thickness.to(samples.device)
            slant = slant.to(samples.device)

            e_t, e_s, e_x = self.pyro_model.infer(thickness, slant, x)
            data = {'e_t': e_t, 'e_s': e_s, 'e_x': e_x}

            counter, *_ = pyro.poutine.do(pyro.condition(self.pyro_model.sample_scm, data=data), data={'thickness': thickness + 2})(8)
            counter = counter.reshape(-1, 1, 32, 32).cpu().data
            grid = torchvision.utils.make_grid(torch.cat([x_.cpu(), counter], 0), normalize=True)
            self.logger.experiment.add_image('counter_thickness', grid, self.current_epoch)

            measured_thickness, measured_slant = self.measure_image(counter)
            self.logger.experiment.add_scalar('counter_thickness/thickness_mae', torch.mean(torch.abs(thickness.cpu() - measured_thickness + 2)), self.current_epoch)
            self.logger.experiment.add_scalar('counter_thickness/slant_mae', torch.mean(torch.abs(slant.cpu() - measured_slant)), self.current_epoch)

            counter, *_ = pyro.poutine.do(pyro.condition(self.pyro_model.sample, data=data), data={'slant': slant + 20})(8)
            counter = counter.reshape(-1, 1, 32, 32).cpu().data
            grid = torchvision.utils.make_grid(torch.cat([x_.cpu(), counter], 0), normalize=True)
            self.logger.experiment.add_image('counter_slant', grid, self.current_epoch)

            measured_thickness, measured_slant = self.measure_image(counter)
            self.logger.experiment.add_scalar('counter_slant/thickness_mae', torch.mean(torch.abs(thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('counter_slant/slant_mae', torch.mean(torch.abs(slant.cpu() - measured_slant + 20)), self.current_epoch)


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(logger=True, checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    experiment_group.add_argument('--num_scales', default=3, type=int, help="number of scales (defaults to 4)")
    experiment_group.add_argument('--flows_per_scale', default=5, type=int, help="number of flows per scale (defaults to 2)")
    experiment_group.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing", choices=['realnvp', 'glow'])
    experiment_group.add_argument('--hidden_channels', default=256, type=int, help="number of hidden channels in convnet (defaults to 256)")
    experiment_group.add_argument('--lr', default=1e-4, type=float, help="latent dimension of model (defaults to 1e-4)")
    experiment_group.add_argument('--use_actnorm', default=False, action='store_true', help="whether to use activation norm (defaults to False)")
    experiment_group.add_argument('--validate', default=False, action='store_true', help="latent dimension of model (defaults to False)")
    experiment_group.add_argument('--train_batch_size', default=256, type=int, help="train batch size (defaults to 256)")
    experiment_group.add_argument('--test_batch_size', default=256, type=int, help="test batch size (defaults to 256)")

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

    experiment = CNFExperiment(hparams)

    trainer.fit(experiment)
