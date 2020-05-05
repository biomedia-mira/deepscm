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
import seaborn as sns
import matplotlib.pyplot as plt


class FlowModel(PyroModule):
    def __init__(self, num_scales: int = 4, flows_per_scale: int = 2, preprocessing: str = 'realnvp', hidden_channels: int = 256,
                 use_actnorm: bool = False, use_affine_ex: bool = True, use_rad: bool = False):
        super().__init__()
        self.num_scales = num_scales
        self.flows_per_scale = flows_per_scale
        self.preprocessing = preprocessing
        self.hidden_channels = hidden_channels
        self.use_actnorm = use_actnorm
        self.use_affine_ex = use_affine_ex
        self.use_rad = use_rad

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
        self.t_flow_lognorm = AffineTransform(loc=0., scale=1.)
        self.t_flow_constraint_transforms = ComposeTransform([self.t_flow_lognorm, ExpTransform()])
        self.t_flow_transforms = ComposeTransform([self.t_flow_components, self.t_flow_constraint_transforms])

        # affine flow for s normal
        slant_net = DenseNN(1, [1], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.s_flow_components = ConditionalAffineTransform(context_nn=slant_net, event_dim=0)
        self.s_flow_norm = AffineTransform(loc=0., scale=1.)
        self.s_flow_transforms = [self.s_flow_components, self.s_flow_norm]
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
        for _ in range(self.num_scales):
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

        if self.use_affine_ex:
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

        thickness_ = self.t_flow_constraint_transforms.inv(thickness)

        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        s_dist = ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness_)

        slant = pyro.sample('slant', s_dist.to_event(1))

        slant_ = self.s_flow_norm.inv(slant)

        context = torch.cat([thickness_, slant_], 1)

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
        thickness_ = self.t_flow_constraint_transforms.inv(thickness)

        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        e_s = pyro.sample('e_s', s_bd)

        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness_).transforms)

        slant = cond_s_transforms(e_s)
        slant = pyro.deterministic('slant', slant)
        slant_ = self.s_flow_norm.inv(slant)

        context = torch.cat([thickness_, slant_], 1)

        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)
        e_x = pyro.sample('e_x', x_bd)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms).inv

        x = cond_x_transforms(e_x)
        x = pyro.deterministic('x', x)

        return x, thickness, slant

    @pyro_method
    def infer_e_t(self, thickness):
        return self.t_flow_transforms.inv(thickness)

    @pyro_method
    def infer_e_s(self, thickness, slant):
        s_bd = Normal(self.e_s_loc, self.e_s_scale)

        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness_).transforms)
        return cond_s_transforms.inv(slant)

    @pyro_method
    def infer_e_x(self, thickness, slant, x):
        x_bd = Normal(self.e_x_loc, self.e_x_scale)

        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        slant_ = self.s_flow_norm.inv(slant)

        context = torch.cat([thickness_, slant_], 1)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms)
        return cond_x_transforms(x)

    def infer(self, thickness, slant, x):
        e_t = self.infer_e_t(thickness)
        e_s = self.infer_e_s(thickness, slant)
        e_x = self.infer_e_x(thickness, slant, x)
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

    @pyro_method
    def counterfactual(self, x, thickness, slant, data=None):
        e_t, e_s, e_x = self.infer(thickness, slant, x)

        counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data={'e_x': e_x, 'e_t': e_t, 'e_s': e_s}), data=data)(x.shape[0])
        return (*counter,)


class CNFExperiment(BaseCovariateExperiment):
    def __init__(self, hparams):
        hparams.latent_dim = 32 * 32
        pyro_model = FlowModel(
            num_scales=hparams.num_scales, flows_per_scale=hparams.flows_per_scale,
            preprocessing=hparams.preprocessing, hidden_channels=hparams.hidden_channels,
            use_actnorm=hparams.use_actnorm, use_affine_ex=hparams.use_affine_ex)

        super().__init__(hparams, pyro_model)

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

    def configure_optimizers(self):
        thickness_params = self.pyro_model.t_flow_components.parameters()
        slant_params = self.pyro_model.s_flow_components.parameters()

        x_params = self.pyro_model.trans_modules.parameters()

        return torch.optim.Adam([
                {'params': x_params, 'lr': self.hparams.lr},
                {'params': thickness_params, 'lr': self.hparams.pgm_lr},
                {'params': slant_params, 'lr': self.hparams.pgm_lr},
            ], lr=self.hparams.lr, eps=1e-5)

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
        if self.current_epoch < 10:
            return

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

            e_t, e_s, e_x = self.pyro_model.infer(thickness, slant, x)

            self.logger.experiment.add_histogram('e_t', e_t, self.current_epoch)
            self.logger.experiment.add_histogram('e_s', e_s, self.current_epoch)
            self.logger.experiment.add_histogram('e_x', e_x, self.current_epoch)

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


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(logger=True, checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    experiment_group.add_argument('--num_scales', default=4, type=int, help="number of scales (default: %(default)s)")
    experiment_group.add_argument('--flows_per_scale', default=10, type=int, help="number of flows per scale (default: %(default)s)")
    experiment_group.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing (default: %(default)s)", choices=['realnvp', 'glow'])
    experiment_group.add_argument('--hidden_channels', default=256, type=int, help="number of hidden channels in convnet (default: %(default)s)")
    experiment_group.add_argument('--lr', default=1e-4, type=float, help="lr for deep part (default: %(default)s)")
    experiment_group.add_argument('--pgm_lr', default=1e-1, type=float, help="lr for pgm (default: %(default)s)")
    experiment_group.add_argument('--l2', default=0., type=float, help="weight decay (default: %(default)s)")
    experiment_group.add_argument('--use_amsgrad', default=False, action='store_true', help="use amsgrad? (default: %(default)s)")
    experiment_group.add_argument('--use_actnorm', default=False, action='store_true', help="whether to use activation norm (default: %(default)s)")
    experiment_group.add_argument('--use_affine_ex', default=False, action='store_true', help="whether to use conditional affine transformation on e_x (default: %(default)s)")
    experiment_group.add_argument('--validate', default=False, action='store_true', help="latent dimension of model (default: %(default)s)")
    experiment_group.add_argument('--train_batch_size', default=256, type=int, help="train batch size (default: %(default)s)")
    experiment_group.add_argument('--test_batch_size', default=256, type=int, help="test batch size (default: %(default)s)")
    experiment_group.add_argument('--use_rad', default=False, action='store_true', help="whether to use rad instead of deg for decoder (default: %(default)s)")
    experiment_group.add_argument('--data_dir', default="/vol/biomedic2/np716/data/gemini/synthetic/2_more_slant/", type=str, help="data dir (default: %(default)s)")
    experiment_group.add_argument('--sample_img_interval', default=10, type=int, help="interval in which to sample and log images (default: %(default)s)")

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
