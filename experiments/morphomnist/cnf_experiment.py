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
    Permute, SigmoidTransform, AffineTransform, GeneralizedChannelPermute
)
import torchvision
from experiments import PyroExperiment
from distributions.transforms.reshape import ReshapeTransform, SqueezeTransform, TransposeTransform
from arch.mnist import BasicFlowConvNet


class FlowModel(PyroModule):
    def __init__(self, num_scales: int = 4, flows_per_scale: int = 2, preprocessing: str = 'realnvp', hidden_mult: int = 2):
        super().__init__()
        self.num_scales = num_scales
        self.flows_per_scale = flows_per_scale
        self.preprocessing = preprocessing
        self.hidden_mult = hidden_mult

        # TODO: This could be handled by passing a product distribution?

        # priors
        self.register_buffer('e_t_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_t_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_s_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_s_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_x_loc', torch.zeros([1, 32, 32], requires_grad=False))
        self.register_buffer('e_x_scale', torch.ones([1, 32, 32], requires_grad=False))

        self.dims = {'x': 32 * 32, 'thickness': 1, 'slant': 1}

        # decoder parts
        # TODO:
        # Flow for modelling t Gamma
        self.t_flow_components = ComposeTransformModule([Spline(1)])
        self.t_flow_transforms = ComposeTransform([self.t_flow_components, ExpTransform()])

        # affine flow for s normal
        self.s_affine_w = torch.nn.Parameter(torch.randn([1, ]))
        self.s_affine_b = torch.nn.Parameter(torch.randn([1, ]))
        # build flow as s_affine_w * t * e_s + b -> depends on t though

        # realnvp or so for x
        self._build_image_flow()

    def _build_image_flow(self):
        alpha = 0.05
        num_bits = 8

        self.trans_modules = ComposeTransformModule([])

        self.x_transforms = [
            ReshapeTransform((1, 32, 32), (4 ** self.num_scales, 32 // 2 ** self.num_scales, 32 // 2 ** self.num_scales))
        ]

        c = 4 ** self.num_scales
        for scale in range(self.num_scales):
            for _ in range(self.flows_per_scale):
                gcp = GeneralizedChannelPermute(channels=c)
                self.trans_modules.append(gcp)
                self.x_transforms.append(gcp)

                self.x_transforms.append(TransposeTransform(torch.tensor((1, 2, 0))))

                ac = ConditionalAffineCoupling(c // 2, BasicFlowConvNet(c // 2, c * self.hidden_mult, (c // 2, c // 2), 2))
                self.trans_modules.append(ac)
                self.x_transforms.append(ac)

                self.x_transforms.append(TransposeTransform(torch.tensor((2, 0, 1))))

            gcp = GeneralizedChannelPermute(channels=c)
            self.trans_modules.append(gcp)
            self.x_transforms.append(gcp)
            self.x_transforms.append(SqueezeTransform().inv)

            c = c // 4

        gcp = GeneralizedChannelPermute(channels=1)
        self.trans_modules.append(gcp)
        self.x_transforms.append(gcp)

        if self.preprocessing == 'glow':
            # Map to [-0.5,0.5]
            a1 = AffineTransform(-0.5, (1. / 2 ** num_bits))
            preprocess_transform = [a1.inv]
        elif self.preprocessing == 'realnvp':
            s = SigmoidTransform()

            # Map into unconstrained space as done in RealNVP
            a1 = AffineTransform(alpha, (1 - alpha))

            # Map to [0,1]
            a2 = AffineTransform(0., (1. / 2 ** num_bits))

            preprocess_transform = [s, a1.inv, a2.inv]

        self.x_transforms += preprocess_transform

    @pyro_method
    def model(self):
        # TODO: disentangle PGM from images
        t_bd = Normal(self.e_t_loc, self.e_t_scale)
        t_dist = TransformedDistribution(t_bd, self.t_flow_transforms)

        thickness = pyro.sample('thickness', t_dist.to_event(1))

        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        s_dist = TransformedDistribution(s_bd, AffineTransform(self.s_affine_w * thickness, self.s_affine_b))

        slant = pyro.sample('slant', s_dist.to_event(1))

        context = torch.cat([thickness, slant], 1)

        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)
        x_dist = ConditionalTransformedDistribution(x_bd, self.x_transforms)
        cond_x_dist = x_dist.condition(context)

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

        slant = AffineTransform(self.s_affine_w * thickness, self.s_affine_b)(e_s)
        slant = pyro.deterministic('slant', slant)

        context = torch.cat([thickness, slant], 1)

        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)
        e_x = pyro.sample('e_x', x_bd)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms)

        x = cond_x_transforms(e_x)
        x = pyro.deterministic('x', x)

        return x, thickness, slant

    @pyro_method
    def infer_e_t(self, t):
        return self.t_flow_transforms.inv(t)

    @pyro_method
    def infer_e_s(self, t, s):
        return AffineTransform(self.s_affine_w * t, self.s_affine_b).inv(s)

    @pyro_method
    def infer_e_x(self, t, s, x):
        x_bd = Normal(self.e_x_loc, self.e_x_scale)
        context = torch.cat([t, s], 1)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms)
        return cond_x_transforms.inv(x)

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

        self.pyro_model = FlowModel(
            num_scales=hparams.num_scales, flows_per_scale=hparams.flows_per_scale,
            preprocessing=hparams.preprocessing, hidden_mult=hparams.hidden_mult)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                log_probs[name] = site["log_prob"].mean()

        return log_probs

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

        if self.hparams.validate:
            self.print_trace_updates(batch)

        log_probs = self.get_logprobs(x, thickness, slant)
        loss = -torch.stack(tuple(log_probs.values())).sum()

        lls = {('train/' + k + '_ll'): v for k, v in log_probs.items()}
        bits_per_dim = {('train/' + k + '_bits_per_dim'): -v / self.pyro_model.dims[k] for k, v in log_probs.items()}

        tensorboard_logs = {'train/loss': loss, **bits_per_dim, **lls}

        return {'loss': loss, 'log': tensorboard_logs, **lls}

    def validation_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        log_probs = self.get_logprobs(x, thickness, slant)
        loss = -torch.stack(tuple(log_probs.values())).sum()

        lls = {(k + '_ll'): v for k, v in log_probs.items()}
        bits_per_dim = {(k + '_bits_per_dim'): -v / self.pyro_model.dims[k] for k, v in log_probs.items()}

        return {'loss': loss, **lls, **bits_per_dim}

    def test_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        log_probs = self.get_logprobs(x, thickness, slant)
        loss = -torch.stack(tuple(log_probs.values())).sum()

        lls = {(k + '_ll'): v for k, v in log_probs.items()}
        bits_per_dim = {(k + '_bits_per_dim'): -v / self.pyro_model.dims[k] for k, v in log_probs.items()}

        return {'loss': loss, **lls, **bits_per_dim}

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
    experiment_group.add_argument('--num_scales', default=4, type=int, help="number of scales (defaults to 4)")
    experiment_group.add_argument('--flows_per_scale', default=2, type=int, help="number of flows per scale (defaults to 2)")
    experiment_group.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing", choices=['realnvp', 'glow'])
    experiment_group.add_argument('--hidden_mult', default=2, type=int, help="multiplier for num hidden channels in convnet (defaults to 2)")
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

    experiment = CNFExperiment(hparams)

    trainer.fit(experiment)
