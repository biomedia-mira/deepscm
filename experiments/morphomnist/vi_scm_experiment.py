import torch
import pyro

from arch.mnist import Decoder, Encoder
from distributions.deep import DeepBernoulli, DeepIndepNormal, DeepIndepGamma, Conv2dIndepBeta, Conv2dIndepNormal

from pyro.nn import PyroModule, pyro_method
from pyro.distributions import Normal, TransformedDistribution
from torch.distributions import constraints
from pyro.distributions.transforms import (
    ComposeTransform, SigmoidTransform, AffineTransform, ExpTransform, Spline
)
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from experiments.morphomnist.base_experiment import BaseCovariateExperiment
from distributions.transforms.affine import ConditionalAffineTransform
from pyro.nn import DenseNN

from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam

import torchvision
import seaborn as sns
import matplotlib.pyplot as plt


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


class SCMVAE(PyroModule):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # TODO: This could be handled by passing a product distribution?

        # priors
        self.register_buffer('e_t_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_t_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_s_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_s_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_z_loc', torch.zeros([latent_dim, ], requires_grad=False))
        self.register_buffer('e_z_scale', torch.ones([latent_dim, ], requires_grad=False))

        self.register_buffer('e_x_loc', torch.zeros([1, 28, 28], requires_grad=False))
        self.register_buffer('e_x_scale', torch.ones([1, 28, 28], requires_grad=False))

        # decoder parts
        self.decoder = Conv2dIndepNormal(Decoder(latent_dim + 2), 1)
        # Flow for modelling t Gamma
        self.t_flow_components = ComposeTransformModule([Spline(1)])
        self.t_flow_transforms = ComposeTransform([self.t_flow_components, ExpTransform()])

        # affine flow for s normal
        slant_net = DenseNN(1, [1], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.s_flow_components = ConditionalAffineTransform(context_nn=slant_net, event_dim=0)
        self.s_flow_transforms = [self.s_flow_components]

        # encoder parts
        self.encoder = Encoder(hidden_dim)

        # TODO: do we need to replicate the PGM here to be able to run conterfactuals? oO
        latent_layers = torch.nn.Sequential(torch.nn.Linear(hidden_dim + 2, hidden_dim), torch.nn.ReLU())
        self.latent_encoder = DeepIndepNormal(latent_layers, hidden_dim, latent_dim)

    @pyro_method
    def pgm_model(self):
        t_bd = Normal(self.e_t_loc, self.e_t_scale)
        t_dist = TransformedDistribution(t_bd, self.t_flow_transforms)

        thickness = pyro.sample('thickness', t_dist.to_event(1))
        # pseudo call to t_flow_transforms to register with pyro
        _ = self.t_flow_components

        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        s_dist = ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness)

        slant = pyro.sample('slant', s_dist.to_event(1))
        # pseudo call to s_flow_transforms to register with pyro
        _ = self.s_flow_components

        return thickness, slant

    @pyro_method
    def model(self):
        thickness, slant = self.pgm_model()

        z = pyro.sample('z', Normal(self.e_z_loc, self.e_z_scale).to_event(1))

        latent = torch.cat([z, thickness, slant], 1)

        x_normal = self.decoder.predict(latent)
        x_loc = x_normal.base_dist.loc
        x_scale = x_normal.base_dist.scale
        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)

        x_dist = TransformedDistribution(x_bd, ComposeTransform([AffineTransform(x_loc, x_scale, 3), SigmoidTransform()]))

        x = pyro.sample('x', x_dist)

        return x, z, thickness, slant

    @pyro_method
    def pgm_scm(self):
        t_bd = Normal(self.e_t_loc, self.e_t_scale).to_event(1)
        e_t = pyro.sample('e_t', t_bd)

        thickness = self.t_flow_transforms(e_t)
        thickness = pyro.deterministic('thickness', thickness)

        s_bd = Normal(self.e_s_loc, self.e_s_scale).to_event(1)
        e_s = pyro.sample('e_s', s_bd)

        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness).transforms)

        slant = cond_s_transforms(e_s)
        slant = pyro.deterministic('slant', slant)

        return thickness, slant

    @pyro_method
    def scm(self):
        thickness, slant = self.pgm_model()

        z = pyro.sample('z', Normal(self.e_z_loc, self.e_z_scale).to_event(1))

        latent = torch.cat([z, thickness, slant], 1)

        x_normal = self.decoder.predict(latent)
        x_loc = x_normal.base_dist.loc
        x_scale = x_normal.base_dist.scale

        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)
        e_x = pyro.sample('e_x', x_bd)

        x = pyro.deterministic('x', ComposeTransform([AffineTransform(x_loc, x_scale, 3), SigmoidTransform()])(e_x))

        return x, z, thickness, slant

    @pyro_method
    def guide(self, x, thickness, slant):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            hidden = torch.cat([hidden, thickness, slant], 1)
            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z

    @pyro_method
    def svi_guide(self, x, thickness, slant):
        self.guide(x, thickness, slant)

    @pyro_method
    def svi_model(self, x, thickness, slant):
        with pyro.plate('observations', x.shape[0]):
            pyro.condition(self.model, data={'x': x, 'thickness': thickness, 'slant': slant})()

    @pyro_method
    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            x, z, thickness, slant = self.model()

        return x, z, thickness, slant

    @pyro_method
    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            x, z, thickness, slant = self.scm()

        return x, z, thickness, slant

    @pyro_method
    def infer_e_t(self, t):
        return self.t_flow_transforms.inv(t)

    @pyro_method
    def infer_e_s(self, t, s):
        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(t).transforms)
        return cond_s_transforms.inv(s)

    @pyro_method
    def infer_z(self, x, thickness, slant):
        return self.guide(x, thickness, slant)

    @pyro_method
    def infer_e_x(self, x, z, thickness, slant):
        latent = torch.cat([z, thickness, slant], 1)

        x_normal = self.decoder.predict(latent)
        x_loc = x_normal.base_dist.loc
        x_scale = x_normal.base_dist.scale

        e_x = ComposeTransform([AffineTransform(x_loc, x_scale, 3), SigmoidTransform()]).inv(x)
        return e_x

    @pyro_method
    def infer(self, x, thickness, slant):
        e_t = self.infer_e_t(thickness)

        e_s = self.infer_e_s(thickness, slant)

        z = self.infer_z(x, thickness, slant)

        e_x = self.infer_e_x(x, z, thickness, slant)

        return e_t, e_s, z, e_x


class CovariateSCMExperiment(BaseCovariateExperiment):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.pyro_model = SCMVAE(hidden_dim=hparams.hidden_dim, latent_dim=hparams.latent_dim)
        loss = CustomELBO()
        self._build_svi(loss=loss)
        self.svi.loss_class = loss

    def _build_svi(self, loss=TraceGraph_ELBO()):
        def per_param_callable(module_name, param_name):
            if module_name == 's_flow_components' or module_name == 't_flow_components':
                return {"lr": self.hparams.pgm_lr}
            else:
                return {"lr": self.hparams.lr}

        self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, Adam(per_param_callable), loss)

    def get_trace_metrics(self, batch):
        metrics = {}
        x, thickness, slant = self.prep_batch(batch)

        model = self.svi.loss_class.trace_storage['model']
        guide = self.svi.loss_class.trace_storage['guide']

        metrics['log p(x)'] = model.nodes['x']['log_prob_sum']
        metrics['log p(slant)'] = model.nodes['slant']['log_prob_sum']
        metrics['log p(thickness)'] = model.nodes['thickness']['log_prob_sum']
        metrics['log p(z) - log q(z)'] = model.nodes['z']['log_prob_sum'] - guide.nodes['z']['log_prob_sum']
        metrics['p(z)'] = model.nodes['z']['log_prob_sum']
        metrics['q(z)'] = guide.nodes['z']['log_prob_sum']

        return metrics

    def training_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        if self.hparams.validate:
            print('Validation:')
            self.print_trace_updates(batch)

        loss = self.svi.step(x, thickness, slant)

        metrics = self.get_trace_metrics(batch)

        tensorboard_logs = {('train/' + k): v for k, v in metrics.items()}
        tensorboard_logs['train/loss'] = loss

        return {'loss': torch.Tensor([loss]), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(x, thickness, slant)

        metrics = self.get_trace_metrics(batch)

        return {'loss': loss, **metrics}

    def test_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(x, thickness, slant)

        metrics = self.get_trace_metrics(batch)

        return {'loss': loss, **metrics}

    def sample_images(self):
        with torch.no_grad():
            samples, z, sampled_thickness, sampled_slant = self.pyro_model.sample(128)
            samples = samples.reshape(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(samples.data[:8], normalize=True)
            self.logger.experiment.add_image('samples', grid, self.current_epoch)

            measured_thickness, measured_slant = self.measure_image(samples)
            self.logger.experiment.add_scalar('samples/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('samples/slant_mae', torch.mean(torch.abs(sampled_slant.cpu() - measured_slant)), self.current_epoch)

            thicknesses = 1. + torch.arange(3, device=samples.device, dtype=torch.float)
            thicknesses = thicknesses.repeat(3).unsqueeze(1)
            slants = 10 * (torch.arange(3, device=samples.device, dtype=torch.float) - 1)
            slants = slants.repeat_interleave(3).unsqueeze(1)

            samples, *_ = pyro.condition(self.pyro_model.sample, data={'thickness': thicknesses, 'slant': slants})(9)
            samples = samples.reshape(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(samples.data, normalize=True, nrow=3)
            self.logger.experiment.add_image('cond_samples', grid, self.current_epoch)

            x, thickness, slant = self.prep_batch(next(iter(self.val_loader)))

            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            sns.kdeplot(thickness.cpu().numpy().squeeze(), slant.cpu().numpy().squeeze(), ax=ax[0], shade=True, shade_lowest=False)
            sns.kdeplot(sampled_thickness.cpu().numpy().squeeze(), sampled_slant.cpu().numpy().squeeze(), ax=ax[1], shade=True, shade_lowest=False)

            ax[0].set_title('batch')
            ax[1].set_title('sampled')

            ax[0].set_xlabel('thickness')
            ax[1].set_xlabel('thickness')

            ax[0].set_xlabel('slant')
            ax[1].set_xlabel('slant')
            self.logger.experiment.add_figure('sample_kde', fig, self.current_epoch)

            x = x.to(samples.device)
            thickness = thickness.to(samples.device)
            slant = slant.to(samples.device)

            e_t, e_s, z, e_x = self.pyro_model.infer(x, thickness, slant)

            self.logger.experiment.add_histogram('e_t', e_t, self.current_epoch)
            self.logger.experiment.add_histogram('e_s', e_s, self.current_epoch)
            self.logger.experiment.add_histogram('e_x', e_x, self.current_epoch)
            self.logger.experiment.add_histogram('z', z, self.current_epoch)

            x = x[:8]
            thickness = thickness[:8]
            slant = slant[:8]

            e_t = e_t[:8]
            e_s = e_s[:8]
            e_x = e_x[:8]
            z = z[:8]

            x_ = x.reshape(-1, 1, 28, 28)

            grid = torchvision.utils.make_grid(x_, normalize=True)
            self.logger.experiment.add_image('input', grid, self.current_epoch)

            counter, _, sampled_thickness, sampled_slant = pyro.condition(self.pyro_model.sample, data={'z': z, 'slant': slant, 'thickness': thickness})(8)
            counter = counter.reshape(-1, 1, 28, 28).cpu().data
            grid = torchvision.utils.make_grid(torch.cat([x_.cpu(), counter], 0), normalize=True)
            self.logger.experiment.add_image('reconstruction', grid, self.current_epoch)

            measured_thickness, measured_slant = self.measure_image(counter)
            self.logger.experiment.add_scalar('reconstruction/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('reconstruction/slant_mae', torch.mean(torch.abs(sampled_slant.cpu() - measured_slant)), self.current_epoch)

            data = {'e_t': e_t, 'e_s': e_s, 'e_x': e_x, 'z': z}

            counter, _, sampled_thickness, sampled_slant = pyro.poutine.do(pyro.condition(self.pyro_model.sample_scm, data=data), data={'thickness': thickness + 2})(8)
            counter = counter.reshape(-1, 1, 28, 28).cpu().data
            grid = torchvision.utils.make_grid(torch.cat([x_.cpu(), counter], 0), normalize=True)
            self.logger.experiment.add_image('counter_thickness', grid, self.current_epoch)

            measured_thickness, measured_slant = self.measure_image(counter)
            self.logger.experiment.add_scalar('counter_thickness/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('counter_thickness/slant_mae', torch.mean(torch.abs(sampled_slant.cpu() - measured_slant)), self.current_epoch)

            counter, _, sampled_thickness, sampled_slant = pyro.poutine.do(pyro.condition(self.pyro_model.sample, data=data), data={'slant': slant + 20})(8)
            counter = counter.reshape(-1, 1, 28, 28).cpu().data
            grid = torchvision.utils.make_grid(torch.cat([x_.cpu(), counter], 0), normalize=True)
            self.logger.experiment.add_image('counter_slant', grid, self.current_epoch)

            measured_thickness, measured_slant = self.measure_image(counter)
            self.logger.experiment.add_scalar('counter_slant/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('counter_slant/slant_mae', torch.mean(torch.abs(sampled_slant.cpu() - measured_slant)), self.current_epoch)


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
    experiment_group.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (defaults to 1e-4)")
    experiment_group.add_argument('--pgm_lr', default=5e-2, type=float, help="lr of pgm (defaults to 5e-2)")
    experiment_group.add_argument('--validate', default=False, action='store_true', help="whether to validate (defaults to False)")

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
