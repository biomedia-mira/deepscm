import pyro

from pyro.infer import SVI, TraceGraph_ELBO
from pyro.nn import pyro_method
from pyro.optim import Adam
from torch.distributions import Independent

import torch

from experiments.morphomnist.base_experiment import BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401

from pyro.distributions.transforms import ComposeTransform


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
        self.register_buffer('e_t_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_t_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_s_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('e_s_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('e_z_loc', torch.zeros([latent_dim, ], requires_grad=False))
        self.register_buffer('e_z_scale', torch.ones([latent_dim, ], requires_grad=False))

        self.register_buffer('e_x_loc', torch.zeros([1, 28, 28], requires_grad=False))
        self.register_buffer('e_x_scale', torch.ones([1, 28, 28], requires_grad=False))

    def _get_preprocess_transforms(self):
        return super()._get_preprocess_transforms().inv

    @pyro_method
    def guide(self, x, thickness, slant):
        raise NotImplementedError()

    @pyro_method
    def svi_guide(self, x, thickness, slant):
        self.guide(x, thickness, slant)

    @pyro_method
    def svi_model(self, x, thickness, slant):
        with pyro.plate('observations', x.shape[0]):
            pyro.condition(self.model, data={'x': x, 'thickness': thickness, 'slant': slant})()

    @pyro_method
    def infer_z(self, *args, **kwargs):
        return self.guide(*args, **kwargs)

    @pyro_method
    def infer_exogeneous(self, x, z, thickness, slant):
        # assuming that we use transformed distributions for everything:
        cond_sample = pyro.condition(self.sample, data={'x': x, 'z': z, 'thickness': thickness, 'slant': slant})
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(x.shape[0])

        output = {}
        for (node, short) in [('thickness', 'e_t'), ('slant', 'e_s'), ('x', 'e_x')]:
            fn = cond_trace.nodes[node]['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            output[short] = ComposeTransform(fn.transforms).inv(cond_trace.nodes[node]['value'])

        return output

    @pyro_method
    def infer(self, x, thickness, slant):
        z = self.infer_z(x, thickness, slant)

        exogeneous = self.infer_exogeneous(x, z, thickness, slant)
        exogeneous['z'] = z

        return exogeneous

    @pyro_method
    def reconstruct(self, x, thickness, slant, num_particles=1):
        z_dist = pyro.poutine.trace(self.guide).get_trace(x, thickness, slant).nodes['z']['fn']

        recons = []
        for _ in range(num_particles):
            z = pyro.sample('z', z_dist)
            recon, *_ = pyro.poutine.condition(self.sample, data={'thickness': thickness, 'slant': slant, 'z': z})(x.shape[0])
            recons += [recon]
        return torch.stack(recons).mean(0)

    @pyro_method
    def counterfactual(self, x, thickness, slant, data=None, num_particles=1):
        z_dist = pyro.poutine.trace(self.guide).get_trace(x, thickness, slant).nodes['z']['fn']

        counterfactuals = []
        for _ in range(num_particles):
            z = pyro.sample('z', z_dist)

            exogeneous = self.infer_exogeneous(x, z, thickness, slant)
            exogeneous['z'] = z
            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=data)(x.shape[0])
            counterfactuals += [counter]
        return (torch.stack(c).mean(0) for c in zip(*counterfactuals))

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--latent_dim', default=10, type=int, help="latent dimension of model (default: %(default)s)")
        parser.add_argument('--hidden_dim', default=100, type=int, help="hidden dimension of model (default: %(default)s)")
        parser.add_argument('--logstd_init', default=-5, type=float, help="init of logstd (default: %(default)s)")

        return parser


class SVIExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model):
        super().__init__(hparams, pyro_model)

        self.svi_loss = CustomELBO(num_particles=hparams.num_svi_particles)

        self._build_svi()

    def _build_svi(self, loss=None):
        def per_param_callable(module_name, param_name):
            params = {'eps': 1e-5, 'amsgrad': self.hparams.use_amsgrad, 'weight_decay': self.hparams.l2}
            if module_name == 's_flow_components' or module_name == 't_flow_components':
                params['lr'] = self.hparams.pgm_lr
                return params
            else:
                params['lr'] = self.hparams.lr
                return params

        if loss is None:
            loss = self.svi_loss

        self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, Adam(per_param_callable), loss)
        self.svi.loss_class = loss

    def print_trace_updates(self, batch):
        print('Traces:\n' + ('#' * 10))

        x, thickness, slant = self.prep_batch(batch)

        guide_trace = pyro.poutine.trace(self.pyro_model.svi_guide).get_trace(x, thickness, slant)
        model_trace = pyro.poutine.trace(pyro.poutine.replay(self.pyro_model.svi_model, trace=guide_trace)).get_trace(x, thickness, slant)

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

    def prep_batch(self, batch):
        x = batch['image']
        thickness = batch['thickness'].unsqueeze(1).float()
        slant = batch['slant'].unsqueeze(1).float()

        x = x.float()

        x += torch.rand_like(x)

        x = x.unsqueeze(1)

        return x, thickness, slant

    def training_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        if self.hparams.validate:
            print('Validation:')
            self.print_trace_updates(batch)

        loss = self.svi.step(x, thickness, slant)

        if torch.isnan(loss):
            self.logger.experiment.add_text('nan', f'nand at {self.current_epoch}')
            raise ValueError('loss went to nan')

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

    def log_img_grid(self, tag, imgs, normalize=True, save_img=True, **kwargs):
        super().log_img_grid(tag, imgs, normalize, save_img, **kwargs)

    def build_reconstruction(self, x, thickness, slant, tag='reconstruction'):
        recon = self.pyro_model.reconstruct(x, thickness, slant, num_particles=self.hparams.num_sample_particles)
        self.log_img_grid(tag, torch.cat([x, recon], 0))
        self.logger.experiment.add_scalar(f'{tag}/mse', torch.mean(torch.square(x - recon).sum((1, 2, 3))), self.current_epoch)

        measured_thickness, measured_slant = self.measure_image(recon)
        self.logger.experiment.add_scalar(
            f'{tag}/thickness_mae', torch.mean(torch.abs(thickness.cpu() - measured_thickness)), self.current_epoch)
        self.logger.experiment.add_scalar(
            f'{tag}/slant_mae', torch.mean(torch.abs(slant.cpu() - measured_slant)), self.current_epoch)

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
            counter, _, sampled_thickness, sampled_slant = self.pyro_model.counterfactual(
                x, thickness, slant, data=data, num_particles=self.hparams.num_sample_particles)

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
            samples, z, sampled_thickness, sampled_slant = self.pyro_model.sample(128)
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
            self.logger.experiment.add_histogram('z', z, self.current_epoch)

            x = x[:8]
            thickness = thickness[:8]
            slant = slant[:8]

            self.log_img_grid('input', x, save_img=True)

            vals = [x, thickness, slant]

            self.build_reconstruction(*vals)

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

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--num_svi_particles', default=4, type=int, help="number of particles to use for ELBO (default: %(default)s)")
        parser.add_argument('--num_sample_particles', default=32, type=int, help="number of particles to use for MC sampling (default: %(default)s)")

        return parser


EXPERIMENT_REGISTRY[SVIExperiment.__name__] = SVIExperiment
