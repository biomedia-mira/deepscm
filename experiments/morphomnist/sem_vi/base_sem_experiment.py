import pyro

from pyro.infer import SVI, TraceGraph_ELBO
from pyro.nn import PyroModule, pyro_method
from pyro.optim import Adam
from torch.distributions import Independent

import torch

from experiments.morphomnist.base_experiment import BaseCovariateExperiment

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


class BaseSEM(PyroModule):
    def __init__(self):
        super().__init__()

    @pyro_method
    def pgm_model(self):
        raise NotImplementedError()

    @pyro_method
    def model(self):
        raise NotImplementedError()

    @pyro_method
    def pgm_scm(self):
        raise NotImplementedError()

    @pyro_method
    def scm(self):
        raise NotImplementedError()

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
    def infer_e_t(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_e_s(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_z(self, *args, **kwargs):
        return self.guide(*args, **kwargs)

    @pyro_method
    def infer_e_x(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_exogeneous(self, x, z, thickness, slant):
        # assuming that we use transformed distributions for everything:
        cond_sample = pyro.condition(self.sample, data={'x': x, 'z': z, 'thickness': thickness, 'slant': slant})
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(x.shape[0])

        output = []
        for node in ['x', 'thickness', 'slant']:
            fn = cond_trace.nodes[node]['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            output.append(ComposeTransform(fn.transforms).inv(cond_trace.nodes[node]['value']))

        return tuple(output)

    @pyro_method
    def infer(self, x, thickness, slant):
        z = self.infer_z(x, thickness, slant)

        e_x, e_t, e_s = self.infer_exogeneous(x, z, thickness, slant)

        return e_t, e_s, z, e_x

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

            e_x, e_t, e_s = self.infer_exogeneous(x, z, thickness, slant)
            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data={'e_x': e_x, 'e_t': e_t, 'e_s': e_s, 'z': z}), data=data)(x.shape[0])
            counterfactuals += [counter]
        return (torch.stack(c).mean(0) for c in zip(*counterfactuals))


class BaseSEMExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model):
        super().__init__(hparams, pyro_model)

        self.svi_loss = CustomELBO(num_particles=hparams.num_svi_particles)

        self._build_svi()

    def log_img_grid(self, tag, imgs, normalize=True, save_img=True, **kwargs):
        super().log_img_grid(tag, imgs, normalize, save_img, **kwargs)

    def _build_svi(self, loss=None):
        def per_param_callable(module_name, param_name):
            if module_name == 's_flow_components' or module_name == 't_flow_components':
                return {"lr": self.hparams.pgm_lr}
            else:
                return {"lr": self.hparams.lr}

        if loss is None:
            loss = self.svi_loss

        self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, Adam(per_param_callable), loss)
        self.svi.loss_class = loss

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

            e_t, e_s, z, e_x = self.pyro_model.infer(x, thickness, slant)

            self.logger.experiment.add_histogram('e_t', e_t, self.current_epoch)
            self.logger.experiment.add_histogram('e_s', e_s, self.current_epoch)
            self.logger.experiment.add_histogram('e_x', e_x, self.current_epoch)
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
