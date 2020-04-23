import pyro

from datasets.morphomnist import MorphoMNISTLike

from torch.distributions import Independent
import torchvision.utils
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader, random_split
from experiments import PyroExperiment
import torch
import numpy as np

from morphomnist import measure


class BaseCovariateExperiment(PyroExperiment):
    def __init__(self, hparams):
        super().__init__()

        hparams.experiment = self.__class__.__name__
        self.hparams = hparams
        self.data_dir = "/vol/biomedic2/np716/data/gemini/synthetic/2/"
        self.train_batch_size = 256
        self.test_batch_size = 32

        if hparams.validate:
            pyro.enable_validation()

    def _build_svi(self, loss=TraceGraph_ELBO()):
        self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, Adam({'lr': self.hparams.lr}), loss)

    def measure_image(self, x, normalize=True):
        imgs = x.detach().cpu().numpy()[:, 0]
        imgs -= imgs.min()
        imgs /= imgs.max() + 1e-6
        measurements = measure.measure_batch(imgs)

        return measurements['thickness'].values, np.rad2deg(measurements['slant'].values)

    def prepare_data(self):
        # prepare transforms standard to MNIST
        mnist_train = MorphoMNISTLike(self.data_dir, train=True, columns=['thickness', 'slant'])
        self.mnist_test = MorphoMNISTLike(self.data_dir, train=False, columns=['thickness', 'slant'])

        num_val = int(len(mnist_train) * 0.1)
        num_train = len(mnist_train) - num_val
        self.mnist_train, self.mnist_val = random_split(mnist_train, [num_train, num_val])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        self.val_loader = DataLoader(self.mnist_test, batch_size=self.test_batch_size, shuffle=False)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(self.mnist_test, batch_size=self.test_batch_size, shuffle=False)
        return self.test_loader

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

    def prep_batch(self, batch):
        x = batch['image']
        thickness = batch['thickness'].unsqueeze(1)
        slant = batch['slant'].unsqueeze(1)

        x = x.float()
        # dequantise
        x = (x + torch.rand_like(x)) / 256.
        # constrain to [0.05, 0.95]
        x = ((((x * 2) - 1) * 0.9) + 1) / 2.

        x = x.unsqueeze(1)

        return x, thickness, slant

    def training_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        if self.hparams.validate:
            self.print_trace_updates(batch)

        loss = self.svi.step(x, thickness, slant)

        tensorboard_logs = {'train/loss': loss}

        return {'loss': torch.Tensor([loss]), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(x, thickness, slant)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        num_items = len(outputs)
        metrics = {('val/' + k): v / num_items for k, v in outputs[0].items()}
        for r in outputs[1:]:
            for k, v in r.items():
                metrics[('val/' + k)] += v / num_items

        self.sample_images()

        return {'val_loss': metrics['val/loss'], 'log': metrics}

    def test_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(x, thickness, slant)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        num_items = len(outputs)
        metrics = {('test/' + k): v / num_items for k, v in outputs[0].items()}
        for r in outputs[1:]:
            for k, v in r.items():
                metrics[('test/' + k)] += v / num_items

        return {'test_loss': metrics['test/loss'], 'log': metrics}

    def sample_images(self):
        with torch.no_grad():
            samples, _, thickness, slant = self.pyro_model.sample(8)
            grid = torchvision.utils.make_grid(samples.data, normalize=True)
            self.logger.experiment.add_image('samples', grid, self.current_epoch)

            measured_thickness, measured_slant = self.measure_image(samples)
            self.logger.experiment.add_scalar('samples/thickness_mae', torch.mean(torch.abs(thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('samples/slant_mae', torch.mean(torch.abs(slant.cpu() - measured_slant)), self.current_epoch)

            thicknesses = 1. + torch.arange(3, device=samples.device, dtype=torch.float)
            thicknesses = thicknesses.repeat(3).unsqueeze(1)
            slants = 10 * (torch.arange(3, device=samples.device, dtype=torch.float) - 1)
            slants = slants.repeat_interleave(3).unsqueeze(1)

            with pyro.plate('observations', 9):
                samples, *_ = pyro.condition(self.pyro_model.sample, data={'thickness': thicknesses, 'slant': slants})(None)
            grid = torchvision.utils.make_grid(samples.data, normalize=True, nrow=3)
            self.logger.experiment.add_image('cond_samples', grid, self.current_epoch)

            x, thickness, slant = self.prep_batch(next(iter(self.val_loader)))
            x = x[:8]
            thickness = thickness[:8]
            slant = slant[:8]

            grid = torchvision.utils.make_grid(x, normalize=True)
            self.logger.experiment.add_image('input', grid, self.current_epoch)
            grid = torchvision.utils.make_grid((x > 0.1).float(), normalize=True)
            self.logger.experiment.add_image('input_binary', grid, self.current_epoch)
            x = x.to(samples.device)
            thickness = thickness.to(samples.device)
            slant = slant.to(samples.device)

            recons = self.pyro_model.reconstruct(x)
            grid = torchvision.utils.make_grid(torch.cat([x, recons.data], 0), normalize=True)
            self.logger.experiment.add_image('reconstruction', grid, self.current_epoch)

            grid = torchvision.utils.make_grid(torch.cat([(x > 0.2).float(), (recons.data > 0.2).float()], 0), normalize=True)
            self.logger.experiment.add_image('reconstruction_binary', grid, self.current_epoch)

            z, *_ = self.pyro_model.encode(x)
            cond_recons, *_ = pyro.condition(self.pyro_model.sample, data={'z': z, 'thickness': thickness, 'slant': slant})(8)
            grid = torchvision.utils.make_grid(torch.cat([x, cond_recons.data], 0), normalize=True)
            self.logger.experiment.add_image('cond_reconstruction', grid, self.current_epoch)

            counter, *_ = pyro.condition(self.pyro_model.sample, data={'z': z, 'thickness': thickness + 3, 'slant': slant})(8)
            grid = torchvision.utils.make_grid(torch.cat([x, counter], 0), normalize=True)
            self.logger.experiment.add_image('counter_thickness', grid, self.current_epoch)

            counter, *_ = pyro.condition(self.pyro_model.sample, data={'z': z, 'thickness': thickness, 'slant': slant + 10})(8)
            grid = torchvision.utils.make_grid(torch.cat([x, counter], 0), normalize=True)
            self.logger.experiment.add_image('counter_slant', grid, self.current_epoch)
