import pyro

from datasets.morphomnist import MorphoMNISTLike

from torch.distributions import Independent
import torchvision.utils
from pyro.infer import SVI, TraceGraph_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader, random_split
from experiments import PyroExperiment
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from morphomnist import measure
import os

EXPERIMENT_REGISTRY = {}
MODEL_REGISTRY = {}


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
    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.model()

        return (*samples,)

    @pyro_method
    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()

        return (*samples,)

    @pyro_method
    def infer_e_t(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_e_s(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_e_x(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer(self, x, thickness, slant):
        raise NotImplementedError()

    @pyro_method
    def counterfactual(self, x, thickness, slant, data=None):
        raise NotImplementedError()

    @classmethod
    def add_arguments(parser):

        return parser


class BaseCovariateExperiment(PyroExperiment):
    def __init__(self, hparams, pyro_model):
        super().__init__()

        self.pyro_model = pyro_model

        hparams.experiment = self.__class__.__name__
        hparams.model = pyro_model.__class__.__name__
        self.hparams = hparams
        self.data_dir = hparams.data_dir
        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size

        if hparams.validate:
            pyro.enable_validation()

    def _build_svi(self, loss=TraceGraph_ELBO()):
        self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, Adam({'lr': self.hparams.lr}), loss)

    def measure_image(self, x, normalize=True, threshold=0.3):
        imgs = x.detach().cpu().numpy()[:, 0]
        imgs -= imgs.min()
        imgs /= imgs.max() + 1e-6
        measurements = measure.measure_batch(imgs, threshold=threshold, use_progress_bar=False)

        return measurements['thickness'].values, np.rad2deg(measurements['slant'].values)

    def prepare_data(self):
        # prepare transforms standard to MNIST
        mnist_train = MorphoMNISTLike(self.data_dir, train=True, columns=['thickness', 'slant'])
        self.mnist_test = MorphoMNISTLike(self.data_dir, train=False, columns=['thickness', 'slant'])

        num_val = int(len(mnist_train) * 0.1)
        num_train = len(mnist_train) - num_val
        self.mnist_train, self.mnist_val = random_split(mnist_train, [num_train, num_val])

        self.device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device
        thicknesses = 1. + torch.arange(3, dtype=torch.float, device=self.device)
        self.thickness_range = thicknesses.repeat(3).unsqueeze(1)
        slants = 10 * (torch.arange(3, dtype=torch.float, device=self.device) - 1)
        self.slant_range = slants.repeat_interleave(3).unsqueeze(1)
        self.z_range = torch.zeros([9, self.hparams.latent_dim], device=self.device, dtype=torch.float)

        self.pyro_model.s_flow_norm.loc = mnist_train.metrics['slant'].mean().to(self.device).float()
        self.pyro_model.s_flow_norm.scale = mnist_train.metrics['slant'].std().to(self.device).float()

        self.pyro_model.t_flow_lognorm.loc = mnist_train.metrics['thickness'].log().mean().to(self.device).float()
        self.pyro_model.t_flow_lognorm.scale = mnist_train.metrics['thickness'].log().std().to(self.device).float()

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
        thickness = batch['thickness'].unsqueeze(1).float()
        slant = batch['slant'].unsqueeze(1).float()

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

        if self.current_epoch % self.hparams.sample_img_interval == 0:
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

    def log_img_grid(self, tag, imgs, normalize=True, save_img=False, **kwargs):
        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            torchvision.utils.save_image(imgs, p)
        grid = torchvision.utils.make_grid(imgs, normalize=normalize, **kwargs)
        self.logger.experiment.add_image(tag, grid, self.current_epoch)

    def get_batch(self, loader):
        batch = next(iter(self.val_loader))
        if self.trainer.on_gpu:
            batch = self.trainer.transfer_batch_to_gpu(batch, self.device)
        return batch

    def log_kdes(self, tag, data, save_img=False):
        """
        requires data to be:
        {'data1': {'x': x, 'y': y}, 'data2': {'x': x, 'y': y}, ..}
        """
        def np_val(x):
            return x.cpu().numpy().squeeze() if isinstance(x, torch.Tensor) else x.squeeze()

        fig, ax = plt.subplots(1, len(data), figsize=(5 * len(data), 3))
        for i, (name, covariates) in enumerate(data.items()):
            if len(covariates) == 1:
                (x_n, x), = tuple(covariates.items())
                sns.kdeplot(np_val(x), ax=ax[i], shade=True, shade_lowest=False)
            elif len(covariates) == 2:
                (x_n, x), (y_n, y) = tuple(covariates.items())
                sns.kdeplot(np_val(x), np_val(y), ax=ax[i], shade=True, shade_lowest=False)
                ax[i].set_ylabel(y_n)
            else:
                raise ValueError(f'got too many values: {len(covariates)}')

            ax[i].set_title(name)
            ax[i].set_xlabel(x_n)

        sns.despine()

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            plt.savefig(p, dpi=300)

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    def sample_images(self):
        with torch.no_grad():
            samples, _, thickness, slant = self.pyro_model.sample(8)
            self.log_img_grid('samples', samples.data)

            measured_thickness, measured_slant = self.measure_image(samples)
            self.logger.experiment.add_scalar('samples/thickness_mae', torch.mean(torch.abs(thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('samples/slant_mae', torch.mean(torch.abs(slant.cpu() - measured_slant)), self.current_epoch)

            with pyro.plate('observations', 9):
                samples, *_ = pyro.condition(self.pyro_model.sample, data={'thickness': self.thickness_range, 'slant': self.slant_range})(None)
            self.log_img_grid('cond_samples', samples.data, nrow=3)

            x, thickness, slant = self.prep_batch(self.get_batch(self.val_loader))
            x = x[:8]
            thickness = thickness[:8]
            slant = slant[:8]

            self.log_img_grid('input', x.data)
            self.log_img_grid('input_binary', (x > 0.1).float())

            recons = self.pyro_model.reconstruct(x)
            self.log_img_grid('reconstruction', torch.cat([x, recons.data], 0))
            self.log_img_grid('reconstruction_binary', torch.cat([(x > 0.2).float(), (recons.data > 0.2).float()], 0))

            z, *_ = self.pyro_model.encode(x)
            cond_recons, *_ = pyro.condition(self.pyro_model.sample, data={'z': z, 'thickness': thickness, 'slant': slant})(8)
            self.log_img_grid('cond_reconstruction', torch.cat([x, cond_recons.data], 0))

            counter, *_ = pyro.condition(self.pyro_model.sample, data={'z': z, 'thickness': thickness + 3, 'slant': slant})(8)
            self.log_img_grid('counter_thickness', torch.cat([x, counter.data], 0),)

            counter, *_ = pyro.condition(self.pyro_model.sample, data={'z': z, 'thickness': thickness, 'slant': slant + 10})(8)
            self.log_img_grid('counter_slant', torch.cat([x, counter.data], 0),)

    @classmethod
    def add_arguments(parser):
        parser.add_argument('--data_dir', default="/vol/biomedic2/np716/data/gemini/synthetic/2_more_slant/", type=str, help="data dir (default: %(default)s)")
        parser.add_argument('--sample_img_interval', default=10, type=int, help="interval in which to sample and log images (default: %(default)s)")
        parser.add_argument('--train_batch_size', default=256, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test_batch_size', default=256, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--validate', default=False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm_lr', default=5e-2, type=float, help="lr of pgm (default: %(default)s)")

        return parser


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    import argparse

    exp_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_parser.add_argument('--experiment', '-e', help='which experiment to load')
    exp_parser.add_argument('--model', '-m', help='which model to load')

    exp_args, other_args = exp_parser.parse_known_args()

    exp_class = EXPERIMENT_REGISTRY[exp_args.experiment]
    model_class = MODEL_REGISTRY[exp_args.model]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(logger=True, checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    exp_class.add_arguments(experiment_group)

    model_group = parser.add_argument_group('model')
    model_class.add_arguments(model_group)

    args = parser.parse_args(other_args)

    # TODO: push to lightning
    args.gradient_clip_val = float(args.gradient_clip_val)

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)

    lightning_args = groups['lightning_options']
    hparams = groups['experiment']
    model_params = groups['model']

    hparams.model_params = model_params

    trainer = Trainer.from_argparse_args(lightning_args)

    model = model_class(**model_params)
    experiment = exp_class(hparams, model)

    trainer.fit(experiment)
