import pyro

from pyro.nn import PyroModule, pyro_method

from pyro.distributions import TransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from torch.distributions import Independent

from datasets.morphomnist import MorphoMNISTLike
from pyro.distributions.transforms import ComposeTransform, SigmoidTransform, AffineTransform

import torchvision.utils
from torch.utils.data import DataLoader, random_split
from experiments import PyroExperiment
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from morphomnist import measure
import os
from functools import partial


EXPERIMENT_REGISTRY = {}
MODEL_REGISTRY = {}


class BaseSEM(PyroModule):
    def __init__(self, preprocessing: str = 'realnvp'):
        super().__init__()

        self.preprocessing = preprocessing

    def _get_preprocess_transforms(self):
        alpha = 0.05
        num_bits = 8

        if self.preprocessing == 'glow':
            # Map to [-0.5,0.5]
            a1 = AffineTransform(-0.5, (1. / 2 ** num_bits))
            preprocess_transform = ComposeTransform([a1])
        elif self.preprocessing == 'realnvp':
            # Map to [0,1]
            a1 = AffineTransform(0., (1. / 2 ** num_bits))

            # Map into unconstrained space as done in RealNVP
            a2 = AffineTransform(alpha, (1 - alpha))

            s = SigmoidTransform()

            preprocess_transform = ComposeTransform([a1, a2, s.inv])

        return preprocess_transform

    @pyro_method
    def pgm_model(self):
        raise NotImplementedError()

    @pyro_method
    def model(self):
        raise NotImplementedError()

    @pyro_method
    def pgm_scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.pgm_model, config=config)(*args, **kwargs)

    @pyro_method
    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

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
    def infer_exogeneous(self, **obs):
        # assuming that we use transformed distributions for everything:
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['x'].shape[0])

        output = {}
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node.keys():
                continue

            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            if isinstance(fn, TransformedDistribution):
                output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])

        return output

    @pyro_method
    def infer(self, **obs):
        raise NotImplementedError()

    @pyro_method
    def counterfactual(self, obs, condition=None):
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing (default: %(default)s)", choices=['realnvp', 'glow'])

        return parser


class BaseCovariateExperiment(PyroExperiment):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__()

        self.pyro_model = pyro_model

        hparams.experiment = self.__class__.__name__
        hparams.model = pyro_model.__class__.__name__
        self.hparams = hparams
        self.data_dir = hparams.data_dir
        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size

        if hasattr(hparams, 'num_sample_particles'):
            self.pyro_model._gen_counterfactual = partial(self.pyro_model.counterfactual, num_particles=self.hparams.num_sample_particles)
        else:
            self.pyro_model._gen_counterfactual = self.pyro_model.counterfactual

        if hparams.validate:
            import random

            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.autograd.set_detect_anomaly(self.hparams.validate)
            pyro.enable_validation()

    def measure_image(self, x, normalize=False, threshold=0.3):
        imgs = x.detach().cpu().numpy()[:, 0]

        if normalize:
            imgs -= imgs.min()
            imgs /= imgs.max() + 1e-6

        measurements = measure.measure_batch(imgs, threshold=threshold, use_progress_bar=False)

        def get_intensity(imgs, threshold):

            img_min, img_max = imgs.min(axis=(1, 2), keepdims=True), imgs.max(axis=(1, 2), keepdims=True)
            mask = (imgs >= img_min + (img_max - img_min) * threshold)

            return np.array([i[m].mean() for i, m in zip(imgs, mask)])

        return measurements['thickness'].values, get_intensity(imgs, threshold)

    def prepare_data(self):
        # prepare transforms standard to MNIST
        mnist_train = MorphoMNISTLike(self.data_dir, train=True, columns=['thickness', 'intensity'])
        self.mnist_test = MorphoMNISTLike(self.data_dir, train=False, columns=['thickness', 'intensity'])

        num_val = int(len(mnist_train) * 0.1)
        num_train = len(mnist_train) - num_val
        self.mnist_train, self.mnist_val = random_split(mnist_train, [num_train, num_val])

        self.device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device
        thicknesses = 1. + torch.arange(3, dtype=torch.float, device=self.device)
        self.thickness_range = thicknesses.repeat(3).unsqueeze(1)
        intensity = 48 * torch.arange(3, dtype=torch.float, device=self.device) + 64
        self.intensity_range = intensity.repeat_interleave(3).unsqueeze(1)
        self.z_range = torch.randn([1, self.hparams.latent_dim], device=self.device, dtype=torch.float).repeat((9, 1))

        self.pyro_model.intensity_flow_norm.loc = mnist_train.metrics['intensity'].min().to(self.device).float()
        self.pyro_model.intensity_flow_norm.scale = (mnist_train.metrics['intensity'].max() - mnist_train.metrics['intensity'].min()).to(self.device).float()

        self.pyro_model.thickness_flow_lognorm.loc = mnist_train.metrics['thickness'].log().mean().to(self.device).float()
        self.pyro_model.thickness_flow_lognorm.scale = mnist_train.metrics['thickness'].log().std().to(self.device).float()

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        self.val_loader = DataLoader(self.mnist_test, batch_size=self.test_batch_size, shuffle=False)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(self.mnist_test, batch_size=self.test_batch_size, shuffle=False)
        return self.test_loader

    def prep_batch(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        num_items = len(outputs)
        metrics = {('val/' + k): v / num_items for k, v in outputs[0].items()}
        for r in outputs[1:]:
            for k, v in r.items():
                metrics[('val/' + k)] += v / num_items

        if self.current_epoch % self.hparams.sample_img_interval == 0:
            self.sample_images()

        return {'val_loss': metrics['val/loss'], 'log': metrics}

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

        fig, ax = plt.subplots(1, len(data), figsize=(5 * len(data), 3), sharex=True, sharey=True)
        for i, (name, covariates) in enumerate(data.items()):
            try:
                if len(covariates) == 1:
                    (x_n, x), = tuple(covariates.items())
                    sns.kdeplot(np_val(x), ax=ax[i], shade=True, shade_lowest=False)
                elif len(covariates) == 2:
                    (x_n, x), (y_n, y) = tuple(covariates.items())
                    sns.kdeplot(np_val(x), np_val(y), ax=ax[i], shade=True, shade_lowest=False)
                    ax[i].set_ylabel(y_n)
                else:
                    raise ValueError(f'got too many values: {len(covariates)}')
            except np.linalg.LinAlgError:
                print(f'got a linalg error when plotting {tag}/{name}')

            ax[i].set_title(name)
            ax[i].set_xlabel(x_n)

        sns.despine()

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            plt.savefig(p, dpi=300)

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    def build_reconstruction(self, x, thickness, intensity, tag='reconstruction'):
        obs = {'x': x, 'thickness': thickness, 'intensity': intensity}

        recon = self.pyro_model.reconstruct(**obs, num_particles=self.hparams.num_sample_particles)
        self.log_img_grid(tag, torch.cat([x, recon], 0))
        self.logger.experiment.add_scalar(f'{tag}/mse', torch.mean(torch.square(x - recon).sum((1, 2, 3))), self.current_epoch)

        measured_thickness, measured_intensity = self.measure_image(recon)
        self.logger.experiment.add_scalar(
            f'{tag}/thickness_mae', torch.mean(torch.abs(thickness.cpu() - measured_thickness)), self.current_epoch)
        self.logger.experiment.add_scalar(
            f'{tag}/intensity_mae', torch.mean(torch.abs(intensity.cpu() - measured_intensity)), self.current_epoch)

    def build_counterfactual(self, tag, obs, conditions, absolute=None):
        _required_data = ('x', 'thickness', 'intensity')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        imgs = [obs['x']]
        if absolute == 'thickness':
            sampled_kdes = {'orig': {'intensity': obs['intensity']}}
        elif absolute == 'intensity':
            sampled_kdes = {'orig': {'thickness': obs['thickness']}}
        else:
            sampled_kdes = {'orig': {'thickness': obs['thickness'], 'intensity': obs['intensity']}}
        measured_kdes = {'orig': {'thickness': obs['thickness'], 'intensity': obs['intensity']}}

        for name, data in conditions.items():
            counterfactual = self.pyro_model._gen_counterfactual(obs=obs, condition=data)

            counter = counterfactual['x']
            sampled_thickness = counterfactual['thickness']
            sampled_intensity = counterfactual['intensity']

            measured_thickness, measured_intensity = self.measure_image(counter.cpu())

            imgs.append(counter)
            if absolute == 'thickness':
                sampled_kdes[name] = {'intensity': sampled_intensity}
            elif absolute == 'intensity':
                sampled_kdes[name] = {'thickness': sampled_thickness}
            else:
                sampled_kdes[name] = {'thickness': sampled_thickness, 'intensity': sampled_intensity}
            measured_kdes[name] = {'thickness': measured_thickness, 'intensity': measured_intensity}

            self.logger.experiment.add_scalar(
                f'{tag}/{name}/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar(
                f'{tag}/{name}/intensity_mae', torch.mean(torch.abs(sampled_intensity.cpu() - measured_intensity)), self.current_epoch)

        self.log_img_grid(tag, torch.cat(imgs, 0))
        self.log_kdes(f'{tag}_sampled', sampled_kdes, save_img=True)
        self.log_kdes(f'{tag}_measured', measured_kdes, save_img=True)

    def sample_images(self):
        with torch.no_grad():
            sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)

            samples = sample_trace.nodes['x']['value']
            sampled_thickness = sample_trace.nodes['thickness']['value']
            sampled_intensity = sample_trace.nodes['intensity']['value']

            self.log_img_grid('samples', samples.data[:8])

            measured_thickness, measured_intensity = self.measure_image(samples)
            self.logger.experiment.add_scalar('samples/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar('samples/intensity_mae', torch.mean(torch.abs(sampled_intensity.cpu() - measured_intensity)), self.current_epoch)

            cond_data = {'thickness': self.thickness_range, 'intensity': self.intensity_range, 'z': self.z_range}
            samples, *_ = pyro.condition(self.pyro_model.sample, data=cond_data)(9)
            self.log_img_grid('cond_samples', samples.data, nrow=3)

            obs_batch = self.prep_batch(self.get_batch(self.val_loader))

            kde_data = {
                'batch': {'thickness': obs_batch['thickness'], 'intensity': obs_batch['intensity']},
                'sampled': {'thickness': sampled_thickness, 'intensity': sampled_intensity}
            }
            self.log_kdes('sample_kde', kde_data, save_img=True)

            exogeneous = self.pyro_model.infer(**obs_batch)

            for (tag, val) in exogeneous.items():
                self.logger.experiment.add_histogram(tag, val, self.current_epoch)

            obs_batch = {k: v[:8] for k, v in obs_batch.items()}

            self.log_img_grid('input', obs_batch['x'], save_img=True)

            if hasattr(self.pyro_model, 'reconstruct'):
                self.build_reconstruction(**obs_batch)

            conditions = {
                '+1': {'thickness': obs_batch['thickness'] + 1},
                '+2': {'thickness': obs_batch['thickness'] + 2},
                '+3': {'thickness': obs_batch['thickness'] + 3}
            }
            self.build_counterfactual('do(thickess=+x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '1': {'thickness': torch.ones_like(obs_batch['thickness'])},
                '2': {'thickness': torch.ones_like(obs_batch['thickness']) * 2},
                '3': {'thickness': torch.ones_like(obs_batch['thickness']) * 3}
            }
            self.build_counterfactual('do(thickess=x)', obs=obs_batch, conditions=conditions, absolute='thickness')

            conditions = {
                '-64': {'intensity': obs_batch['intensity'] - 64},
                '-32': {'intensity': obs_batch['intensity'] - 32},
                '+32': {'intensity': obs_batch['intensity'] + 32},
                '+64': {'intensity': obs_batch['intensity'] + 64}
            }
            self.build_counterfactual('do(intensity=+x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '64': {'intensity': torch.zeros_like(obs_batch['intensity']) + 64},
                '96': {'intensity': torch.zeros_like(obs_batch['intensity']) + 96},
                '192': {'intensity': torch.zeros_like(obs_batch['intensity']) + 192},
                '224': {'intensity': torch.zeros_like(obs_batch['intensity']) + 224}
            }
            self.build_counterfactual('do(intensity=x)', obs=obs_batch, conditions=conditions, absolute='intensity')

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--data_dir', default="/vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/2/", type=str, help="data dir (default: %(default)s)")
        parser.add_argument('--sample_img_interval', default=10, type=int, help="interval in which to sample and log images (default: %(default)s)")
        parser.add_argument('--train_batch_size', default=256, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test_batch_size', default=256, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--validate', default=False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm_lr', default=1e-1, type=float, help="lr of pgm (default: %(default)s)")
        parser.add_argument('--l2', default=0., type=float, help="weight decay (default: %(default)s)")
        parser.add_argument('--use_amsgrad', default=False, action='store_true', help="use amsgrad? (default: %(default)s)")

        return parser
