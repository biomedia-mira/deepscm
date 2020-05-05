import pyro

from pyro.nn import PyroModule, pyro_method

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
    def add_arguments(cls, parser):
        parser.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing (default: %(default)s)", choices=['realnvp', 'glow'])

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
            import random

            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.autograd.set_detect_anomaly(self.hparams.validate)
            pyro.enable_validation()

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
        slants = 25 * (torch.arange(3, dtype=torch.float, device=self.device) - 1)
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
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--data_dir', default="/vol/biomedic2/np716/data/gemini/synthetic/2_more_slant/", type=str, help="data dir (default: %(default)s)")
        parser.add_argument('--sample_img_interval', default=10, type=int, help="interval in which to sample and log images (default: %(default)s)")
        parser.add_argument('--train_batch_size', default=256, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test_batch_size', default=256, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--validate', default=False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm_lr', default=1e-1, type=float, help="lr of pgm (default: %(default)s)")
        parser.add_argument('--l2', default=0., type=float, help="weight decay (default: %(default)s)")
        parser.add_argument('--use_amsgrad', default=False, action='store_true', help="use amsgrad? (default: %(default)s)")

        return parser
