import pyro

from pyro.nn import PyroModule, pyro_method

from pyro.distributions import TransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from torch.distributions import Independent

from datasets.medical.ukbb import UKBBDataset
from pyro.distributions.transforms import ComposeTransform, SigmoidTransform, AffineTransform

import torchvision.utils
from torch.utils.data import DataLoader
from experiments import PyroExperiment
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

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

    def prepare_data(self):
        downsample = None if self.hparams.downsample == -1 else self.hparams.downsample
        self.ukbb_train = UKBBDataset('/vol/biomedic2/np716/data/gemini/ukbb/ventricle_brain/train.csv', crop_type='random', downsample=downsample)
        self.ukbb_val = UKBBDataset('/vol/biomedic2/np716/data/gemini/ukbb/ventricle_brain/val.csv', crop_type='center', downsample=downsample)
        self.ukbb_test = UKBBDataset('/vol/biomedic2/np716/data/gemini/ukbb/ventricle_brain/test.csv', crop_type='center', downsample=downsample)

        self.device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device

        # TODO: change ranges and decide what to condition on
        brain_volumes = 800000. + 300000 * torch.arange(3, dtype=torch.float, device=self.device)
        self.brain_volume_range = brain_volumes.repeat(3).unsqueeze(1)
        ventricle_volumes = 10000. + 50000 * torch.arange(3, dtype=torch.float, device=self.device)
        self.ventricle_volume_range = ventricle_volumes.repeat_interleave(3).unsqueeze(1)
        self.z_range = torch.randn([1, self.hparams.latent_dim], device=self.device, dtype=torch.float).repeat((9, 1))

        self.pyro_model.age_flow_lognorm.loc = self.ukbb_train.metrics['age'].log().mean().to(self.device).float()
        self.pyro_model.age_flow_lognorm.scale = self.ukbb_train.metrics['age'].log().std().to(self.device).float()

        self.pyro_model.ventricle_volume_flow_lognorm.loc = self.ukbb_train.metrics['ventricle_volume'].log().mean().to(self.device).float()
        self.pyro_model.ventricle_volume_flow_lognorm.scale = self.ukbb_train.metrics['ventricle_volume'].log().std().to(self.device).float()

        self.pyro_model.brain_volume_flow_lognorm.loc = self.ukbb_train.metrics['brain_volume'].log().mean().to(self.device).float()
        self.pyro_model.brain_volume_flow_lognorm.scale = self.ukbb_train.metrics['brain_volume'].log().std().to(self.device).float()

        if self.hparams.validate:
            print(f'set ventricle_volume_flow_lognorm {self.pyro_model.ventricle_volume_flow_lognorm.loc} +/- {self.pyro_model.ventricle_volume_flow_lognorm.scale}')  # noqa: E501
            print(f'set age_flow_lognorm {self.pyro_model.age_flow_lognorm.loc} +/- {self.pyro_model.age_flow_lognorm.scale}')
            print(f'set brain_volume_flow_lognorm {self.pyro_model.brain_volume_flow_lognorm.loc} +/- {self.pyro_model.brain_volume_flow_lognorm.scale}')

    def train_dataloader(self):
        return DataLoader(self.ukbb_train, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        self.val_loader = DataLoader(self.ukbb_val, batch_size=self.test_batch_size, shuffle=False)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(self.ukbb_test, batch_size=self.test_batch_size, shuffle=False)
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

    def build_reconstruction(self, x, age, sex, ventricle_volume, brain_volume, tag='reconstruction'):
        obs = {'x': x, 'age': age, 'sex': sex, 'ventricle_volume': ventricle_volume, 'brain_volume': brain_volume}

        recon = self.pyro_model.reconstruct(**obs, num_particles=self.hparams.num_sample_particles)
        self.log_img_grid(tag, torch.cat([x, recon], 0))
        self.logger.experiment.add_scalar(f'{tag}/mse', torch.mean(torch.square(x - recon).sum((1, 2, 3))), self.current_epoch)

    def build_counterfactual(self, tag, obs, conditions, absolute=None):
        _required_data = ('x', 'age', 'sex', 'ventricle_volume', 'brain_volume')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        imgs = [obs['x']]
        # TODO: decide which kde's to plot in which configuration
        if absolute == 'brain_volume':
            sampled_kdes = {'orig': {'ventricle_volume': obs['ventricle_volume']}}
        elif absolute == 'ventricle_volume':
            sampled_kdes = {'orig': {'brain_volume': obs['brain_volume']}}
        else:
            sampled_kdes = {'orig': {'brain_volume': obs['brain_volume'], 'ventricle_volume': obs['ventricle_volume']}}

        for name, data in conditions.items():
            counterfactual = self.pyro_model._gen_counterfactual(obs=obs, condition=data)

            counter = counterfactual['x']
            sampled_brain_volume = counterfactual['brain_volume']
            sampled_ventricle_volume = counterfactual['ventricle_volume']

            imgs.append(counter)
            if absolute == 'brain_volume':
                sampled_kdes[name] = {'ventricle_volume': sampled_ventricle_volume}
            elif absolute == 'ventricle_volume':
                sampled_kdes[name] = {'brain_volume': sampled_brain_volume}
            else:
                sampled_kdes[name] = {'brain_volume': sampled_brain_volume, 'ventricle_volume': sampled_ventricle_volume}

        self.log_img_grid(tag, torch.cat(imgs, 0))
        self.log_kdes(f'{tag}_sampled', sampled_kdes, save_img=True)

    def sample_images(self):
        with torch.no_grad():
            # TODO: redo all this....
            sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)

            samples = sample_trace.nodes['x']['value']
            sampled_brain_volume = sample_trace.nodes['brain_volume']['value']
            sampled_ventricle_volume = sample_trace.nodes['ventricle_volume']['value']

            self.log_img_grid('samples', samples.data[:8])

            cond_data = {'brain_volume': self.brain_volume_range, 'ventricle_volume': self.ventricle_volume_range, 'z': self.z_range}
            samples, *_ = pyro.condition(self.pyro_model.sample, data=cond_data)(9)
            self.log_img_grid('cond_samples', samples.data, nrow=3)

            obs_batch = self.prep_batch(self.get_batch(self.val_loader))

            kde_data = {
                'batch': {'brain_volume': obs_batch['brain_volume'], 'ventricle_volume': obs_batch['ventricle_volume']},
                'sampled': {'brain_volume': sampled_brain_volume, 'ventricle_volume': sampled_ventricle_volume}
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
                '40': {'age': torch.zeros_like(obs_batch['age']) + 40},
                '60': {'age': torch.zeros_like(obs_batch['age']) + 60},
                '80': {'age': torch.zeros_like(obs_batch['age']) + 80}
            }
            self.build_counterfactual('do(age=x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '0': {'sex': torch.zeros_like(obs_batch['sex'])},
                '1': {'sex': torch.ones_like(obs_batch['sex'])},
            }
            self.build_counterfactual('do(sex=x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '800000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume'] + 800000)},
                '1100000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume'] + 1100000)},
                '1400000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume'] + 1400000)},
                '1600000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume'] + 1600000)}
            }
            self.build_counterfactual('do(brain_volume=x)', obs=obs_batch, conditions=conditions, absolute='brain_volume')

            conditions = {
                '10000': {'ventricle_volume': torch.zeros_like(obs_batch['ventricle_volume']) + 10000},
                '25000': {'ventricle_volume': torch.zeros_like(obs_batch['ventricle_volume']) + 25000},
                '50000': {'ventricle_volume': torch.zeros_like(obs_batch['ventricle_volume']) + 50000},
                '75000': {'ventricle_volume': torch.zeros_like(obs_batch['ventricle_volume']) + 75000},
                '110000': {'ventricle_volume': torch.zeros_like(obs_batch['ventricle_volume']) + 110000}
            }
            self.build_counterfactual('do(ventricle_volume=x)', obs=obs_batch, conditions=conditions, absolute='ventricle_volume')

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--sample_img_interval', default=10, type=int, help="interval in which to sample and log images (default: %(default)s)")
        parser.add_argument('--train_batch_size', default=64, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test_batch_size', default=64, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--validate', default=False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm_lr', default=1e-1, type=float, help="lr of pgm (default: %(default)s)")
        parser.add_argument('--l2', default=0., type=float, help="weight decay (default: %(default)s)")
        parser.add_argument('--use_amsgrad', default=False, action='store_true', help="use amsgrad? (default: %(default)s)")
        parser.add_argument('--downsample', default=-1, type=int, help="downsampling factor (default: %(default)s)")

        return parser
