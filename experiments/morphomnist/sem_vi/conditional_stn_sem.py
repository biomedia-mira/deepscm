import torch
import pyro
import warnings

from arch.mnist import Decoder, Encoder
from distributions.deep import DeepIndepNormal

from torch import nn
from pyro.nn import pyro_method
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import (
    ComposeTransform, AffineTransform, ExpTransform, Spline, SigmoidTransform
)
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from distributions.transforms.affine import ConditionalAffineTransform
from pyro.nn import DenseNN

from experiments.morphomnist.sem_vi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalSTNVISEM(BaseVISEM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # decoder parts
        self.decoder = Decoder(self.latent_dim + 2)

        self.decoder_mean = torch.nn.Conv2d(1, 1, 1)
        self.decoder_logstd = torch.nn.Parameter(torch.ones([]) * self.logstd_init)

        self.decoder_affine_param_net = nn.Sequential(
            nn.Linear(self.latent_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 6)
        )

        self.decoder_affine_param_net[-1].weight.data.zero_()
        self.decoder_affine_param_net[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Flow for modelling t Gamma
        self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        self.thickness_flow_lognorm = AffineTransform(loc=0., scale=1.)
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = ComposeTransform([self.thickness_flow_components, self.thickness_flow_constraint_transforms])

        # affine flow for s normal
        intensity_net = DenseNN(1, [1], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.intensity_flow_components = ConditionalAffineTransform(context_nn=intensity_net, event_dim=0)
        self.intensity_flow_norm = AffineTransform(loc=0., scale=1.)
        self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        self.intensity_flow_transforms = [self.intensity_flow_components, self.intensity_flow_constraint_transforms]

        # encoder parts
        self.encoder = Encoder(self.hidden_dim)

        # TODO: do we need to replicate the PGM here to be able to run conterfactuals? oO
        latent_layers = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim + 2, self.hidden_dim), torch.nn.ReLU())
        self.latent_encoder = DeepIndepNormal(latent_layers, self.hidden_dim, self.latent_dim)

    @pyro_method
    def pgm_model(self):
        thickness_base_dist = Normal(self.thickness_base_loc, self.thickness_base_scale).to_event(1)
        thickness_dist = TransformedDistribution(thickness_base_dist, self.thickness_flow_transforms)

        thickness = pyro.sample('thickness', thickness_dist)
        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.thickness_flow_components

        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale).to_event(1)
        intensity_dist = ConditionalTransformedDistribution(intensity_base_dist, self.intensity_flow_transforms).condition(thickness_)

        intensity = pyro.sample('intensity', intensity_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.intensity_flow_components

        return thickness, intensity

    @pyro_method
    def model(self):
        thickness, intensity = self.pgm_model()

        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        intensity_ = self.intensity_flow_norm.inv(intensity)

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        latent = torch.cat([z, thickness_, intensity_], 1)

        x_loc = self.decoder_mean(self.decoder(latent))

        theta = self.decoder_affine_param_net(latent).view(-1, 2, 3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            grid = nn.functional.affine_grid(theta, x_loc.size())
            x_loc_deformed = nn.functional.grid_sample(x_loc, grid)

        x_scale = torch.exp(self.decoder_logstd)
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)

        preprocess_transform = self._get_preprocess_transforms()
        x_dist = TransformedDistribution(x_base_dist, ComposeTransform([AffineTransform(x_loc_deformed, x_scale, 3), preprocess_transform]))

        x = pyro.sample('x', x_dist)

        return x, z, thickness, intensity

    @pyro_method
    def guide(self, x, thickness, intensity):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
            intensity_ = self.intensity_flow_norm.inv(intensity)

            hidden = torch.cat([hidden, thickness_, intensity_], 1)
            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z

    @pyro_method
    def infer_thickness_base(self, thickness):
        return self.thickness_flow_transforms.inv(thickness)

    @pyro_method
    def infer_intensity_base(self, thickness, intensity):
        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale)
        cond_intensity_transforms = ComposeTransform(
            ConditionalTransformedDistribution(intensity_base_dist, self.intensity_flow_transforms).condition(thickness).transforms)
        return cond_intensity_transforms.inv(intensity)


MODEL_REGISTRY[ConditionalSTNVISEM.__name__] = ConditionalSTNVISEM
