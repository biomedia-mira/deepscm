import torch
import pyro

from arch.mnist import Decoder, Encoder
from distributions.deep import DeepIndepNormal

from pyro.nn import pyro_method
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import (
    ComposeTransform, AffineTransform, ExpTransform, Spline, SigmoidTransform
)
from pyro.distributions.torch_transform import ComposeTransformModule
from distributions.transforms.affine import LearnedAffineTransform

from experiments.morphomnist.sem_vi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalDecoderVISEM(BaseVISEM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # decoder parts
        self.decoder = Decoder(self.latent_dim + 2)

        self.decoder_mean = torch.nn.Conv2d(1, 1, 1)
        self.decoder_logstd = torch.nn.Parameter(torch.ones([]) * self.logstd_init)
        # Flow for modelling t Gamma
        self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        self.thickness_flow_lognorm = AffineTransform(loc=0., scale=1.)
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = ComposeTransform([self.thickness_flow_components, self.thickness_flow_constraint_transforms])

        # affine flow for s normal
        self.width_flow_components = ComposeTransformModule([LearnedAffineTransform(), Spline(1)])
        self.width_flow_norm = AffineTransform(loc=0., scale=1.)
        self.width_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.width_flow_norm])
        self.width_flow_transforms = [self.width_flow_components, self.width_flow_constraint_transforms]

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
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.thickness_flow_components

        width_base_dist = Normal(self.width_base_loc, self.width_base_scale).to_event(1)
        width_dist = TransformedDistribution(width_base_dist, self.width_flow_transforms)

        width = pyro.sample('width', width_dist)
        # pseudo call to width_flow_transforms to register with pyro
        _ = self.width_flow_components

        return thickness, width

    @pyro_method
    def model(self):
        thickness, width = self.pgm_model()

        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        width_ = self.width_flow_norm.inv(width)

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        latent = torch.cat([z, thickness_, width_], 1)

        x_loc = self.decoder_mean(self.decoder(latent))
        x_scale = torch.exp(self.decoder_logstd)
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)

        preprocess_transform = self._get_preprocess_transforms()
        x_dist = TransformedDistribution(x_base_dist, ComposeTransform([AffineTransform(x_loc, x_scale, 3), preprocess_transform]))

        x = pyro.sample('x', x_dist)

        return x, z, thickness, width

    @pyro_method
    def guide(self, x, thickness, width):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
            width_ = self.width_flow_norm.inv(width)

            hidden = torch.cat([hidden, thickness_, width_], 1)
            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z

    @pyro_method
    def infer_thickness_base(self, thickness):
        return self.thickness_flow_transforms.inv(thickness)

    @pyro_method
    def infer_width_base(self, width):
        return self.width_flow_transforms.inv(width)


MODEL_REGISTRY[ConditionalDecoderVISEM.__name__] = ConditionalDecoderVISEM
