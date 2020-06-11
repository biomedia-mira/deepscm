import torch
import pyro

from pyro.nn import pyro_method
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import (
    ComposeTransform, ExpTransform, Spline, SigmoidTransform
)
from pyro.distributions.torch_transform import ComposeTransformModule
from deepscm.distributions.transforms.affine import LearnedAffineTransform

from .base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalDecoderVISEM(BaseVISEM):
    context_dim = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Flow for modelling t Gamma
        self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = ComposeTransform([self.thickness_flow_components, self.thickness_flow_constraint_transforms])

        # affine flow for s normal
        self.intensity_flow_components = ComposeTransformModule([LearnedAffineTransform(), Spline(1)])
        self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        self.intensity_flow_transforms = [self.intensity_flow_components, self.intensity_flow_constraint_transforms]

    @pyro_method
    def pgm_model(self):
        thickness_base_dist = Normal(self.thickness_base_loc, self.thickness_base_scale).to_event(1)
        thickness_dist = TransformedDistribution(thickness_base_dist, self.thickness_flow_transforms)

        thickness = pyro.sample('thickness', thickness_dist)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.thickness_flow_components

        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale).to_event(1)
        intensity_dist = TransformedDistribution(intensity_base_dist, self.intensity_flow_transforms)

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

        x_dist = self._get_transformed_x_dist(latent)

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
    def infer_intensity_base(self, intensity):
        return self.intensity_flow_transforms.inv(intensity)


MODEL_REGISTRY[ConditionalDecoderVISEM.__name__] = ConditionalDecoderVISEM
