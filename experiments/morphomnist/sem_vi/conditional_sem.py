import torch
import pyro

from arch.mnist import Decoder, Encoder
from distributions.deep import DeepIndepNormal

from pyro.nn import pyro_method
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import (
    ComposeTransform, AffineTransform, ExpTransform, Spline
)
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from distributions.transforms.affine import ConditionalAffineTransform
from pyro.nn import DenseNN

from experiments.morphomnist.sem_vi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalVISEM(BaseVISEM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # decoder parts
        self.decoder = Decoder(self.latent_dim + 2)

        self.decoder_mean = torch.nn.Conv2d(1, 1, 1)
        self.decoder_logstd = torch.nn.Parameter(torch.ones([]) * self.logstd_init)
        # Flow for modelling t Gamma
        self.t_flow_components = ComposeTransformModule([Spline(1)])
        self.t_flow_lognorm = AffineTransform(loc=0., scale=1.)
        self.t_flow_constraint_transforms = ComposeTransform([self.t_flow_lognorm, ExpTransform()])
        self.t_flow_transforms = ComposeTransform([self.t_flow_components, self.t_flow_constraint_transforms])

        # affine flow for s normal
        slant_net = DenseNN(1, [1], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.s_flow_components = ConditionalAffineTransform(context_nn=slant_net, event_dim=0)
        self.s_flow_norm = AffineTransform(loc=0., scale=1.)
        self.s_flow_transforms = [self.s_flow_components, self.s_flow_norm]

        # encoder parts
        self.encoder = Encoder(self.hidden_dim)

        # TODO: do we need to replicate the PGM here to be able to run conterfactuals? oO
        latent_layers = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim + 2, self.hidden_dim), torch.nn.ReLU())
        self.latent_encoder = DeepIndepNormal(latent_layers, self.hidden_dim, self.latent_dim)

    @pyro_method
    def pgm_model(self):
        t_bd = Normal(self.e_t_loc, self.e_t_scale).to_event(1)
        t_dist = TransformedDistribution(t_bd, self.t_flow_transforms)

        thickness = pyro.sample('thickness', t_dist)
        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        # pseudo call to t_flow_transforms to register with pyro
        _ = self.t_flow_components

        s_bd = Normal(self.e_s_loc, self.e_s_scale).to_event(1)
        s_dist = ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness_)

        slant = pyro.sample('slant', s_dist)
        # pseudo call to s_flow_transforms to register with pyro
        _ = self.s_flow_components

        return thickness, slant

    @pyro_method
    def model(self):
        thickness, slant = self.pgm_model()

        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        slant_ = self.s_flow_norm.inv(slant)

        z = pyro.sample('z', Normal(self.e_z_loc, self.e_z_scale).to_event(1))

        latent = torch.cat([z, thickness_, slant_], 1)

        x_loc = self.decoder_mean(self.decoder(latent))
        x_scale = torch.exp(self.decoder_logstd)
        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)

        preprocess_transform = self._get_preprocess_transforms()
        x_dist = TransformedDistribution(x_bd, ComposeTransform([AffineTransform(x_loc, x_scale, 3), preprocess_transform]))

        x = pyro.sample('x', x_dist)

        return x, z, thickness, slant

    @pyro_method
    def guide(self, x, thickness, slant):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            thickness_ = self.t_flow_constraint_transforms.inv(thickness)
            slant_ = self.s_flow_norm.inv(slant)

            hidden = torch.cat([hidden, thickness_, slant_], 1)
            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z

    @pyro_method
    def infer_e_t(self, t):
        return self.t_flow_transforms.inv(t)

    @pyro_method
    def infer_e_s(self, t, s):
        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(t).transforms)
        return cond_s_transforms.inv(s)


MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM
