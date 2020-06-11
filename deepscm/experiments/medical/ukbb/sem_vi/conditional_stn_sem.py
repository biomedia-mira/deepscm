import torch
import pyro
import warnings

from deepscm.arch.medical import Decoder, Encoder
from deepscm.distributions.deep import DeepIndepNormal

from pyro.nn import pyro_method
from pyro.distributions import Normal, Bernoulli, TransformedDistribution
from pyro.distributions.transforms import (
    ComposeTransform, AffineTransform, ExpTransform, Spline
)
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from deepscm.distributions.transforms.affine import ConditionalAffineTransform
from pyro.nn import DenseNN
from torch import nn

from deepscm.experiments.medical.ukbb.sem_vi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalSTNVISEM(BaseVISEM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # decoder parts
        self.decoder = Decoder(num_convolutions=self.num_convolutions, filters=self.dec_filters, latent_dim=self.latent_dim + 3, upconv=self.use_upconv)

        self.decoder_mean = torch.nn.Conv2d(1, 1, 1)
        self.decoder_logstd = torch.nn.Parameter(torch.ones([]) * self.logstd_init)

        self.decoder_affine_param_net = nn.Sequential(
            nn.Linear(self.latent_dim + 3, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 6)
        )

        self.decoder_affine_param_net[-1].weight.data.zero_()
        self.decoder_affine_param_net[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # age flow
        self.age_flow_components = ComposeTransformModule([Spline(1)])
        self.age_flow_lognorm = AffineTransform(loc=0., scale=1.)
        self.age_flow_constraint_transforms = ComposeTransform([self.age_flow_lognorm, ExpTransform()])
        self.age_flow_transforms = ComposeTransform([self.age_flow_components, self.age_flow_constraint_transforms])

        # ventricle_volume flow
        # TODO: decide on how many things to condition on
        ventricle_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.ventricle_volume_flow_components = ConditionalAffineTransform(context_nn=ventricle_volume_net, event_dim=0)
        self.ventricle_volume_flow_lognorm = AffineTransform(loc=0., scale=1.)
        self.ventricle_volume_flow_constraint_transforms = ComposeTransform([self.ventricle_volume_flow_lognorm, ExpTransform()])
        self.ventricle_volume_flow_transforms = [self.ventricle_volume_flow_components, self.ventricle_volume_flow_constraint_transforms]

        # brain_volume flow
        # TODO: decide on how many things to condition on
        brain_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.brain_volume_flow_components = ConditionalAffineTransform(context_nn=brain_volume_net, event_dim=0)
        self.brain_volume_flow_lognorm = AffineTransform(loc=0., scale=1.)
        self.brain_volume_flow_constraint_transforms = ComposeTransform([self.brain_volume_flow_lognorm, ExpTransform()])
        self.brain_volume_flow_transforms = [self.brain_volume_flow_components, self.brain_volume_flow_constraint_transforms]

        # encoder parts
        self.encoder = Encoder(num_convolutions=self.num_convolutions, filters=self.enc_filters, latent_dim=self.latent_dim)

        # TODO: do we need to replicate the PGM here to be able to run conterfactuals? oO
        latent_layers = torch.nn.Sequential(torch.nn.Linear(self.latent_dim + 3, self.latent_dim), torch.nn.ReLU())
        self.latent_encoder = DeepIndepNormal(latent_layers, self.latent_dim, self.latent_dim)

    @pyro_method
    def pgm_model(self):
        sex_dist = Bernoulli(self.sex_logits).to_event(1)

        sex = pyro.sample('sex', sex_dist)

        age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)

        age = pyro.sample('age', age_dist)
        age_ = self.age_flow_constraint_transforms.inv(age)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.age_flow_transforms

        context = torch.cat([sex, age_], 1)

        ventricle_volume_base_dist = Normal(self.ventricle_volume_base_loc, self.ventricle_volume_base_scale).to_event(1)
        ventricle_volume_dist = ConditionalTransformedDistribution(ventricle_volume_base_dist, self.ventricle_volume_flow_transforms).condition(context)

        ventricle_volume = pyro.sample('ventricle_volume', ventricle_volume_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.ventricle_volume_flow_transforms

        brain_volume_base_dist = Normal(self.brain_volume_base_loc, self.brain_volume_base_scale).to_event(1)
        brain_volume_dist = ConditionalTransformedDistribution(brain_volume_base_dist, self.brain_volume_flow_transforms).condition(context)

        brain_volume = pyro.sample('brain_volume', brain_volume_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.brain_volume_flow_transforms

        return age, sex, ventricle_volume, brain_volume

    @pyro_method
    def model(self):
        age, sex, ventricle_volume, brain_volume = self.pgm_model()

        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(ventricle_volume)
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)
        age_ = self.age_flow_constraint_transforms.inv(age)

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        latent = torch.cat([z, age_, ventricle_volume_, brain_volume_], 1)

        x_loc = self.decoder_mean(self.decoder(latent))
        x_scale = torch.exp(self.decoder_logstd)

        theta = self.decoder_affine_param_net(latent).view(-1, 2, 3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            grid = nn.functional.affine_grid(theta, x_loc.size())
            x_loc_deformed = nn.functional.grid_sample(x_loc, grid)

        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)

        preprocess_transform = self._get_preprocess_transforms()
        x_dist = TransformedDistribution(x_base_dist, ComposeTransform([AffineTransform(x_loc_deformed, x_scale, 3), preprocess_transform]))

        x = pyro.sample('x', x_dist)

        return x, z, age, sex, ventricle_volume, brain_volume

    @pyro_method
    def guide(self, x, age, sex, ventricle_volume, brain_volume):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(ventricle_volume)
            brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)
            age_ = self.age_flow_constraint_transforms.inv(age)

            hidden = torch.cat([hidden, age_, ventricle_volume_, brain_volume_], 1)

            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z


MODEL_REGISTRY[ConditionalSTNVISEM.__name__] = ConditionalSTNVISEM
