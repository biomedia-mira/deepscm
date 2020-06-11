import torch
import pyro

from pyro.nn import pyro_method
from pyro.distributions import Normal, Bernoulli, TransformedDistribution
from pyro.distributions.conditional import ConditionalTransformedDistribution
from deepscm.distributions.transforms.affine import ConditionalAffineTransform
from pyro.nn import DenseNN

from deepscm.experiments.medical.ukbb.sem_vi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalVISEM(BaseVISEM):
    context_dim = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ventricle_volume flow
        ventricle_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.ventricle_volume_flow_components = ConditionalAffineTransform(context_nn=ventricle_volume_net, event_dim=0)
        self.ventricle_volume_flow_transforms = [self.ventricle_volume_flow_components, self.ventricle_volume_flow_constraint_transforms]

        # brain_volume flow
        brain_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.brain_volume_flow_components = ConditionalAffineTransform(context_nn=brain_volume_net, event_dim=0)
        self.brain_volume_flow_transforms = [self.brain_volume_flow_components, self.brain_volume_flow_constraint_transforms]

    @pyro_method
    def pgm_model(self):
        sex_dist = Bernoulli(logits=self.sex_logits).to_event(1)

        _ = self.sex_logits

        sex = pyro.sample('sex', sex_dist)

        age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)

        age = pyro.sample('age', age_dist)
        age_ = self.age_flow_constraint_transforms.inv(age)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.age_flow_components

        brain_context = torch.cat([sex, age_], 1)

        brain_volume_base_dist = Normal(self.brain_volume_base_loc, self.brain_volume_base_scale).to_event(1)
        brain_volume_dist = ConditionalTransformedDistribution(brain_volume_base_dist, self.brain_volume_flow_transforms).condition(brain_context)

        brain_volume = pyro.sample('brain_volume', brain_volume_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.brain_volume_flow_components

        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

        ventricle_context = torch.cat([age_, brain_volume_], 1)

        ventricle_volume_base_dist = Normal(self.ventricle_volume_base_loc, self.ventricle_volume_base_scale).to_event(1)
        ventricle_volume_dist = ConditionalTransformedDistribution(ventricle_volume_base_dist, self.ventricle_volume_flow_transforms).condition(ventricle_context)  # noqa: E501

        ventricle_volume = pyro.sample('ventricle_volume', ventricle_volume_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.ventricle_volume_flow_components

        return age, sex, ventricle_volume, brain_volume

    @pyro_method
    def model(self):
        age, sex, ventricle_volume, brain_volume = self.pgm_model()

        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(ventricle_volume)
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        latent = torch.cat([z, ventricle_volume_, brain_volume_], 1)

        x_dist = self._get_transformed_x_dist(latent)

        x = pyro.sample('x', x_dist)

        return x, z, age, sex, ventricle_volume, brain_volume

    @pyro_method
    def guide(self, x, age, sex, ventricle_volume, brain_volume):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(ventricle_volume)
            brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

            hidden = torch.cat([hidden, ventricle_volume_, brain_volume_], 1)

            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z


MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM
