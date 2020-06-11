import torch
import pyro

from pyro.nn import pyro_method

from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.distributions.transforms import (
    Spline, ExpTransform, ComposeTransform, ConditionalAffineCoupling,
    GeneralizedChannelPermute, SigmoidTransform
)
from deepscm.distributions.transforms.reshape import ReshapeTransform, SqueezeTransform, TransposeTransform
from deepscm.distributions.transforms.affine import ConditionalAffineTransform
from deepscm.arch.mnist import BasicFlowConvNet
from pyro.nn import DenseNN
from deepscm.distributions.transforms.normalisation import ActNorm

from .base_nf_experiment import BaseFlowSEM, MODEL_REGISTRY


class ConditionalFlowSEM(BaseFlowSEM):
    def __init__(self, use_affine_ex: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_affine_ex = use_affine_ex

        # decoder parts

        # Flow for modelling t Gamma
        self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = ComposeTransform([self.thickness_flow_components, self.thickness_flow_constraint_transforms])

        # affine flow for s normal
        intensity_net = DenseNN(1, [1], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.intensity_flow_components = ConditionalAffineTransform(context_nn=intensity_net, event_dim=0)
        self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        self.intensity_flow_transforms = [self.intensity_flow_components, self.intensity_flow_constraint_transforms]
        # build flow as s_affine_w * t * e_s + b -> depends on t though

        # realnvp or so for x
        self._build_image_flow()

    def _build_image_flow(self):

        self.trans_modules = ComposeTransformModule([])

        self.x_transforms = []

        self.x_transforms += [self._get_preprocess_transforms()]

        c = 1
        for _ in range(self.num_scales):
            self.x_transforms.append(SqueezeTransform())
            c *= 4

            for _ in range(self.flows_per_scale):
                if self.use_actnorm:
                    actnorm = ActNorm(c)
                    self.trans_modules.append(actnorm)
                    self.x_transforms.append(actnorm)

                gcp = GeneralizedChannelPermute(channels=c)
                self.trans_modules.append(gcp)
                self.x_transforms.append(gcp)

                self.x_transforms.append(TransposeTransform(torch.tensor((1, 2, 0))))

                ac = ConditionalAffineCoupling(c // 2, BasicFlowConvNet(c // 2, self.hidden_channels, (c // 2, c // 2), 2))
                self.trans_modules.append(ac)
                self.x_transforms.append(ac)

                self.x_transforms.append(TransposeTransform(torch.tensor((2, 0, 1))))

            gcp = GeneralizedChannelPermute(channels=c)
            self.trans_modules.append(gcp)
            self.x_transforms.append(gcp)

        self.x_transforms += [
            ReshapeTransform((4**self.num_scales, 32 // 2**self.num_scales, 32 // 2**self.num_scales), (1, 32, 32))
        ]

        if self.use_affine_ex:
            affine_net = DenseNN(2, [16, 16], param_dims=[1, 1])
            affine_trans = ConditionalAffineTransform(context_nn=affine_net, event_dim=3)

            self.trans_modules.append(affine_trans)
            self.x_transforms.append(affine_trans)

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
        # pseudo call to w_flow_transforms to register with pyro
        _ = self.intensity_flow_components

        return thickness, intensity

    @pyro_method
    def model(self):
        thickness, intensity = self.pgm_model()

        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        intensity_ = self.intensity_flow_norm.inv(intensity)

        context = torch.cat([thickness_, intensity_], 1)

        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_base_dist, self.x_transforms).condition(context).transforms).inv
        cond_x_dist = TransformedDistribution(x_base_dist, cond_x_transforms)

        x = pyro.sample('x', cond_x_dist)

        return x, thickness, intensity

    @pyro_method
    def infer_thickness_base(self, thickness):
        return self.thickness_flow_transforms.inv(thickness)

    @pyro_method
    def infer_intensity_base(self, thickness, intensity):
        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale)

        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        cond_intensity_transforms = ComposeTransform(
            ConditionalTransformedDistribution(intensity_base_dist, self.intensity_flow_transforms).condition(thickness_).transforms)
        return cond_intensity_transforms.inv(intensity)

    @pyro_method
    def infer_x_base(self, thickness, intensity, x):
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale)

        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        intensity_ = self.intensity_flow_norm.inv(intensity)

        context = torch.cat([thickness_, intensity_], 1)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_base_dist, self.x_transforms).condition(context).transforms)
        return cond_x_transforms(x)

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument(
            '--use_affine_ex', default=False, action='store_true', help="whether to use conditional affine transformation on e_x (default: %(default)s)")

        return parser


MODEL_REGISTRY[ConditionalFlowSEM.__name__] = ConditionalFlowSEM
