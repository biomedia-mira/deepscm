import torch
import pyro

from pyro.nn import pyro_method

from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.distributions.transforms import (
    Spline, ExpTransform, ComposeTransform, ConditionalAffineCoupling,
    GeneralizedChannelPermute, AffineTransform
)
from distributions.transforms.reshape import ReshapeTransform, SqueezeTransform, TransposeTransform
from distributions.transforms.affine import ConditionalAffineTransform
from arch.mnist import BasicFlowConvNet
from pyro.nn import DenseNN
from distributions.transforms.normalisation import ActNorm

from experiments.morphomnist.nf.base_nf_experiment import BaseFlowSEM, MODEL_REGISTRY


class ConditionalFlowSEM(BaseFlowSEM):
    def __init__(self, use_affine_ex: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_affine_ex = use_affine_ex

        # decoder parts

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
        t_bd = Normal(self.e_t_loc, self.e_t_scale)
        t_dist = TransformedDistribution(t_bd, self.t_flow_transforms)

        thickness = pyro.sample('thickness', t_dist.to_event(1))
        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        # pseudo call to t_flow_transforms to register with pyro
        _ = self.t_flow_components

        s_bd = Normal(self.e_s_loc, self.e_s_scale)
        s_dist = ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness_)

        slant = pyro.sample('slant', s_dist.to_event(1))
        # pseudo call to s_flow_transforms to register with pyro
        _ = self.s_flow_components

        return thickness, slant

    @pyro_method
    def model(self):
        thickness, slant = self.pgm_model()

        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        slant_ = self.s_flow_norm.inv(slant)

        context = torch.cat([thickness_, slant_], 1)

        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms).inv
        cond_x_dist = TransformedDistribution(x_bd, cond_x_transforms)

        x = pyro.sample('x', cond_x_dist)

        return x, thickness, slant

    @pyro_method
    def pgm_scm(self):
        t_bd = Normal(self.e_t_loc, self.e_t_scale).to_event(1)
        e_t = pyro.sample('e_t', t_bd)

        thickness = self.t_flow_transforms(e_t)
        thickness = pyro.deterministic('thickness', thickness)
        thickness_ = self.t_flow_constraint_transforms.inv(thickness)

        s_bd = Normal(self.e_s_loc, self.e_s_scale).to_event(1)
        e_s = pyro.sample('e_s', s_bd)

        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness_).transforms)

        slant = cond_s_transforms(e_s)
        slant = pyro.deterministic('slant', slant)

        return thickness, slant

    @pyro_method
    def scm(self):
        thickness, slant = self.pgm_scm()

        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        slant_ = self.s_flow_norm.inv(slant)

        context = torch.cat([thickness_, slant_], 1)

        x_bd = Normal(self.e_x_loc, self.e_x_scale).to_event(3)
        e_x = pyro.sample('e_x', x_bd)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms).inv

        x = cond_x_transforms(e_x)
        x = pyro.deterministic('x', x)

        return x, thickness, slant

    @pyro_method
    def infer_e_t(self, thickness):
        return self.t_flow_transforms.inv(thickness)

    @pyro_method
    def infer_e_s(self, thickness, slant):
        s_bd = Normal(self.e_s_loc, self.e_s_scale)

        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        cond_s_transforms = ComposeTransform(ConditionalTransformedDistribution(s_bd, self.s_flow_transforms).condition(thickness_).transforms)
        return cond_s_transforms.inv(slant)

    @pyro_method
    def infer_e_x(self, thickness, slant, x):
        x_bd = Normal(self.e_x_loc, self.e_x_scale)

        thickness_ = self.t_flow_constraint_transforms.inv(thickness)
        slant_ = self.s_flow_norm.inv(slant)

        context = torch.cat([thickness_, slant_], 1)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_bd, self.x_transforms).condition(context).transforms)
        return cond_x_transforms(x)

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument(
            '--use_affine_ex', default=False, action='store_true', help="whether to use conditional affine transformation on e_x (default: %(default)s)")

        return parser


MODEL_REGISTRY[ConditionalFlowSEM.__name__] = ConditionalFlowSEM
