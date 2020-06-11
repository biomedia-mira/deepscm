import torch
from torch import nn

import numpy as np

from collections.abc import Iterable


class Encoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.h1_nchan = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.h1_nchan, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h2_nchan),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.h3_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * self.h2_nchan, self.h3_dim),
            nn.BatchNorm1d(self.h3_dim),
            nn.LeakyReLU(.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, upconv=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.h1_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.ReLU(inplace=True)
        )
        self.h2_nchan = 128
        h2_dim = 7 * 7 * self.h2_nchan
        self.fc2 = nn.Sequential(
            nn.Linear(self.h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.ReLU(inplace=True)
        )
        self.h3_nchan = 64

        if upconv:
            conv1_ops = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.h2_nchan, self.h3_nchan, kernel_size=5, stride=1, padding=2)
            ]
            conv2_ops = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.h3_nchan, 1, kernel_size=5, stride=1, padding=2)
            ]
        else:
            conv1_ops = [nn.ConvTranspose2d(self.h2_nchan, self.h3_nchan,
                                            kernel_size=4, stride=2, padding=1)]
            conv2_ops = [nn.ConvTranspose2d(self.h3_nchan, 1, kernel_size=4, stride=2, padding=1)]

        self.conv1 = nn.Sequential(
            *conv1_ops,
            nn.BatchNorm2d(self.h3_nchan),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(*conv2_ops)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        return self.conv2(x)


class BasicFlowConvNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, param_dims, context_dims: int = None, param_nonlinearities=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_dims = sum(param_dims)

        self.context_dims = context_dims
        self.param_nonlinearities = param_nonlinearities

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels + context_dims if context_dims is not None else in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.output_dims, kernel_size=3, padding=1)
        )

        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0., 1e-4)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.)

        self.apply(weights_init)

    def forward(self, inputs, context=None):
        # pyro affine coupling splits on the last dimenion and not the channel dimension
        # -> we want to permute the dimensions to move the last dimension into the channel dimension
        # and then transpose back

        if not ((self.context_dims is None) == (context is None)):
            raise ValueError('Given context does not match context dims: context: {} and context_dims:{}'.format(context, self.context_dims))

        *batch_dims, h, w, c = inputs.size()
        num_batch = len(batch_dims)

        permutation = np.array((2, 0, 1)) + num_batch
        outputs = inputs.permute(*np.arange(num_batch), *permutation).contiguous()

        if context is not None:
            # assuming scalar inputs [B, C]
            context = context.view(*context.shape, 1, 1).expand(-1, -1, *outputs.shape[2:])
            outputs = torch.cat([outputs, context], 1)

        outputs = self.seq1(outputs)

        permutation = np.array((1, 2, 0)) + num_batch
        outputs = outputs.permute(*np.arange(num_batch), *permutation).contiguous()

        if self.count_params > 1:
            outputs = tuple(outputs[..., s] for s in self.param_slices)

        if self.param_nonlinearities is not None:
            if isinstance(self.param_nonlinearities, Iterable):
                outputs = tuple(n(o) for o, n in zip(outputs, self.param_nonlinearities))
            else:
                outputs = tuple(self.param_nonlinearities(o) for o in outputs)

        return outputs


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)
