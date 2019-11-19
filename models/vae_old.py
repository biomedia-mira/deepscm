import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
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
        self.h3_dim = 1024
        self.fc1 = nn.Sequential(
                nn.Linear(7 * 7 * self.h2_nchan, self.h3_dim),
                nn.BatchNorm1d(self.h3_dim),
                nn.LeakyReLU(.1, inplace=True)
        )
        self.fc2_mean = nn.Linear(self.h3_dim, latent_dim)
        self.fc2_logvar = nn.Linear(self.h3_dim, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        x = self.fc1(x)
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar


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
        self.conv2 = nn.Sequential(
            *conv2_ops,
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def sample_noise(num, dim, device=None) -> torch.Tensor:
    return torch.randn(num, dim, device=device)


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = Encoder(self.latent_dim)
        self.dec = Decoder(self.latent_dim)

        self.apply(_weights_init)

    def sample_latent(self, num: int):
        return sample_noise(num, self.latent_dim, self.device)

    def sample_posterior(self, data, num: int = 1):
        noise = torch.randn(data.shape[0], num, self.latent_dim, device=self.device)
        mean, logvar = self.enc(data)
        latent = mean.unsqueeze(1) + (.5 * logvar).exp().unsqueeze(1) * noise

    def forward(self, data):
        noise = self.sample_latent(data.shape[0])
        mean, logvar = self.enc(data)
        latent = mean + (.5 * logvar).exp() * noise
        recon = self.dec(latent)
        return mean, logvar, latent, recon

    def reconstruct(self, data):
        noise = self.sample_latent(data.shape[0])
        mean, logvar = self.enc(data)
        latent = mean + (.5 * logvar).exp() * noise
        recon = self.dec(latent)
        return mean, logvar, latent, recon

    @property
    def device(self):
        return next(self.parameters()).device


def _weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)


class Trainer(nn.Module):
    def __init__(self, model: VAE, beta: float = 1., lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.beta = beta

        params = list(self.model.enc.parameters()) + list(self.model.dec.parameters())
        self.opt = torch.optim.Adam(params, lr=lr, betas=(.5, .99))

    def step(self, real_data, verbose: bool = False):
        mean, logvar, latent, fake_data = self.model(real_data)

        rec_loss = F.binary_cross_entropy(fake_data, (real_data > .5).float(), size_average=False)
        # rec_loss = F.binary_cross_entropy(fake_data, real_data, size_average=False)
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).sum()

        self.opt.zero_grad()
        (rec_loss + self.beta * kl_div).backward()
        self.opt.step()

        if verbose:
            print(f"rec_loss = {rec_loss.item():6g}, KL_div = {kl_div.item():6g}, ")

    def forward(self, real_data, verbose: bool = False):
        self.step(real_data, verbose)
