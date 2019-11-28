import torch
import torch.distributions as td
from torch import nn


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

    def posterior(self, x):
        mean, logvar = self.forward(x)
        std = (.5 * logvar).exp()
        return td.Independent(td.Normal(mean, std), 1)


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

    def likelihood(self, x):
        probs = self.forward(x)
        return td.Independent(td.Bernoulli(probs), 3)


class VAE(nn.Module):
    def __init__(self, latent_dim: int, device=None,
                 encoder=Encoder, decoder=Decoder):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = encoder(self.latent_dim)
        self.dec = decoder(self.latent_dim)
        if device is None:
            device = self.device

        self.prior = self._get_prior(latent_dim, device)

        self.apply(_weights_init)

    def _get_prior(self, latent_dim, device):
        mean, std = torch.zeros(latent_dim, device=device), torch.ones(latent_dim, device=device)
        return td.Independent(td.Normal(mean, std), 1)

    def forward(self, data):
        posteriors = self.enc.posterior(data)
        latents = posteriors.rsample()
        likelihoods = self.dec.likelihood(latents)
        return posteriors, latents, likelihoods

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

    def step(self, data, verbose: bool = False):
        posteriors, latents, likelihoods = self.model(data)

        rec_loss = -likelihoods.log_prob((data > .5).float())
        kl_div = td.kl_divergence(posteriors, self.model.prior)

        rec_loss = rec_loss.sum()
        kl_div = kl_div.sum()

        self.opt.zero_grad()
        (rec_loss + self.beta * kl_div).backward()
        self.opt.step()

        if verbose:
            print(f"rec_loss = {rec_loss.item():6g}, KL_div = {kl_div.item():6g}")

    def forward(self, real_data, verbose: bool = False):
        self.step(real_data, verbose)
