import torch
import torch.distributions as td
from torch import nn
from .vae import VAE


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
        cov_low_tri_dim = int((latent_dim * (latent_dim - 1)) / 2)
        self.fc2_tril = nn.Linear(self.h3_dim, cov_low_tri_dim)
        self.fc2_diag = nn.Linear(self.h3_dim, latent_dim)

    def tril(self, diag: torch.Tensor, tril_vec: torch.Tensor):
        dim = diag.shape[-1]
        L = torch.diag_embed(diag)  # L is lower-triangular
        L = L.to(diag.device)
        i, j = torch.tril_indices(dim, dim, offset=-1)
        L[..., i, j] = tril_vec

        return L

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        x = self.fc1(x)
        mean = self.fc2_mean(x)
        diag = torch.exp(self.fc2_diag(x))
        tril_vec = self.fc2_tril(x)

        tril = self.tril(diag, tril_vec)

        return mean, tril

    def posterior(self, x):
        mean, tril = self.forward(x)

        return td.MultivariateNormal(mean, scale_tril=tril)


class MVVAE(VAE):
    def _get_prior(self, latent_dim, device):
        mean = torch.zeros(latent_dim, device=device)
        cov = torch.diag(torch.ones(latent_dim, device=device))
        return td.MultivariateNormal(mean, cov, validate_args=True)
