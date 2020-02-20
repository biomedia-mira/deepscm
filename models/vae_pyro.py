import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vae import Decoder, Encoder
from scripts import data_util


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def model(self, x: torch.Tensor):
        pyro.module('decoder', self.decoder)
        with pyro.plate('observations', x.shape[0]):
            z_loc = x.new_zeros(self.latent_dim)
            z_scale = x.new_ones(self.latent_dim)
            z = pyro.sample('z', dist.Normal(z_loc, z_scale).to_event(1))
            mean = self.decoder.forward(z)
            x = pyro.sample('x', dist.Bernoulli(mean).to_event(2), obs=x)

    def guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('observations', x.shape[0]):
            z_mean, z_logvar = self.encoder.forward(x)
            z_scale = (.5 * z_logvar).exp()
            z = pyro.sample('z', dist.Normal(z_mean, z_scale).to_event(1))


if __name__ == '__main__':
    use_cuda = True
    device = 'cuda' if use_cuda else 'cpu'
    vae = VAE(latent_dim=10).to(device)
    svi = SVI(vae.model, vae.guide, Adam({'lr': 1e-3}), Trace_ELBO())

    data_dir = "/vol/biomedic/users/dc315/mnist/original"
    train_set = data_util.get_dataset([data_dir], train=True)
    test_set = data_util.get_dataset([data_dir], train=False)

    train_batch_size = 256
    test_batch_size = 32
    dl_kwargs = dict(num_workers=1, pin_memory=True) if use_cuda else {}
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, **dl_kwargs)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, **dl_kwargs)

    n_epochs = 10
    for epoch in range(n_epochs):
        epoch_loss = 0.
        for x, _ in tqdm(train_loader, leave=False):
            x = x.to(device).unsqueeze(1) / 255.
            epoch_loss += svi.step(x)
        print(f"Epoch {epoch}: {epoch_loss / len(train_loader.dataset)}")
