import pyro
import torch
from pyro.distributions import Normal
from torch import nn

from distributions.deep import DeepConditional


class VAE(nn.Module):
    def __init__(self, decoder: DeepConditional, encoder: DeepConditional,
                 hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.decoder = decoder
        self.encoder = encoder

    def model(self, x: torch.Tensor):
        decoder = pyro.module('decoder', self.decoder)
        with pyro.plate('observations', x.shape[0]):
            z_loc = x.new_zeros(self.latent_dim)
            z_scale = x.new_ones(self.latent_dim)
            z = pyro.sample('z', Normal(z_loc, z_scale).to_event(1))
            x = pyro.sample('x', decoder.predict(z), obs=x)
        return x, z

    def guide(self, x: torch.Tensor):
        encoder = pyro.module('encoder', self.encoder)
        with pyro.plate('observations', x.shape[0]):
            z = pyro.sample('z', encoder.predict(x))
        return z


if __name__ == '__main__':
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from arch.mnist import Decoder, Encoder
    from distributions.deep import DeepBernoulli, DeepIndepNormal
    from datasets.mnist import MNISTLike

    use_cuda = True
    device = 'cuda' if use_cuda else 'cpu'

    latent_dim = 10
    hidden_dim = 400
    vae = VAE(
        decoder=DeepBernoulli(Decoder(latent_dim)),
        encoder=DeepIndepNormal(Encoder(hidden_dim), hidden_dim, latent_dim),
        hidden_dim=hidden_dim, latent_dim=latent_dim
    ).to(device)
    svi = SVI(vae.model, vae.guide, Adam({'lr': 1e-3}), Trace_ELBO())

    data_dir = "/vol/biomedic/users/dc315/mnist/original"
    train_set = MNISTLike(data_dir, train=True)
    test_set = MNISTLike(data_dir, train=False)

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
