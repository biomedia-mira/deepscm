import pytorch_lightning as pl
import torch
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader

from arch.mnist import Decoder, Encoder
from distributions.deep import DeepBernoulli, DeepIndepNormal
from models.vae_pyro import VAE
from datasets.mnist import MNISTLike


def get_traces(model, guide, *args, **kwargs):
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_replay = poutine.replay(model, trace=guide_trace)
    model_trace = poutine.trace(model_replay).get_trace(*args, **kwargs)
    return model_trace


class BasicMNISTConvVAE(pl.LightningModule):
    def __init__(self, use_cuda=True):
        super().__init__()
        self.data_dir = "/vol/biomedic/users/dc315/mnist/original"
        self.train_batch_size = 256
        self.test_batch_size = 32
        self.dl_kwargs = dict(num_workers=1, pin_memory=True) if use_cuda else {}

        latent_dim = 10
        hidden_dim = 400
        self.vae = VAE(
            decoder=DeepBernoulli(Decoder(latent_dim)),
            encoder=DeepIndepNormal(Encoder(hidden_dim), hidden_dim, latent_dim),
            hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        # self.elbo = Trace_ELBO()
        self.svi = SVI(self.vae.model, self.vae.guide, Adam({'lr': 1e-3}), Trace_ELBO())

    def train_dataloader(self):
        train_set = MNISTLike(self.data_dir, train=True)
        return DataLoader(train_set, batch_size=self.train_batch_size,
                          shuffle=True, **self.dl_kwargs)

    def test_dataloader(self):
        test_set = MNISTLike(self.data_dir, train=False)
        return DataLoader(test_set, batch_size=self.test_batch_size,
                          shuffle=True, **self.dl_kwargs)

    def _get_parameters(self, *args, **kwargs):
        # Adapted from pyro.infer.svi.step()
        with poutine.trace(param_only=True) as param_capture:
            self.elbo.loss(self.model, self.guide, *args, **kwargs)

        return set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())

    def configure_optimizers(self):
        return [None]

    def optimizer_step(self, *args, **kwargs):
        pass

    def forward(self, x):
        pass

    def backward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1) / 255.
        loss = self.svi.step(x)
        return {'loss': torch.Tensor([loss])}


if __name__ == '__main__':
    from pytorch_lightning import Trainer

    Trainer(gpus=1).fit(BasicMNISTConvVAE())
