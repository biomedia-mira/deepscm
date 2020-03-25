import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader

from arch.mnist import Decoder, Encoder
from datasets.mnist import MNISTLike
from distributions.deep import DeepBernoulli, DeepIndepNormal
from experiments import PyroExperiment
from models.vae_pyro import VAE


class BasicMNISTConvVAE(PyroExperiment):
    def __init__(self):
        super().__init__()
        self.data_dir = "/vol/biomedic/users/dc315/mnist/original"
        self.train_batch_size = 256
        self.test_batch_size = 32

        latent_dim = 10
        hidden_dim = 400
        self.vae = VAE(
            decoder=DeepBernoulli(Decoder(latent_dim)),
            encoder=DeepIndepNormal(Encoder(hidden_dim), hidden_dim, latent_dim),
            hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.svi = SVI(self.vae.model, self.vae.guide, Adam({'lr': 1e-3}), Trace_ELBO())

    def train_dataloader(self):
        train_set = MNISTLike(self.data_dir, train=True)
        return DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True)

    def test_dataloader(self):
        test_set = MNISTLike(self.data_dir, train=False)
        return DataLoader(test_set, batch_size=self.test_batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1) / 255.
        loss = self.svi.step(x)
        return {'loss': torch.Tensor([loss])}


if __name__ == '__main__':
    from pytorch_lightning import Trainer

    experiment = BasicMNISTConvVAE()
    trainer = Trainer(gpus=0, max_steps=1)
    trainer.fit(experiment)

    # Debug saving & loading Pyro param store
    PyroExperiment.debug_pyro_checkpoint = True
    ckpt_path = "test.ckpt"
    trainer.save_checkpoint(ckpt_path)
    BasicMNISTConvVAE.load_from_checkpoint(ckpt_path)
