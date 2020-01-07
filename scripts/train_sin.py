import os
from numbers import Number
from typing import Optional, Sequence, Union
import torch
import torch.distributions as td
from torch.utils.data import DataLoader
from scripts import data_util, spec_util
from models import sin, gmm, vae, mixture, mv_vae
import itertools
from scripts.train_util import TensorBoardLogger, save_checkpoint, load_checkpoint, set_seed


def main(device: str,
         job_dir: str,
         resume: bool,
         save_every: int,
         data_dirs: Union[str, Sequence[str]],
         weights: Optional[Sequence[Number]],
         latent_dim: int,
         n_components: int,
         num_epochs: int,
         train_batch_size: int,
         test_batch_size: int,
         seed: int,
         lr: float,
         sample_all_components: bool,
         use_double: bool):

    set_seed(seed)

    use_cuda = (args.device == 'cuda') if device else torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    dataset_names = [os.path.split(data_dir)[-1] for data_dir in data_dirs]

    job_name = spec_util.format_setup_spec('SIN', latent_dim, dataset_names)
    print(f"Training {job_name}...")
    job_dir = os.path.join(job_dir, job_name)
    logger = TensorBoardLogger(os.path.join(job_dir, 'logs'))
    ckpt_dir = os.path.join(job_dir, 'checkpoints')

    train_set = data_util.get_dataset(data_dirs, weights, train=True)
    test_set = data_util.get_dataset(data_dirs, weights, train=False)

    dl_kwargs = dict(num_workers=1, pin_memory=True) if use_cuda else {}
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, **dl_kwargs)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, **dl_kwargs)
    num_batches = len(train_loader.dataset) // train_loader.batch_size

    model = sin.MixtureSIN(
        gmm.MultivariateGMM(n_components, latent_dim),
        mv_vae.Encoder(latent_dim), vae.Decoder(latent_dim),
        gmm.MultivariateGMM(n_components, latent_dim))
    if use_double:
        model = model.double()
    model.to(device)

    trainer = sin.Trainer(model, lr, sample_all_components)
    tester = vae.Tester(model)

    test_cycle = itertools.cycle(test_loader)

    start_epoch = -1
    if resume:
        start_epoch = load_checkpoint(model, ckpt_dir)
        test_outputs = tester.step(next(test_cycle)[0])
        logger.log(test_outputs, start_epoch)
    else:
        if os.path.isdir(ckpt_dir):
            print(f"Clearing existing checkpoints in {ckpt_dir}")
            for filename in os.listdir(ckpt_dir):
                os.remove(os.path.join(ckpt_dir, filename))

    for epoch in range(start_epoch + 1, num_epochs):
        trainer.model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            verbose = batch_idx % 10 == 0
            if verbose:
                print(f"[{epoch}/{num_epochs}: {batch_idx:3d}/{num_batches:3d}] ", end='')

            real_data = data.to(device).unsqueeze(1)
            if use_double:
                real_data = real_data.double() / 255.
            else:
                real_data = real_data.float() / 255.
            trainer.step(real_data, verbose)

        losses = trainer.get_and_reset_losses()
        logger.log(losses, epoch)

        if save_every is not None:
            if epoch % save_every == 0:
                save_checkpoint(model, ckpt_dir, epoch)

        test_outputs = tester.step(next(test_cycle)[0])
        logger.log(test_outputs, epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                        help="device to use for training (default: use CUDA if available)")
    parser.add_argument('--job-dir',
                        help="directory where logs and checkpoints will be saved")
    parser.add_argument('--resume', type=bool, default=False,
                        help="resume training from latest checkpoint, if available")
    parser.add_argument('--save-every', type=int, default=None,
                        help="save training state every given amount of epochs, if 0 never save")
    parser.add_argument('--num-epochs', type=int, required=True,
                        help="total number of epochs")
    parser.add_argument('--train-batch-size', type=int, default=64,
                        help="training batch size (default: %(default)d)")
    parser.add_argument('--test-batch-size', type=int, default=32,
                        help="test batch size (default: %(default)d)")
    parser.add_argument('--data-dirs', nargs='+',
                        required=True,
                        help=("MNIST-like data directory(ies); if more than one is given, "
                              "data will be randomly interleaved"))
    parser.add_argument('--weights', type=float, nargs='+', required=False,
                        help=("weights for randomly interleaving data directories; must be "
                              "positive of the same length as the list of directories"))
    parser.add_argument('--latent-dim', type=int, help="VAE latent dimension", default=64)
    parser.add_argument('--n-components', type=int, help="Number GMM components", default=10)
    parser.add_argument('--seed', type=int, help="Seed", default=42)
    parser.add_argument('--lr', type=float, help="Learning rate", default=1e-4)
    parser.add_argument('--sample-all-components', default=False, action="store_true",
                        help="sample from all components")
    parser.add_argument('--use_double', default=False, action="store_true",
                        help="use double precision")

    args = parser.parse_args()
    print(args)

    main(**args.__dict__)
