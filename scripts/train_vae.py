import os
from numbers import Number
from typing import Optional, Sequence, Union
import torch
from torch.utils.data import DataLoader
from scripts import data_util, spec_util
from models import mv_vae, vae
import itertools
from scripts.train_util import TensorBoardLogger, save_checkpoint, load_checkpoint


def main(device: str,
         job_dir: str,
         resume: bool,
         save_every: int,
         data_dirs: Union[str, Sequence[str]],
         weights: Optional[Sequence[Number]],
         latent_dim: int,
         num_epochs: int,
         train_batch_size: int,
         test_batch_size: int,
         mvvae: bool):

    use_cuda = (args.device == 'cuda') if device else torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    dataset_names = [os.path.split(data_dir)[-1] for data_dir in data_dirs]

    job_name = spec_util.format_setup_spec('VAE', latent_dim, dataset_names)
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

    model = mv_vae.MVVAE(latent_dim, device, encoder=mv_vae.Encoder) if mvvae else vae.VAE(latent_dim, device)
    model.to(device)
    trainer = vae.Trainer(model)
    tester = vae.Tester(model)

    test_cycle = itertools.cycle(test_loader)

    start_epoch = -1
    if resume:
        start_epoch = load_checkpoint(model, ckpt_dir)
        test_outputs = tester.step(next(test_cycle)[0])
        logger.log(test_outputs, start_epoch)
    else:
        print(f"Clearing existing checkpoints in {ckpt_dir}")
        for filename in os.listdir(ckpt_dir):
            os.remove(os.path.join(ckpt_dir, filename))

    for epoch in range(start_epoch + 1, num_epochs):
        trainer.model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            verbose = batch_idx % 10 == 0
            if verbose:
                print(f"[{epoch}/{num_epochs}: {batch_idx:3d}/{num_batches:3d}] ", end='')

            real_data = data.to(device).unsqueeze(1).float() / 255.
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
    parser.add_argument('--latent-dim', type=int, required=True,
                        help="VAE latent dimension", default=64)
    parser.add_argument('--mvvae', default=False, action='store_true',
                        help="Whether to predict full covariance")

    args = parser.parse_args()
    print(args)

    main(**args.__dict__)
