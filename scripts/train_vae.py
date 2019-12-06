import os
from numbers import Number
from typing import Optional, Sequence, Union
import torchvision
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from scripts import data_util, spec_util
# from models import vae_old as vae, load_checkpoint, save_checkpoint
from models import mv_vae, vae, load_checkpoint, save_checkpoint
from morphomnist.util import plot_grid
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, tensor_dict, epoch):
        for name, tensor in tensor_dict.items():
            ndim = tensor.ndim
            if ndim <= 1:
                self.writer.add_scalar(name, tensor, epoch)
            elif ndim == 3:
                self.writer.add_image(name, tensor, epoch)
            elif ndim == 4:
                self.writer.add_images(name, tensor, epoch)
            else:
                raise ValueError(f'No method to log tensor with ndim {ndim:3d}.')
#
# def grayscale_to_rgb(images):


def test(model: vae.VAE, real_data, logger, epoch):
    model.eval()
    with torch.no_grad():
        real_data = real_data.to(model.device).unsqueeze(1).float() / 255.
        recon_data = model(real_data)[-1]
        samples = model.dec(model.prior.sample((len(real_data), )))
        merged = torchvision.utils.make_grid(torch.cat((real_data, recon_data.mean, recon_data.stddev)), nrow=len(real_data))
        logger.log({'data/real': real_data,
                    'data/recon_mean': recon_data.mean,
                    'data/recon_stddev': recon_data.stddev,
                    'data_stacked': merged,
                    'samples': samples},
                   epoch)
        # plot_grid(torch.cat([real_data[:32], recon_data[:32]], dim=0).cpu(),
        #           figsize=(8, 8), gridspec_kw=dict(wspace=0, hspace=0))
        # plt.show()


def main(use_cuda: bool, data_dirs: Union[str, Sequence[str]], weights: Optional[Sequence[Number]],
         ckpt_root: str, latent_dim: int, num_epochs: int,
         batch_size: int, save: bool, resume: bool, plot: bool, mvvae: bool):
    device = torch.device('cuda' if use_cuda else 'cpu')

    logger = TensorBoardLogger(log_dir=os.path.join(ckpt_root, 'logs'))

    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    dataset_names = [os.path.split(data_dir)[-1] for data_dir in data_dirs]
    ckpt_name = spec_util.format_setup_spec('VAE', latent_dim, dataset_names)
    print(f"Training {ckpt_name}...")
    ckpt_dir = None if ckpt_root is None else os.path.join(ckpt_root, ckpt_name)

    train_set = data_util.get_dataset(data_dirs, weights, train=True)
    test_set = data_util.get_dataset(data_dirs, weights, train=False)

    test_batch_size = 32
    dl_kwargs = dict(num_workers=1, pin_memory=True) if use_cuda else {}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **dl_kwargs)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, **dl_kwargs)
    num_batches = len(train_loader.dataset) // train_loader.batch_size

    if mvvae:
        model = mv_vae.MVVAE(latent_dim, device, encoder=mv_vae.Encoder)
    else:
        model = vae.VAE(latent_dim, device)
    trainer = vae.Trainer(model)
    trainer.to(device)

    test_iterator = iter(test_loader)

    start_epoch = -1
    if resume:
        try:
            start_epoch = load_checkpoint(trainer, ckpt_dir)
            if plot:
                test(model, next(test_iterator)[0], logger, start_epoch)
        except ValueError:
            print(f"No checkpoint to resume from in {ckpt_dir}")
        except FileNotFoundError:
            print(f"Invalid checkpoint directory: {ckpt_dir}")
    elif save:
        if os.path.exists(ckpt_dir):
            print(f"Clearing existing checkpoints in {ckpt_dir}")
            for filename in os.listdir(ckpt_dir):
                os.remove(os.path.join(ckpt_dir, filename))
    for epoch in range(start_epoch + 1, num_epochs):
        epoch_outputs = None
        trainer.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            verbose = batch_idx % 10 == 0
            if verbose:
                print(f"[{epoch}/{num_epochs}: {batch_idx:3d}/{num_batches:3d}] ", end='')

            real_data = data.to(device).unsqueeze(1).float() / 255.
            batch_outputs = trainer.step(real_data, verbose)
            if epoch_outputs is None:
                epoch_outputs = batch_outputs
            else:
                for key, value in batch_outputs.items():
                    epoch_outputs[key] += value

        logger.log(epoch_outputs, epoch)

        if save:
            save_checkpoint(trainer, ckpt_dir, epoch)

        if plot:
            test(model, next(test_iterator)[0], logger, epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                        help="device to use for training (default: use CUDA if available)")
    parser.add_argument('--save', action='store_true', default=False,
                        help="save training state after each training epoch")
    parser.add_argument('--resume', action='store_true', default=False,
                        help="resume training from latest checkpoint, if available")
    parser.add_argument('--checkpoint',
                        help="root directory where checkpoints are saved")
    parser.add_argument('--epochs', type=int, required=True,
                        help="total number of epochs")
    parser.add_argument('--batchsize', type=int, default=64,
                        help="training batch size (default: %(default)d)")
    parser.add_argument('--data', nargs='+',
                        required=True,
                        help=("MNIST-like data directory(ies); if more than one is given, "
                              "data will be randomly interleaved"))
    parser.add_argument('--weights', type=float, nargs='+', required=False,
                        help=("weights for randomly interleaving data directories; must be "
                              "positive of the same length as the list of directories"))
    parser.add_argument('--latent', type=int, required=True,
                        help="VAE latent dimension")
    parser.add_argument('--mvvae', default=False, action='store_true',
                        help="Whether to predict full covariance")

    # argv = None
    # argv = ("--epochs 20 --data /vol/biomedic/users/dc315/mnist/plain "
    #         "--checkpoint /data/deepgen/checkpoints "
    #         "--latent 10 --save").split()
    args = parser.parse_args()
    print(args)

    use_cuda = (args.device == 'cuda') if args.device else torch.cuda.is_available()

    # train_batch_size = 64
    # data_dirs = ["/vol/biomedic/users/dc315/mnist/plain"]

    # latent_dim = 64
    # save = False
    # resume = True
    # plot = True
    # num_epochs = 10
    # ckpt_root = "/data/morphomnist/checkpoints"

    main(use_cuda=use_cuda, data_dirs=args.data, weights=args.weights, ckpt_root=args.checkpoint,
         latent_dim=args.latent, num_epochs=args.epochs, batch_size=args.batchsize,
         save=args.save, resume=args.resume, plot=True, mvvae=args.mvvae)
