import os
import re
import torch
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(model: torch.nn.Module, ckpt_dir: str, epoch: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    class_name = model.__class__.__name__
    ckpt_path = os.path.join(ckpt_dir, f"{class_name}-{epoch:03d}.pth")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def load_checkpoint(model: torch.nn.Module, path: str, epoch=None) -> int:
    pattern = re.compile(r'(?<=-)\d+(?=.pth$)')

    def get_epoch(f):
        # return int(f.split('.')[0].split('-')[-1])
        return int(pattern.findall(f)[0])

    filenames = [filename for filename in os.listdir(path) if filename.endswith('.pth')]
    print(f"Available checkpoints in {path}:")
    for f in filenames:
        print(f"- {f}")
    if epoch is None:
        ckpt_filename = max(filenames, key=get_epoch)
        epoch = get_epoch(ckpt_filename)
    else:
        try:
            ckpt_filename = next(f for f in filenames if get_epoch(f) == epoch)
        except StopIteration:
            raise ValueError(f"No checkpoint for epoch {epoch} in {path}")
    ckpt_path = os.path.join(path, ckpt_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Invalid checkpoint directory: {path}")
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    return epoch


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
