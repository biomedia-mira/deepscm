import pyro
import torch
from pyro.distributions import Categorical, Gamma, Normal
from torch.distributions import constraints

from models.pyro_util import gather, param_scope, log_likelihood


@param_scope('slant')
def slant_dist():
    loc = pyro.param('loc', torch.ones(1))
    scale = pyro.param('scale', torch.ones(1), constraints.positive)
    return Normal(loc, scale)


@param_scope('thickness')
def thickness_dist():
    conc = pyro.param('conc', torch.ones(1), constraints.positive)
    rate = pyro.param('rate', torch.ones(1), constraints.positive)
    return Gamma(conc, rate)


@param_scope('label')
def label_dist():
    probs = pyro.param('probs', torch.ones(10) / 10., constraints.simplex)
    return Categorical(probs)


@param_scope('width')
def width_dist(label):
    locs = pyro.param('locs', torch.randn(10))
    scales = pyro.param('scales', torch.ones(10), constraints.positive)
    loc = gather(locs, label, -1)
    scale = gather(scales, label, -1)
    return Normal(loc, scale)


@param_scope('length')
def length_dist(width, label):
    biases = pyro.param('biases', torch.zeros(10))
    weights = pyro.param('weights', torch.zeros(10))
    scale = pyro.param('scale', torch.ones(1), constraints.positive)
    loc = gather(biases, label, -1) + gather(weights, label, -1) * width
    return Normal(loc, scale)


@param_scope('area')
def area_dist(length, thickness, label):
    weights = pyro.param('weights', torch.ones(10))
    scale = pyro.param('scale', torch.ones(1), constraints.positive)
    loc = gather(weights, label, -1) * (length * thickness)
    return Normal(loc, scale)


def model(n_samples=None):
    with pyro.plate('observations', n_samples):
        slant = pyro.sample('slant', slant_dist())
        thickness = pyro.sample('thickness', thickness_dist())
        label = pyro.sample('label', label_dist())
        width = pyro.sample('width', width_dist(label))
        length = pyro.sample('length', length_dist(width, label))
        area = pyro.sample('area', area_dist(length, thickness, label))

    return slant, thickness, label, width, length, area


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    from morphomnist import io

    data_dir = "/vol/biomedic/users/dc315/mnist/original/"
    labels = pd.Series(io.load_idx(data_dir + "t10k-labels-idx1-ubyte.gz"))
    metrics = pd.read_csv(data_dir + "t10k-morpho.csv", index_col='index')
    metrics['label'] = labels.astype(np.long)

    columns = ['slant', 'thickness', 'length', 'width', 'area', 'length', 'label']
    metrics_tensors = {name: torch.as_tensor(metrics[name]) for name in columns}

    print(metrics_tensors)

    from pyro.poutine import condition, trace
    from torch.optim import Adam

    # Model conditioned on the observations
    morpho_model = condition(model, data=metrics_tensors)

    # Trace a dry run to initialise parameters
    trace(morpho_model, param_only=True).get_trace()
    params = [param.unconstrained() for param in pyro.get_param_store().values()]
    opt = Adam(params, 1e-1)

    losses = []
    for i in range(1000):
        # morpho_trace = trace(morpho_model).get_trace(**metrics_tensors)
        loss = -log_likelihood(morpho_model)
        losses.append(loss.item())
        print(i, loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
