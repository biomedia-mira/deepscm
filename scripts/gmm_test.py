import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F

from models.mixture import MultivariateNormalMixture, NaturalMultivariateNormalMixture
from distributions.natural_mvn import NaturalMultivariateNormal


class LearnableGMM(nn.Module):
    def __init__(self, n_components, n_dimensions):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(n_components))
        self.means = nn.Parameter(torch.randn(n_components, n_dimensions))
        self.scales = nn.Parameter(torch.randn(n_components, n_dimensions, n_dimensions))

        self.distribution = self._get_distribution()

    def _get_distribution(self):
        covariances = self.scales @ self.scales.transpose(1, 2)
        mixing = td.Categorical(logits=self.logits)
        components = td.MultivariateNormal(self.means, covariances)
        return MultivariateNormalMixture(mixing, components)

    def forward(self, data):
        return self._get_distribution().log_prob(data)


class LearnableNGMM(nn.Module):
    def __init__(self, n_components, n_dimensions):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(n_components))
        self.locs = nn.Parameter(torch.randn(n_components, n_dimensions))
        self.scales = nn.Parameter(torch.randn(n_components, n_dimensions, n_dimensions))
        self.diag_coeff = nn.Parameter(torch.randn(n_components))
        # self._eye = torch.eye(n_dimensions).unsqueeze(0).expand(n_components, -1, -1)
        self.register_buffer('_eye', torch.eye(n_dimensions).unsqueeze(0).expand(n_components, -1, -1))

        self.distribution = self._get_distribution()

    def _get_distribution(self):
        mixing = td.Categorical(logits=self.logits)
        # diag = F.softplus(self.diag_coeff)[:, None, None] * self._eye
        diag = 1e-6 * self._eye
        precisions = self.scales @ self.scales.transpose(-1, -2) + diag
        # idx = torch.arange(precisions.shape[-1], device=precisions.device)
        # precisions[:, idx, idx] += self.diag_coeff.exp()[:, None]
        components = NaturalMultivariateNormal(self.locs, -precisions)
        return NaturalMultivariateNormalMixture(mixing, components)

    def forward(self, data):
        return self._get_distribution().log_prob(data)


def eval_grid(xx, yy, fcn):
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return fcn(xy).reshape_as(xx)


def plot_gmm(xx, yy, gmm: LearnableGMM, data=None):
    xlim = [xx.min(), xx.max()]
    ylim = [yy.min(), yy.max()]
    with torch.no_grad():
        zz = eval_grid(xx, yy, gmm)
    plt.imshow(zz.exp().T, interpolation='bilinear', origin='lower', extent=[*xlim, *ylim])
    # plt.contourf(xx, yy, zz.exp(), cmap='viridis')
    if data is not None:
        plt.scatter(*data.T, c='k', s=4)
    plt.xlim(xlim)
    plt.ylim(ylim)


if __name__ == '__main__':
    K = 3
    gmm = LearnableGMM(K, 2)
    X = gmm.distribution.sample([1000])

    import matplotlib.pyplot as plt

    xlim, ylim = zip(X.min(0)[0], X.max(0)[0])
    x = torch.linspace(*xlim, 200)
    y = torch.linspace(*ylim, 200)
    xx, yy = torch.meshgrid(x, y)

    plot_gmm(xx, yy, gmm, X)
    plt.title("Source GMM")
    plt.show()

    new_gmm = LearnableNGMM(K, 2)
    opt = torch.optim.Adam(list(new_gmm.parameters()), lr=1e-1)

    plot_gmm(xx, yy, new_gmm, X)
    plt.title("New GMM")
    plt.show()

    X = X.to('cuda')
    new_gmm = new_gmm.to('cuda')

    max_iter = 500
    for iter in range(max_iter):
        loss = -new_gmm(X).sum()
        print("loss =", loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    X = X.to('cpu')
    new_gmm = new_gmm.to('cpu')

    plot_gmm(xx, yy, new_gmm, X)
    plt.title("Fitted GMM")
    plt.show()
