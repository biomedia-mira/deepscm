import numpy as np
import scipy.stats as ss
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import util
from distributions.wishart import Wishart


def plot_cov(cov, ax, color):
    l, v = np.linalg.eig(cov)
    ax.add_patch(Ellipse(xy=(0., 0.),
                         width=2. * np.sqrt(l[0]),
                         height=2. * np.sqrt(l[1]),
                         angle=np.rad2deg(np.arctan2(*v[:, 0][::-1])),
                         ec=color, fill=False, lw=.1))


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    N, D = 10000, 3
    a = 2.
    # B = np.array([[1., 0.], [0., 1.]])
    B = 2. * (+.2*np.ones((D, D)) + np.eye(D))
    size = None

    ss_dist = ss.wishart(2. * a, np.linalg.inv(2. * B))
    dist = Wishart(torch.as_tensor(a, dtype=torch.float64),
                   torch.as_tensor(B, dtype=torch.float64))
    W1 = ss_dist.rvs(N)
    print(W1.shape)
    W2 = dist.sample((N,)).numpy()
    print(W2.shape)

    if D == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='row', sharey='row')
        for prec in W1:
            plot_cov(np.linalg.inv(prec), ax1, 'k')
        for prec in W2:
            plot_cov(np.linalg.inv(prec), ax2, 'k')
        lim = 2
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_aspect(1.)
        ax2.set_aspect(1.)
        plt.show()

    logdet_W1 = util.posdef_logdet(torch.as_tensor(W1))[0]
    logdet_W2 = util.posdef_logdet(torch.as_tensor(W2))[0]
    print(f"log|W1| = {logdet_W1.mean(): .3f} \u00b1 {logdet_W1.std():.3f}")
    print(f"log|W2| = {logdet_W2.mean(): .3f} \u00b1 {logdet_W2.std():.3f}")
    print(f"E[log|W|] = {dist.expected_logdet(): .3f}, "
          f"\u221aVar[log|W|] = {dist.variance_logdet().sqrt():.3f}")
    print()
    np.set_printoptions(precision=3)
    print(f"<W1> = \n{W1.mean(0)}")
    print(f"<W2> = \n{W2.mean(0)}")
    print(f"E[W] = \n{dist.mean.numpy()}")
    print(f"Ê[W] = \n{a * np.linalg.inv(B)}")
    print()
    print(f"H[W1] = {ss_dist.entropy(): .3f}")
    print(f"H[W2] = {dist.entropy(): .3f}")
    logpdf_W1 = np.asarray([ss_dist.logpdf(w) for w in W1])
    logpdf_W2 = dist.log_prob(torch.as_tensor(W2)).numpy()
    def sem(x):
        return np.std(np.asarray(x)) / np.sqrt(len(x))
    def conf_int_mean(x, alpha=.05):
        mean = np.mean(np.asarray(x))
        std_err = sem(x)
        crit = ss.t.isf(.5 * alpha, df=len(x) - 1)
        return mean - crit * std_err, mean + crit * std_err
    def format_ci(x, alpha=.05):
        lo, hi = conf_int_mean(x, alpha)
        return f"[{lo: .3f}, {hi: .3f}] @ {100.*(1.-alpha):g}%"
    print(f"^H[W1] \u2208 {format_ci(-logpdf_W1)}")
    print(f"^H[W2] \u2208 {format_ci(-logpdf_W2)}")
