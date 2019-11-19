import torch
from torch.distributions import Independent, Normal, kl_divergence


def std_normal_kl(mean, logvar):
    return -.5 * (1. + logvar - mean ** 2 - logvar.exp()).sum()


def main():
    device = torch.device('cpu')
    num = 10
    dim = 5
    mean = torch.zeros([num, dim], device=device) + 1.
    logvar = torch.zeros([num, dim], device=device) + 1.
    mean.requires_grad = True
    logvar.requires_grad = True
    std = torch.exp(.5*logvar)
    print(mean, std)
    prior = Independent(Normal(torch.zeros(dim, device=device), torch.ones(dim, device=device)), 1)
    dist2 = Independent(Normal(mean, std), 1)
    print(prior.batch_shape, prior.event_shape)
    print(dist2.batch_shape, dist2.event_shape)
    def print_normal_params(d):
        base = d.base_dist
        print(f"loc = {base.loc}, scale = {base.scale}")
    print_normal_params(prior)
    print(prior)
    print_normal_params(dist2)
    print(dist2)
    with torch.no_grad():
        print(std_normal_kl(mean, logvar))
        print(kl_divergence(prior, dist2))
        print(kl_divergence(dist2, prior))
    kl_divergence(dist2, prior).mean().backward()
    print("Gradients:")
    print(mean.grad)
    print(logvar.grad)
    dist2.rsample()


if __name__ == '__main__':
    main()
