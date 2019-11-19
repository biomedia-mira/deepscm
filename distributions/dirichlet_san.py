import torch


def expected_log_pi(alpha_standard):
    return torch.digamma(alpha_standard), - torch.digamma(alpha_standard.sum(-1, keep_dims=True))


def standard_to_natural(alpha):
    return alpha - 1.


def natural_to_standard(alpha_nat):
    return alpha_nat + 1.
