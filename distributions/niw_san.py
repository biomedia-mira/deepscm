import torch


def expected_values(niw_standard_params):
    beta, m, C, v = niw_standard_params
    exp_m = m
    C_inv = torch.inverse(C)
    C_inv_sym = .5 * (C_inv + C_inv.T)
    exp_C = torch.inverse(v * C_inv_sym)
    return exp_m, exp_C


def standard_to_natural(beta: torch.Tensor, m: torch.Tensor, C: torch.Tensor, v: torch.Tensor):
    K, D = m.shape

    b = beta.unsqueeze(-1) * m
    A = C + b.unsqueeze(-1) * m.unsqueeze(-2)
    v_hat = v + D + 2

    return A, b, beta, v_hat


def natural_to_standard(A: torch.Tensor, b: torch.Tensor, beta: torch.Tensor, v_hat: torch.Tensor):
    m = b / beta.expand(-1)

    K, D = m.shape

    C = A - b.unsqueeze(-1) * m.unsqueeze(-2)
    v = v_hat - D - 2
    return beta, m, C, v
