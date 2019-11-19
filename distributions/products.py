import torch
import torch.distributions as td

from util import posdef_inverse


def categorical_categorical(p: td.Categorical, q: td.Categorical):
    return td.Categorical(logits=p.logits + q.logits)


def mvn_mvn(p: td.MultivariateNormal, q: td.MultivariateNormal):
    pass


class ConjugateExponentialFamily:
    def a(self):
        raise NotImplementedError


def condition_mvn(mvn: td.MultivariateNormal, y_dims, x_dims):
    """Conditional distribution of Y|X"""
    x_dims = torch.as_tensor(x_dims)#.unsqueeze(0)
    y_dims = torch.as_tensor(y_dims)#.unsqueeze(0)
    # TODO: Correctly handle single dimensions (int)
    mx = mvn.loc[..., x_dims]  # (..., Dx)
    my = mvn.loc[..., y_dims]  # (..., Dy)
    Sxx = mvn.covariance_matrix[..., x_dims.unsqueeze(-1), x_dims.unsqueeze(-2)]  # (..., Dx, Dx)
    Sxy = mvn.covariance_matrix[..., x_dims.unsqueeze(-1), y_dims.unsqueeze(-2)]  # (..., Dx, Dy)
    Syy = mvn.covariance_matrix[..., y_dims.unsqueeze(-1), y_dims.unsqueeze(-2)]  # (..., Dy, Dy)
    Syx = Sxy.transpose(-2, -1)  # (..., Dy, Dx)
    if x_dims.dim() > 0:
        Syx_iSxx = Syx @ posdef_inverse(Sxx)[0]  # (..., Dy, Dx)
    else:
        Syx_iSxx = Syx / Sxx  # (..., Dy, Dx)
    Syy_x = Syy - Syx_iSxx @ Sxy  # (..., Dy, Dy)
    def condition(x):
        my_x = my + Syx_iSxx @ (x - mx)  # (..., Dy)
        if y_dims.dim() > 0:
            return td.MultivariateNormal(my_x, covariance_matrix=Syy_x)
        else:
            return td.Normal(my_x.squeeze(-1), torch.sqrt(Syy_x).squeeze(-1))
    return condition


class ClosedExponentialFamily(td.ExponentialFamily):
    def product(self, other: 'ClosedExponentialFamily') -> 'ClosedExponentialFamily':
        pass


if __name__ == '__main__':
    D = 5
    mean = torch.zeros(D)
    cov = torch.eye(D)
    mvn = td.MultivariateNormal(mean, cov)
    print(condition_mvn(mvn, (1,), (0,))(mean[0]))
    print(condition_mvn(mvn, (1,), 0)(mean[0]))
    print(condition_mvn(mvn, 1, (0,))(mean[0]))
    print(condition_mvn(mvn, [1, 2], [0])(mean[[0]]))
