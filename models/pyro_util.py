import pyro
from pyro import poutine
from pyro.distributions import util


def gather(x, index, dim):
    new_shape = index.shape + tuple([1] * x.dim())
    collapsed_shape = list(x.shape)
    del collapsed_shape[dim]
    out_shape = index.shape + tuple(collapsed_shape)
    return util.gather(x, index.reshape(new_shape), dim).reshape(out_shape)


def log_likelihood(model, *args, **kwargs):
    return poutine.trace(model).get_trace(*args, **kwargs).log_prob_sum()


class param_scope:
    def __init__(self, prefix: str, sep: str = '.'):
        self.prefix = prefix
        self.sep = sep
        self._orig_pyro_param = None

    def __enter__(self):
        self._orig_pyro_param = pyro.param

        def new_param(name, *args, **kwargs):
            new_name = self.sep.join([self.prefix, name])
            return self._orig_pyro_param(new_name, *args, **kwargs)

        pyro.param = new_param

    def __exit__(self, type, value, traceback):
        pyro.param = self._orig_pyro_param
        self._orig_pyro_param = None

    def __call__(self, fn):
        def decorator(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return decorator
