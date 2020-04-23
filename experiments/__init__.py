import pyro
import pytorch_lightning as pl
import torch
from pyro import poutine

_PARAM_STORE_KEY = 'pyro_param_store'


def get_traces(model, guide, *args, **kwargs):
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_replay = poutine.replay(model, trace=guide_trace)
    model_trace = poutine.trace(model_replay).get_trace(*args, **kwargs)
    return model_trace


def _clone_param_store():
    return pyro.get_param_store()._params.copy()


def _compare_param_dicts(params1: dict, params2: dict):
    params1 = params1.copy()
    params2 = params2.copy()
    for name1, value1 in params1.items():
        if name1 in params2:
            value2 = params2.pop(name1)
            equal = torch.allclose(value1, value2)
            print(f"{name1} in both; equal: {equal}")
        else:
            print(f"{name1} missing from second dict")
    for name2 in params2:
        assert name2 not in params1
        print(f"{name2} missing from first dict")


class PyroExperiment(pl.LightningModule):
    debug_pyro_checkpoint = False

    def _get_parameters(self, *args, **kwargs):
        # Adapted from pyro.infer.svi.step()
        with poutine.trace(param_only=True) as param_capture:
            self.elbo.loss(self.model, self.guide, *args, **kwargs)
        return set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())

    def forward(self, *args, **kwargs):
        pass  # Must implement abstract forward() method from LightningModule

    def backward(self, *args, **kwargs):
        pass  # No loss to backpropagate since we're using Pyro's optimisation machinery

    def state_dict(self, *args, **kwargs):
        return {}  # Avoid serialising Pyro params twice; delegated to on_save_checkpoint()

    def load_state_dict(self, *args, **kwargs):
        pass  # Avoid de-serialising Pyro params twice; delegated to on_load_checkpoint()

    def on_save_checkpoint(self, checkpoint):
        checkpoint[_PARAM_STORE_KEY] = pyro.get_param_store().get_state()

        if PyroExperiment.debug_pyro_checkpoint:
            assert len(checkpoint['state_dict']) == 0, "Non-empty 'state_dict' in checkpoint"
            PyroExperiment._saved_pyro_params = _clone_param_store()

    def on_load_checkpoint(self, checkpoint):
        if PyroExperiment.debug_pyro_checkpoint:
            existing_pyro_params = _clone_param_store()
            print(">>> Saved vs. existing:")
            _compare_param_dicts(PyroExperiment._saved_pyro_params, existing_pyro_params)

        pyro.clear_param_store()

        if PyroExperiment.debug_pyro_checkpoint:
            cleared_pyro_params = _clone_param_store()
            assert len(cleared_pyro_params) == 0, "Non-empty param store after clearing"

        pyro.get_param_store().set_state(checkpoint[_PARAM_STORE_KEY])

        if PyroExperiment.debug_pyro_checkpoint:
            loaded_pyro_params = _clone_param_store()
            print("\n>>> Saved vs. loaded:")
            _compare_param_dicts(PyroExperiment._saved_pyro_params, loaded_pyro_params)
