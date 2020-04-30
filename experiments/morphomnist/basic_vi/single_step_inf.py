import torch
import pyro

from pyro.nn import pyro_method

from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
from pyro.distributions.score_parts import ScoreParts

from experiments.morphomnist.basic_vi.basic_covariate_pgm_vae import Experiment, CovariatePGMVAE
from pyro.distributions.util import is_identically_zero
import warnings
import experiments.morphomnist.multi_step_inf as msi


class SupervisedTraceELBO(msi.SupervisedTraceELBO):
    def _loss_and_surrogate_loss(self, model, guide, args, kwargs):
        # copied from https://github.com/pyro-ppl/pyro/blob/05d27a14ccc17a1502ab46644fee9540174c7c12/pyro/infer/tracegraph_elbo.py#L264-L278
        assert 'data' in kwargs.keys()
        data = kwargs.pop('data')

        _ = super(msi.SupervisedTraceELBO, self)._loss_and_surrogate_loss(model, guide, args, kwargs)

        cond_model = pyro.condition(model, data=data)
        cond_guide = pyro.condition(guide, data=data)
        loss, surrogate_loss = super(msi.SupervisedTraceELBO, self)._loss_and_surrogate_loss(cond_model, cond_guide, args, kwargs)

        return loss, surrogate_loss

    def loss(self, model, guide, *args, **kwargs):
        assert 'data' in kwargs.keys()
        data = kwargs.pop('data')

        _ = super(msi.SupervisedTraceELBO, self).loss(model, guide, *args, **kwargs)

        cond_model = pyro.condition(model, data=data)
        cond_guide = pyro.condition(guide, data=data)
        loss = super(msi.SupervisedTraceELBO, self).loss(cond_model, cond_guide, *args, **kwargs)

        return loss


class SingleStepExperiment(msi.MultiStepExperiment):
    def __init__(self, hparams):
        pyro_model = msi.MultiStepCovariatePGMVAE(hidden_dim=hparams.hidden_dim, latent_dim=hparams.latent_dim)

        super().__init__(hparams, pyro_model)

        loss_class = SupervisedTraceELBO()
        self._build_svi(loss=loss_class)
        self.svi.loss_class = loss_class


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(logger=True, checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    experiment_group.add_argument('--latent_dim', default=10, type=int, help="latent dimension of model (defaults to 10)")
    experiment_group.add_argument('--hidden_dim', default=100, type=int, help="hidden dimension of model (defaults to 100)")
    experiment_group.add_argument('--lr', default=1e-4, type=float, help="latent dimension of model (defaults to 1e-4)")
    experiment_group.add_argument('--validate', default=False, action='store_true', help="latent dimension of model (defaults to False)")
    experiment_group.add_argument('--train_batch_size', default=256, type=int, help="train batch size (defaults to 256)")
    experiment_group.add_argument('--test_batch_size', default=256, type=int, help="test batch size (defaults to 256)")

    args = parser.parse_args()

    # TODO: push to lightning
    args.gradient_clip_val = float(args.gradient_clip_val)

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)

    lightning_args = groups['lightning_options']
    hparams = groups['experiment']

    trainer = Trainer.from_argparse_args(lightning_args)

    experiment = SingleStepExperiment(hparams)

    trainer.fit(experiment)
