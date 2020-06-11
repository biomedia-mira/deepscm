import torch
import pyro

from pyro.nn import pyro_method

from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
from pyro.distributions.score_parts import ScoreParts

from experiments.morphomnist.basic_vi.basic_covariate_pgm_vae import Experiment, CovariatePGMVAE
from pyro.distributions.util import is_identically_zero
import warnings


class SupervisedTraceELBO(TraceGraph_ELBO):
    # just do one step of regular elbo
    # condition on data (both guide and model) and change https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/tracegraph_elbo.py#L162-L169 from  - to +
    # ^ or simply go through traces and multiply by -1 if node is observed....!!
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': {True: None, False: None}, 'guide': {True: None, False: None}}

    def _loss_and_surrogate_loss(self, model, guide, args, kwargs):
        # copied from https://github.com/pyro-ppl/pyro/blob/05d27a14ccc17a1502ab46644fee9540174c7c12/pyro/infer/tracegraph_elbo.py#L264-L278
        assert 'data' in kwargs.keys()
        data = kwargs.pop('data')

        loss, surrogate_loss = super()._loss_and_surrogate_loss(model, guide, args, kwargs)

        cond_model = pyro.condition(model, data=data)
        cond_guide = pyro.condition(guide, data=data)
        loss_, surrogate_loss_ = super()._loss_and_surrogate_loss(cond_model, cond_guide, args, kwargs)

        loss += loss_
        loss += surrogate_loss_

        return loss, surrogate_loss

    def loss(self, model, guide, *args, **kwargs):
        assert 'data' in kwargs.keys()
        data = kwargs.pop('data')

        loss = super().loss(model, guide, *args, **kwargs)

        cond_model = pyro.condition(model, data=data)
        cond_guide = pyro.condition(guide, data=data)
        loss += super().loss(cond_model, cond_guide, *args, **kwargs)

        return loss

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        is_cond = False
        # this will only work for reparam
        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                if is_identically_zero(site["score_parts"].entropy_term):
                    warnings.warn("Trying to use SupervisedTraceELBO for non-reparam node")
                score_parts = site["score_parts"]
                site["score_parts"] = ScoreParts(-1 * score_parts.log_prob, score_parts.score_function, -1 * score_parts.entropy_term)
                site["log_prob_sum"] *= -1
                is_cond = True

        self.trace_storage['model'][is_cond] = model_trace
        self.trace_storage['guide'][is_cond] = model_trace

        return model_trace, guide_trace


class MultiStepCovariatePGMVAE(CovariatePGMVAE):
    @pyro_method
    def svi_model(self, x, thickness, slant):
        with pyro.plate('observations', x.shape[0]):
            pyro.condition(self.model, data={'x': x})()


class MultiStepExperiment(Experiment):
    def __init__(self, hparams):
        pyro_model = MultiStepCovariatePGMVAE(hidden_dim=hparams.hidden_dim, latent_dim=hparams.latent_dim)

        super().__init__(hparams, pyro_model)

        loss_class = SupervisedTraceELBO()
        self._build_svi(loss=loss_class)
        self.svi.loss_class = loss_class

    def get_trace_metrics(self, batch):
        metrics = {}
        x, thickness, slant = self.prep_batch(batch)

        cond_model = self.svi.loss_class.trace_storage['model'][True]
        cond_guide = self.svi.loss_class.trace_storage['guide'][True]

        prior_model = self.svi.loss_class.trace_storage['model'][False]
        prior_guide = self.svi.loss_class.trace_storage['guide'][False]

        metrics['x_cond_mse'] = ((cond_model.nodes['x']['fn'].base_dist.mean - x)**2).sum((1, 2, 3)).mean().item()
        metrics['x_prior_mse'] = ((prior_model.nodes['x']['fn'].base_dist.mean - x)**2).sum((1, 2, 3)).mean().item()

        metrics['thickness_cond_mae'] = torch.abs(cond_guide.nodes['thickness']['fn'].base_dist.mean - thickness).mean().item()
        metrics['thickness_prior_mae'] = torch.abs(prior_guide.nodes['thickness']['fn'].base_dist.mean - thickness).mean().item()

        metrics['slant_cond_mae'] = torch.abs(cond_guide.nodes['slant']['fn'].base_dist.mean - slant).mean().item()
        metrics['slant_prior_mae'] = torch.abs(prior_guide.nodes['slant']['fn'].base_dist.mean - slant).mean().item()

        metrics['z_loss'] = prior_model.nodes['z']['log_prob_sum'] - prior_guide.nodes['z']['log_prob_sum']

        return metrics

    def training_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        if self.hparams.validate:
            print('Validation:')
            self.print_trace_updates(batch)

        loss = self.svi.step(x, thickness, slant, data={'thickness': thickness, 'slant': slant})

        metrics = self.get_trace_metrics(batch)

        tensorboard_logs = {('train/' + k): v for k, v in metrics.items()}
        tensorboard_logs['train/loss'] = loss

        return {'loss': torch.Tensor([loss]), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(x, thickness, slant, data={'thickness': thickness, 'slant': slant})

        metrics = self.get_trace_metrics(batch)

        return {'loss': loss, **metrics}

    def test_step(self, batch, batch_idx):
        x, thickness, slant = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(x, thickness, slant, data={'thickness': thickness, 'slant': slant})

        metrics = self.get_trace_metrics(batch)

        return {'loss': loss, **metrics}


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

    experiment = MultiStepExperiment(hparams)

    trainer.fit(experiment)
