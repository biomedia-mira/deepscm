import torch
from torch import nn


def make_gradfun(run_inference, recognize, loglike, pgm_prior, data,
                 batch_size, num_samples, natgrad_scale=1.):
    """Adapted from: https://github.com/mattjj/svae/blob/master/svae/svae.py"""
    _, unflat = flatten(pgm_prior)
    num_datapoints = data.shape[0]

    def mc_elbo(pgm_params, loglike_params, recogn_params, i):
        nn_potentials = recognize(recogn_params, get_batch(i))
        samples, stats, global_kl, local_kl = \
            run_inference(pgm_prior, pgm_params, nn_potentials, num_samples)
        return (num_batches * loglike(loglike_params, samples, get_batch(i))
                - global_kl - num_batches * local_kl) / num_datapoints

    def gradfun(params, i):
        pgm_params, loglike_params, recogn_params = params
        objective = lambda loglike_params, recogn_params: \
            -mc_elbo(pgm_params, loglike_params, recogn_params, i)
        val, (loglike_grad, recogn_grad) = vgrad(objective)((loglike_params, recogn_params))

        # this expression for pgm_natgrad drops a term that can be computed using
        # the function autograd.misc.fixed_points.fixed_point
        pgm_natgrad = -natgrad_scale / num_datapoints * \
            (flat(pgm_prior) + num_batches*flat(stats) - flat(pgm_params))
        grad = unflat(pgm_natgrad), loglike_grad, recogn_grad

        return grad

    return gradfun
