"""Adapted from: https://github.com/mattjj/svae/blob/master/svae/models/gmm.py"""
import numpy as np
import numpy.random as npr

from external.san import niw_san, gaussian_san


### inference functions for the SVAE interface
def run_inference(prior_natparam, global_natparam, nn_potentials, num_samples):
    _, stats, local_natparam, local_kl = local_meanfield(global_natparam, nn_potentials)
    samples = gaussian_san.natural_sample(local_natparam[1], num_samples)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return samples, unbox(stats), global_kl, local_kl


def make_encoder_decoder(recognize, decode):
    def encode_mean(data, natparam, recogn_params):
        nn_potentials = recognize(recogn_params, data)
        (_, gaussian_stats), _, _, _ = local_meanfield(natparam, nn_potentials)
        _, Ex, _, _ = gaussian_san.unpack_dense(gaussian_stats)
        return Ex

    def decode_mean(z, phi):
        mu, _ = decode(z, phi)
        return mu.mean(axis=1)

    return encode_mean, decode_mean


### GMM prior on \theta = (\pi, {(\mu_k, \Sigma_k)}_{k=1}^K)
def init_pgm_param(K, N, alpha, niw_conc=10., random_scale=0.):
    def init_niw_natparam(N):
        nu, S, m, kappa = N+niw_conc, (N+niw_conc)*np.eye(N), np.zeros(N), niw_conc
        m = m + random_scale * npr.randn(*m.shape)
        return niw_san.standard_to_natural(S, m, kappa, nu)

    dirichlet_natparam = alpha * (npr.rand(K) if random_scale else np.ones(K))
    niw_natparam = np.stack([init_niw_natparam(N) for _ in range(K)])

    return dirichlet_natparam, niw_natparam


def prior_logZ(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    return dirichlet.logZ(dirichlet_natparam) + niw_san.logZ(niw_natparams)


def prior_expectedstats(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    dirichlet_expectedstats = dirichlet.expectedstats(dirichlet_natparam)
    niw_expectedstats = niw_san.expectedstats(niw_natparams)
    return dirichlet_expectedstats, niw_expectedstats


def prior_kl(global_natparam, prior_natparam):
    expected_stats = flat(prior_expectedstats(global_natparam))
    natparam_difference = flat(global_natparam) - flat(prior_natparam)
    logZ_difference = prior_logZ(global_natparam) - prior_logZ(prior_natparam)
    return np.dot(natparam_difference, expected_stats) - logZ_difference


### GMM mean field functions
def local_meanfield(global_natparam, node_potentials):
    dirichlet_natparam, niw_natparams = global_natparam
    node_potentials = gaussian_san.pack_dense(*node_potentials)

    # compute expected global parameters using current global factors
    label_global = dirichlet.expectedstats(dirichlet_natparam)
    gaussian_globals = niw_san.expectedstats(niw_natparams)

    # compute mean field fixed point using unboxed node_potentials
    label_stats = meanfield_fixed_point(label_global, gaussian_globals, getval(node_potentials))

    # compute values that depend directly on boxed node_potentials at optimum
    gaussian_natparam, gaussian_stats, gaussian_kl = \
        gaussian_meanfield(gaussian_globals, node_potentials, label_stats)
    label_natparam, label_stats, label_kl = \
        label_meanfield(label_global, gaussian_globals, gaussian_stats)

    # collect sufficient statistics for gmm prior (sum across conditional iid)
    dirichlet_stats = np.sum(label_stats, 0)
    niw_stats = np.tensordot(label_stats, gaussian_stats, [0, 0])

    local_stats = label_stats, gaussian_stats
    prior_stats = dirichlet_stats, niw_stats
    natparam = label_natparam, gaussian_natparam
    kl = label_kl + gaussian_kl

    return local_stats, prior_stats, natparam, kl


def meanfield_fixed_point(label_global, gaussian_globals, node_potentials, tol=1e-3, max_iter=100):
    kl = np.inf
    label_stats = initialize_meanfield(label_global, node_potentials)
    for i in range(max_iter):
        gaussian_natparam, gaussian_stats, gaussian_kl = \
            gaussian_meanfield(gaussian_globals, node_potentials, label_stats)
        label_natparam, label_stats, label_kl = \
            label_meanfield(label_global, gaussian_globals, gaussian_stats)

        # recompute gaussian_kl linear term with new label_stats b/c labels were updated
        gaussian_global_potentials = np.tensordot(label_stats, gaussian_globals, [1, 0])
        linear_difference = gaussian_natparam - gaussian_global_potentials - node_potentials
        gaussian_kl = gaussian_kl + np.tensordot(linear_difference, gaussian_stats, 3)

        kl, prev_kl = label_kl + gaussian_kl, kl
        if abs(kl - prev_kl) < tol:
            break
    else:
        print('iteration limit reached')

    return label_stats


def gaussian_meanfield(gaussian_globals, node_potentials, label_stats):
    global_potentials = np.tensordot(label_stats, gaussian_globals, [1, 0])
    natparam = node_potentials + global_potentials
    stats = gaussian_san.expectedstats(natparam)
    kl = np.tensordot(node_potentials, stats, 3) - gaussian_san.logZ(natparam)
    return natparam, stats, kl


def label_meanfield(label_global, gaussian_globals, gaussian_stats):
    node_potentials = np.tensordot(gaussian_stats, gaussian_globals, [[1,2], [1,2]])
    natparam = node_potentials + label_global
    stats = categorical.expectedstats(natparam)
    kl = np.tensordot(stats, node_potentials) - categorical.logZ(natparam)
    return natparam, stats, kl


def initialize_meanfield(label_global, node_potentials):
    T, K = node_potentials.shape[0], label_global.shape[0]
    return normalize(npr.rand(T, K))
