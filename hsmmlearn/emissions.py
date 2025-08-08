from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp

from .utils import NonParametricDistribution


class AbstractEmissions(object):
    """ Base class for emissions distributions.

    To create a HSMM with a custom emission distribution, write a derived
    class that implements some (or all) of the abstract methods. If you
    don't need all of the HSMM functionality, you can get by with implementing
    only some of the methods.

    """

    __meta__ = ABCMeta

    @abstractmethod
    def sample_for_state(self, state, size=None):
        """ Return a random emission given a state.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.sample`.

        Parameters
        ----------
        state : int
            The internal state.
        size : int
            The number of random samples to generate.

        Returns
        -------
        observations : numpy.ndarray, shape=(size, )
            Random emissions.

        """
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, obs):
        """ Compute the likelihood of a sequence of observations.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit` and
        :py:func:`hsmmlearn.hsmm.HSMMModel.decode`.

        Parameters
        ----------
        obs : numpy.ndarray, shape=(n_obs, )
            Sequence of observations.

        Returns
        -------
        likelihood : float

        """
        raise NotImplementedError

    @abstractmethod
    def reestimate(self, gamma, obs):
        r""" Estimate the distribution parameters given sequences of
        smoothed probabilities and observations.

        The parameter ``gamma`` is an array of smoothed probabilities,
        with the entry ``gamma[s, i]`` giving the probability of
        finding the system in state ``s`` given *all* of the observations
        up to index ``i``:

        .. math::

            \gamma_{s, i} = P(s | o_1, \ldots, o_i ).

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit`.

        Parameters
        ----------
        gamma : numpy.ndarray, shape=(n_obs, )
            Smoothed probabilities.
        obs : numpy.ndarray, shape=(n_obs, )
            Observations.

        """
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        """ Make a copy of this object.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit` to
        make a copy of the emissions object before modifying it.

        """
        raise NotImplementedError


class MultinomialEmissions(AbstractEmissions):
    """ An emissions class for multinomial emissions.

    This emissions class models the case where the emissions are categorical
    variables, assuming values from 0 to some value k, and the probability
    of observing an emission given a state is modeled by a multinomial
    distribution.

    """

    # TODO this is only used by sample() and can be eliminated by inferring the
    # dtype from the generated samples.
    dtype = np.int64

    def __init__(self, probabilities):
        self._update(probabilities)

    def _update(self, probabilities):
        _probabilities = np.asarray(probabilities)
        # clip small neg residual (GH #34)
        _probabilities[_probabilities < 0] = 0

        xs = np.arange(_probabilities.shape[1])
        _probability_rvs = [
            NonParametricDistribution(xs, ps) for ps in _probabilities
        ]
        self._probabilities = _probabilities
        self._probability_rvs = _probability_rvs

    def likelihood(self, obs):
        obs = np.squeeze(obs)
        return np.vstack([rv.pmf(obs) for rv in self._probability_rvs])

    def copy(self):
        return MultinomialEmissions(self._probabilities.copy())

    def sample_for_state(self, state, size=None):
        return self._probability_rvs[state].rvs(size=size)

    def reestimate(self, gamma, observations):
        new_emissions = np.empty_like(self._probabilities)
        for em in range(self._probabilities.shape[1]):
            mask = observations == em
            new_emissions[:, em] = (
                gamma[:, mask].sum(axis=1) / gamma.sum(axis=1)
            )
        self._update(new_emissions)


class GaussianEmissions(AbstractEmissions):
    """ An emissions class for Gaussian emissions.

    This emissions class models the case where emissions are real-valued
    and continuous, and the probability of observing an emission given
    the state is modeled by a Gaussian. The means and standard deviations
    for each Gaussian (one for each state) are stored as state on the
    class.

    """

    dtype = np.float64

    def __init__(self, means, scales):
        self.means = means
        self.scales = scales

    def likelihood(self, obs):
        obs = np.squeeze(obs)
        # TODO: build in some check for the shape of the likelihoods, otherwise
        # this will silently fail and give the wrong results.
        return norm.pdf(obs,
                        loc=self.means[:, np.newaxis],
                        scale=self.scales[:, np.newaxis])

    def sample_for_state(self, state, size=None):
        return norm.rvs(self.means[state], self.scales[state], size)

    def copy(self):
        return GaussianEmissions(self.means.copy(), self.scales.copy())

    def reestimate(self, gamma, observations):
        p = np.sum(gamma * observations[np.newaxis, :], axis=1)
        q = np.sum(gamma, axis=1)
        new_means = p / q

        A = observations[np.newaxis, :] - new_means[:, np.newaxis]
        p = np.sum(gamma * A**2, axis=1)
        variances = p / q
        new_scales = np.sqrt(variances)

        self.means = new_means
        self.scales = new_scales


class GMMEmissions(AbstractEmissions):
    """
    A mixture of Gaussians per HSMM state.

    Parameters
    ----------
    weights : list[list[float]]
        Per-state mixture weights; weights[s] has length K_s and sums to 1.
    mus : list[list[np.ndarray]]
        Per-state component means; mus[s][k] is (D,) for component k of state s.
    covs : list[list[np.ndarray]]
        Per-state component covariances; covs[s][k] is (D,D).
    reg_covar : float
        Diagonal loading for covariance PSD stability (added at init and after each update).
    allow_singular : bool
        Passed to scipy's multivariate_normal.
    dtype : numpy dtype
        Numeric dtype for outputs.
    """

    dtype = np.float64

    def __init__(self, weights, mus, covs, reg_covar=1e-6, allow_singular=False):
        self.weights = [np.asarray(w, dtype=self.dtype) for w in weights]
        self.mus     = [[np.asarray(m, dtype=self.dtype) for m in mus_s] for mus_s in mus]
        self.covs    = [[self._as_full_cov(c) for c in covs_s] for covs_s in covs]
        self.n_states = len(self.weights)
        self.reg_covar = float(reg_covar)
        self.allow_singular = bool(allow_singular)
        self._normalize_weights()
        self._regularize_all_covs()
        self._freeze_all_rvs()

    # ---------- required by AbstractEmissions ----------

    def likelihood(self, obs):
        """
        Return P(x_t | state=s) for all s,t as an array of shape (n_states, T).
        Stable computation via log-sum-exp over mixture components.
        """
        X = np.atleast_2d(np.asarray(obs, dtype=self.dtype))  # (T, D)
        T = X.shape[0]
        out = np.empty((self.n_states, T), dtype=self.dtype)

        for s in range(self.n_states):
            # log w_k + log N(x|mu, Sigma)
            log_terms = []
            for k, rv in enumerate(self._frozen[s]):
                log_terms.append(np.log(self.weights[s][k] + 1e-16) + rv.logpdf(X))
            log_terms = np.vstack(log_terms)  # (K_s, T)
            out[s] = np.exp(logsumexp(log_terms, axis=0)) + 1e-300  # tiny floor

        return np.ascontiguousarray(out, dtype=self.dtype)

    def sample_for_state(self, state, size=None):
        """
        Draw samples respecting the mixture (component per sample).
        Returns shape (D,) if size is None; else (size, D).
        """
        K = len(self.weights[state])
        D = self.mus[state][0].shape[0]

        if size is None:
            k = np.random.choice(K, p=self.weights[state])
            return multivariate_normal.rvs(self.mus[state][k], self.covs[state][k], size=None)

        idx = np.random.choice(K, size=size, p=self.weights[state])
        out = np.empty((size, D), dtype=self.dtype)
        # use frozen rvs for speed
        for i, k in enumerate(idx):
            out[i] = self._frozen[state][k].rvs(size=1)
        return out

    def reestimate(self, gamma, observations, reg=None):
        """
        One EM M-step for each state's GMM, weighted by HSMM smoothed state posteriors gamma.

        gamma: (n_states, T)  smoothed state probs P(s | o_1..o_T)
        observations: (T, D)
        reg: optional diagonal loading (defaults to self.reg_covar)
        """
        X = np.atleast_2d(np.asarray(observations, dtype=self.dtype))  # (T, D)
        T, D = X.shape
        reg = self.reg_covar if reg is None else float(reg)

        for s in range(self.n_states):
            K = len(self.weights[s])

            # E-step in log domain
            log_comp = []
            for k, rv in enumerate(self._frozen[s]):
                log_comp.append(np.log(self.weights[s][k] + 1e-16) + rv.logpdf(X))  # (T,)
            log_comp = np.vstack(log_comp)  # (K, T)

            # incorporate state posterior gamma[s] per time step
            log_comp = log_comp + np.log(np.asarray(gamma[s], dtype=self.dtype) + 1e-16)[None, :]
            log_r = log_comp - logsumexp(log_comp, axis=0, keepdims=True)  # (K, T)
            r = np.exp(log_r)  # responsibilities per component
            Nks = r.sum(axis=1) + 1e-16  # (K,)

            # M-step
            # weights
            self.weights[s] = (Nks / Nks.sum()).astype(self.dtype)

            # means
            new_mus = []
            for k in range(K):
                rk = r[k][:, None]  # (T, 1)
                mu = (rk * X).sum(axis=0) / Nks[k]
                new_mus.append(mu.astype(self.dtype))

            # covariances
            new_covs = []
            for k in range(K):
                mu = new_mus[k]
                xc = X - mu  # (T, D)
                rk = r[k][:, None]  # (T, 1)
                # Σ = (1/Nk) * sum_i r_ik * (x_i - μ)(x_i - μ)^T
                Sigma = (rk * xc).T @ xc / Nks[k]
                # diagonal loading for PSD
                Sigma = np.asarray(Sigma, dtype=self.dtype)
                Sigma.flat[::D + 1] += reg
                new_covs.append(Sigma)

            self.mus[s] = new_mus
            self.covs[s] = new_covs

        # refresh frozen distributions after updates
        self._freeze_all_rvs()

    def copy(self):
        import copy
        return GMMEmissions(copy.deepcopy(self.weights),
                            copy.deepcopy(self.mus),
                            copy.deepcopy(self.covs),
                            reg_covar=self.reg_covar,
                            allow_singular=self.allow_singular)

    # ---------- convenience helpers ----------

    @staticmethod
    def from_param_dict(emission_dict, reg_covar=1e-6, allow_singular=False):
        """
        Build from your JSON-like structure:
        emission_dict = {
          0: {"weights": [...], "mu": [[...], ...], "cov": [[[...]], ...]},
          1: {...},
          ...
        }
        """
        states = [k for k in sorted(emission_dict.keys(), key=int)]
        weights = [np.asarray(emission_dict[s]["weights"]) for s in states]
        mus     = [np.asarray(emission_dict[s]["mu"]) for s in states]      # (K, D)
        covs    = [np.asarray(emission_dict[s]["cov"]) for s in states]     # (K, D, D)
        # convert to nested lists of arrays
        mus  = [[m for m in M] for M in mus]
        covs = [[C for C in Cs] for Cs in covs]
        return GMMEmissions(weights, mus, covs, reg_covar=reg_covar, allow_singular=allow_singular)

    # ---------- internal ----------

    def _regularize_all_covs(self):
        for s in range(self.n_states):
            for k in range(len(self.covs[s])):
                C = np.asarray(self.covs[s][k], dtype=self.dtype)
                D = C.shape[0]
                C.flat[::D + 1] += self.reg_covar
                self.covs[s][k] = C

    def _freeze_all_rvs(self):
        """Cache scipy frozen rvs for speed and stability."""
        self._frozen = []
        for s in range(self.n_states):
            comps = []
            for k in range(len(self.covs[s])):
                comps.append(multivariate_normal(mean=self.mus[s][k],
                                                 cov=self.covs[s][k],
                                                 allow_singular=self.allow_singular))
            self._frozen.append(comps)

    def _normalize_weights(self):
        for s in range(self.n_states):
            w = np.asarray(self.weights[s], dtype=self.dtype)
            w = np.maximum(w, 0)
            ssum = w.sum()
            if not np.isfinite(ssum) or ssum <= 0:
                # fallback to uniform if degenerate
                w[:] = 1.0 / len(w)
            else:
                w /= ssum
            self.weights[s] = w

    @staticmethod
    def _as_full_cov(C):
        C = np.asarray(C)
        if C.ndim == 1:  # diagonal given as vector
            return np.diag(C)
        return C
