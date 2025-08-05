import numpy as np
from hsmmlearn.hsmm import HSMMModel
from hsmmlearn.emissions import GMMEmissions
from scipy.stats import gamma, lognorm


class HSMMEngine:
    def __init__(self, params: dict, D_max: int = 400):
        self.state_mapping = params["state_mapping"]
        self.model = self._build_model(params, D_max)

    def _duration_pmf(self, d_params: dict, D_max: int) -> np.ndarray:
        """Discrete PMF over durations 1..D_max from a continuous distribution."""
        dist = d_params.get("dist", "gamma").lower()

        if dist == "gamma":
            a = float(d_params["shape"])
            scale = float(d_params["scale"])
            loc = float(d_params.get("loc", 0.0))
            F = gamma(a=a, scale=scale, loc=loc).cdf

        elif dist in ("lognorm", "lognormal"):
            loc = float(d_params.get("loc", 0.0))
            s = float(d_params["sigma"])       # sigma
            mu = float(d_params["mu"])         # mu
            scale = float(np.exp(mu))          # SciPy: scale = exp(mu)
            F = lognorm(s=s, scale=scale, loc=loc).cdf

        else:
            raise ValueError(f"Unsupported duration distribution '{dist}'. Use 'gamma' or 'lognorm'.")

        # Build PMF on k = 1..D_max via CDF differences; bucket tail mass ≥ D_max into the last bin
        k_prev = np.arange(0, D_max, dtype=float)     # 0..D_max-1
        k_curr = np.arange(1, D_max + 1, dtype=float) # 1..D_max
        cdf_prev = F(k_prev)
        pmf = F(k_curr) - cdf_prev
        pmf[-1] = 1.0 - cdf_prev[-1]                  # pool tail mass into D_max

        # numerical guard + normalize
        pmf = np.maximum(pmf, 0)
        s = pmf.sum()
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"Duration PMF for dist '{dist}' had non-positive/invalid mass.")
        return pmf / s


    def _build_model(self, params, D_max):
        # Emissions (GMM)
        states = sorted(int(s) for s in params["emission"])
        weights, mus, covs = [], [], []
        for s in states:
            e = params["emission"][str(s)]
            weights.append(e["weights"])
            mus.append(e["mu"])
            covs.append(e["cov"])
        emissions = GMMEmissions(weights, mus, covs)

        # Durations
        durations = np.zeros((len(states), D_max), dtype=float)
        for idx, s in enumerate(states):
            d = params["duration"][str(s)]
            durations[idx] = self._duration_pmf(d, D_max)            

        # Transitions
        n_states = len(params["state_mapping"])
        trans_mat = np.zeros((n_states, n_states))
        for i_str, to_dict in params["transition"].items():
            i = int(i_str)
            for j_str, prob in to_dict.items():
                j = int(j_str)
                trans_mat[i, j] = prob

        # Initial state probabilities
        startprob = np.zeros(n_states)
        for s, p in params["initial_state"].items():
            startprob[int(s)] = p

        return HSMMModel(emissions=emissions, durations=durations, tmat=trans_mat, startprob=startprob)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """"Predict the hidden states for the feature set X"""
        raw_states = self.model.decode(X)
        mapped_states = [int(k) for s in raw_states for k, v in self.state_mapping.items() if v == s]
        return np.array(mapped_states, dtype=int)
