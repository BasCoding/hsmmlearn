import numpy as np
from hsmmlearn.hsmm import HSMMModel
from hsmmlearn.emissions import GMMEmissions
from scipy.stats import gamma, lognorm


class HSMMEngine:
    def __init__(self, params: dict, D_max: int = 400):
        self.model = self._build_model(params, D_max)
        self.state_mapping = params.get("state_mapping")  # original -> internal
        self._inv_state_mapping = None
        if self.state_mapping:
            # build internal -> original (numpy array is fast for vectorized mapping)
            max_idx = max(self.state_mapping.values())
            inv = np.full(max_idx + 1, -1, dtype=int)
            for orig, internal in self.state_mapping.items():
                if inv[internal] != -1:
                    raise ValueError(
                        f"state_mapping is not one-to-one. Internal state {internal} "
                        f"has multiple original labels: {inv[internal]} and {orig}"
                    )
                inv[internal] = int(orig)
            if (inv == -1).any():
                missing = np.where(inv == -1)[0]
                raise ValueError(f"state_mapping missing inverse for internal states: {missing}")
            self._inv_state_mapping = inv

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
        pmf[-1] = 1.0 - F(D_max)                 # pool tail mass into D_max

        # numerical guard + normalize
        pmf = np.maximum(pmf, 0)
        s = pmf.sum()
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"Duration PMF for dist '{dist}' had non-positive/invalid mass.")
        return pmf / s

    def _build_model(self, params, D_max):
        # Emissions (GMM)
        states = sorted(int(s) for s in params["emission"])
        emissions = GMMEmissions.from_param_dict(params["emission"], reg_covar=1e-6)

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
        raw_states = np.asarray(self.model.decode(X), dtype=int)
        if self._inv_state_mapping is not None:
            if raw_states.min() < 0 or raw_states.max() >= len(self._inv_state_mapping):
                raise ValueError(
                    f"Decoded state out of range: [{raw_states.min()}, {raw_states.max()}] "
                    f"but inv map length is {len(self._inv_state_mapping)}"
                )
            return self._inv_state_mapping[raw_states]
        return raw_states
