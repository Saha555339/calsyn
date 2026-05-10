"""Noise distribution: fit and sample ε from residuals."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

NoiseKind = Literal["auto", "normal", "t"]


class NoiseModel:
    """Residual noise distribution.

    Parameters
    ----------
    kind : {"auto", "normal", "t"}
        - "auto"   — fit both normal and t to residuals, pick by BIC.
        - "normal" — fit N(0, σ²).
        - "t"      — fit scaled Student-t (df, loc, scale).
    random_seed : int or None
        Seed for sampling. None means a new random seed each call.
    """

    def __init__(self, kind: NoiseKind = "auto", random_seed: int | None = None) -> None:
        self.kind = kind
        self.random_seed = random_seed
        self.dist_name_: str | None = None
        self.params_: dict[str, float] | None = None
        self._frozen: stats.rv_continuous | None = None

    def fit(self, residuals: NDArray) -> "NoiseModel":
        """Fit distribution to residuals.

        Parameters
        ----------
        residuals : array of shape (n_samples,)

        Returns
        -------
        self
        """
        residuals = np.asarray(residuals, dtype=np.float64)

        if self.kind == "normal":
            self._fit_normal(residuals)
        elif self.kind == "t":
            self._fit_t(residuals)
        elif self.kind == "auto":
            self._fit_auto(residuals)
        else:
            raise ValueError(f"Unknown noise kind: {self.kind!r}")
        return self

    def set_scale(self, scale: float) -> None:
        """Override the fitted noise scale without re-fitting.

        Parameters
        ----------
        scale : float
            New scale parameter for the distribution.
        """
        if self._frozen is None:
            raise RuntimeError("Call .fit() first.")
        scale = float(scale)
        self.params_["scale"] = scale
        if self.dist_name_ == "normal":
            self._frozen = stats.norm(loc=self.params_["loc"], scale=scale)
        elif self.dist_name_ == "t":
            self._frozen = stats.t(df=self.params_["df"], loc=self.params_["loc"], scale=scale)

    def sample(self, size: int | tuple[int, ...]) -> NDArray:
        """Sample noise values.

        Parameters
        ----------
        size : int or tuple of ints
            Output shape.

        Returns
        -------
        array of given shape
        """
        if self._frozen is None:
            raise RuntimeError("Call .fit() before .sample().")
        return self._frozen.rvs(size=size, random_state=self.random_seed)

    def sample_rng(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray:
        """Sample with an external Generator (for per-trajectory seeds)."""
        if self._frozen is None:
            raise RuntimeError("Call .fit() before .sample_rng().")
        return self._frozen.rvs(size=size, random_state=rng)

    # ---- internals ----

    def _fit_normal(self, r: NDArray) -> None:
        loc, scale = float(np.mean(r)), float(np.std(r, ddof=1))
        self.dist_name_ = "normal"
        self.params_ = {"loc": loc, "scale": scale}
        self._frozen = stats.norm(loc=loc, scale=scale)

    def _fit_t(self, r: NDArray) -> None:
        df, loc, scale = stats.t.fit(r)
        self.dist_name_ = "t"
        self.params_ = {"df": df, "loc": loc, "scale": scale}
        self._frozen = stats.t(df=df, loc=loc, scale=scale)

    def _fit_auto(self, r: NDArray) -> None:
        n = len(r)

        # normal
        loc_n, scale_n = float(np.mean(r)), float(np.std(r, ddof=1))
        ll_n = np.sum(stats.norm.logpdf(r, loc=loc_n, scale=scale_n))
        bic_n = 2 * np.log(n) - 2 * ll_n  # 2 parameters

        # t
        df_t, loc_t, scale_t = stats.t.fit(r)
        ll_t = np.sum(stats.t.logpdf(r, df=df_t, loc=loc_t, scale=scale_t))
        bic_t = 3 * np.log(n) - 2 * ll_t  # 3 parameters

        if bic_t < bic_n:
            self.dist_name_ = "t"
            self.params_ = {"df": df_t, "loc": loc_t, "scale": scale_t}
            self._frozen = stats.t(df=df_t, loc=loc_t, scale=scale_t)
        else:
            self.dist_name_ = "normal"
            self.params_ = {"loc": loc_n, "scale": scale_n}
            self._frozen = stats.norm(loc=loc_n, scale=scale_n)
