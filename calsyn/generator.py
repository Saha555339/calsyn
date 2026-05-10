"""CalibratedGenerator — main entry point for calsyn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from calsyn.diagnostics import (
    CoverageDiagnostic,
    CorrelationDiagnostic,
    KSResult,
    coverage,
    feature_correlations,
    ks_test,
)
from calsyn.model import CalibrationModel
from calsyn.noise import NoiseKind, NoiseModel
from calsyn.plotting import plot_correlation_comparison, plot_coverage
from calsyn.treatment import build_treatment_vector


@dataclass
class GenerationResult:
    """Container for generation output.

    Attributes
    ----------
    simulations : array of shape (n_simulations, n_samples)
        Synthetic trajectories.
    effect : array of shape (n_samples,)
        The combined effect vector (tau * D or sum of window effects).
    f_X : array of shape (n_samples,)
        Calibrated model predictions f(X).
    """

    simulations: NDArray
    effect: NDArray
    f_X: NDArray


@dataclass
class DiagnosticReport:
    """Aggregated diagnostics."""

    ks: KSResult
    correlations: list[CorrelationDiagnostic]
    coverage: CoverageDiagnostic
    oos_metrics: dict[str, float]


@dataclass
class FeatureNoiseResult:
    """Container for feature-noise generation output.

    Attributes
    ----------
    simulations : array of shape (n_simulations, n_samples)
        Synthetic Y trajectories: f(X̃) + ε_Y.
    X_simulations : dict[str, array of shape (n_simulations, n_samples)]
        Noisy feature values X̃[j] = X[j] + ε_X[j] for each noised feature.
        Only features listed in ``feature_noise`` are included.
    f_X_mean : array of shape (n_samples,)
        Model predictions f(X) on the original (un-noised) features, for reference.
    """

    simulations: NDArray
    X_simulations: dict[str, NDArray]
    f_X_mean: NDArray


def _sample_feature_noise(
    spec: dict, size: tuple, rng: np.random.Generator
) -> NDArray:
    """Sample additive noise for a single feature according to ``spec``.

    Parameters
    ----------
    spec : dict
        Keys: ``distribution`` ("normal" or "t"), ``scale`` (float),
        ``loc`` (float, optional, default 0), ``df`` (float, required for "t").
    size : tuple of ints
        Output shape.
    rng : numpy Generator

    Returns
    -------
    NDArray of given shape
    """
    dist = spec.get("distribution")
    loc = float(spec.get("loc", 0.0))
    scale = float(spec["scale"])

    if dist == "normal":
        if scale == 0.0:
            return np.zeros(size) if loc == 0.0 else np.full(size, loc)
        return rng.normal(loc=loc, scale=scale, size=size)
    elif dist == "t":
        if "df" not in spec:
            raise ValueError("'df' is required for 't' distribution.")
        df = float(spec["df"])
        if scale == 0.0:
            return np.zeros(size) if loc == 0.0 else np.full(size, loc)
        return loc + scale * rng.standard_t(df=df, size=size)
    else:
        raise ValueError(
            f"Unknown distribution {dist!r}. Supported: 'normal', 't'."
        )


class CalibratedGenerator:
    """Calibrated synthetic data generator.

    Data-generating process:  Y = effect(t) + f(X) + eps

    where f is a CatBoost model calibrated on real (X, y) data, effect(t)
    is a date-based treatment vector, and eps is sampled from a distribution
    fitted to the calibration residuals.

    Parameters
    ----------
    date_col : str
        Name of the date/datetime column in X.
    noise : {"auto", "normal", "t"}
        Noise distribution type.
    noise_scale : float or None
        If provided, overrides the fitted noise scale after fitting.
        Useful when residual variance is unreliable (e.g. low data variability).
    auto_tune : bool
        If True, use Optuna to tune CatBoost hyperparameters.
    n_trials : int
        Number of Optuna trials (ignored when auto_tune=False).
    val_fraction : float
        Fraction of each time period held out for OOS validation.
    strat_freq : str
        Stratification period for OOS split: "M" (month), "d" (day),
        "m" (minute), "h" (hour), "W" (week).
    catboost_params : dict or None
        Custom CatBoost parameters (ignored when auto_tune=True).
    random_seed : int or None
        Global random seed. None (default) means outputs vary between runs.

    Examples
    --------
    >>> gen = CalibratedGenerator(date_col="date", noise="auto")
    >>> gen.fit(df, y)
    >>> result = gen.generate(df, treatment_start="2024-06-01", tau=0.02)
    >>> result.simulations.shape
    (500, n_samples)
    """

    def __init__(
        self,
        date_col: str = "date",
        noise: NoiseKind = "auto",
        noise_scale: float | None = None,
        auto_tune: bool = False,
        n_trials: int = 50,
        val_fraction: float = 0.1,
        strat_freq: str = "M",
        catboost_params: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.date_col = date_col
        self.noise = noise
        self.noise_scale = noise_scale
        self.auto_tune = auto_tune
        self.n_trials = n_trials
        self.val_fraction = val_fraction
        self.strat_freq = strat_freq
        self.catboost_params = catboost_params
        self.random_seed = random_seed

        self._model: CalibrationModel | None = None
        self._noise_model: NoiseModel | None = None
        self._X_fit: pl.DataFrame | None = None
        self._y_fit: NDArray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._model.model_ is not None

    @property
    def oos_metrics(self) -> dict[str, float]:
        """Out-of-sample metrics: R2, RMSE, bias."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")
        return self._model.oos_metrics_

    @property
    def noise_params(self) -> dict[str, Any]:
        """Fitted noise distribution parameters."""
        if self._noise_model is None or self._noise_model.params_ is None:
            raise RuntimeError("Call .fit() first.")
        return {
            "distribution": self._noise_model.dist_name_,
            **self._noise_model.params_,
        }

    # ---- fit ----

    def fit(self, X: pl.DataFrame, y: NDArray) -> "CalibratedGenerator":
        """Fit calibration model and noise distribution.

        The model is trained only on the training split (90% by default).
        OOS metrics are computed on the held-out stratified monthly sample.
        Noise distribution is fitted to train-set residuals.

        Parameters
        ----------
        X : pl.DataFrame
            Must contain date_col (Date or Datetime) and numeric feature columns.
        y : array of shape (n_samples,)
            Target variable.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64)
        self._X_fit = X
        self._y_fit = y

        self._model = CalibrationModel(
            date_col=self.date_col,
            auto_tune=self.auto_tune,
            n_trials=self.n_trials,
            val_fraction=self.val_fraction,
            strat_freq=self.strat_freq,
            catboost_params=self.catboost_params,
            random_seed=self.random_seed,
        )
        self._model.fit(X, y)

        residuals = self._model.residuals(X, y)
        self._noise_model = NoiseModel(kind=self.noise, random_seed=self.random_seed)
        self._noise_model.fit(residuals)

        if self.noise_scale is not None:
            self._noise_model.set_scale(self.noise_scale)

        return self

    # ---- generate ----

    def generate(
        self,
        X: pl.DataFrame,
        treatment_start: str | None = None,
        treatment: list[tuple[str, str, float]] | None = None,
        tau: float | list[float] | None = None,
        n_simulations: int = 500,
    ) -> GenerationResult | dict[float, GenerationResult]:
        """Generate synthetic trajectories.

        DGP:  Y_sim = effect(t) + f(X) + eps

        **Mode A — treatment_start + tau:**
            effect = tau * D, where D=1 for dates >= treatment_start.
            If tau is a list, returns dict[float, GenerationResult].

        **Mode B — treatment (list of windows):**
            treatment = [(start, end, tau), ...].
            Each window adds its tau to the effect vector.
            Returns a single GenerationResult.

        Parameters
        ----------
        X : pl.DataFrame
        treatment_start : str or None
        treatment : list of (start_date, end_date, tau) or None
        tau : float or list[float] or None
        n_simulations : int

        Returns
        -------
        GenerationResult or dict[float, GenerationResult]
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

        dates = X[self.date_col]
        f_X = self._model.predict(X)

        effect_or_grid = build_treatment_vector(
            dates,
            treatment_start=treatment_start,
            treatment=treatment,
            tau=tau,
        )

        if isinstance(effect_or_grid, np.ndarray):
            return self._generate_single(f_X, effect_or_grid, n_simulations)

        return {
            t: self._generate_single(f_X, eff, n_simulations)
            for t, eff in effect_or_grid.items()
        }

    def _generate_single(
        self, f_X: NDArray, effect: NDArray, n_simulations: int
    ) -> GenerationResult:
        n = len(f_X)
        rng = np.random.default_rng(self.random_seed)
        eps = self._noise_model.sample_rng(size=(n_simulations, n), rng=rng)
        base = effect + f_X
        simulations = base[np.newaxis, :] + eps
        return GenerationResult(simulations=simulations, effect=effect, f_X=f_X)

    # ---- feature noise generation ----

    def generate_with_feature_noise(
        self,
        X: pl.DataFrame,
        feature_noise: dict[str, dict],
        n_simulations: int = 500,
    ) -> FeatureNoiseResult:
        """Generate Y trajectories by injecting user-specified noise into features.

        DGP:  Y_sim = f(X̃) + ε_Y
              X̃[j] = X[j] + ε_X[j]  for j in feature_noise
              X̃[j] = X[j]            otherwise

        This is a sensitivity analysis tool — it does not use treatment effects.
        Use :meth:`generate` for treatment-based simulations.

        Parameters
        ----------
        X : pl.DataFrame
        feature_noise : dict[str, dict]
            Mapping of feature name → noise spec. Each spec must contain:

            - ``distribution``: ``"normal"`` or ``"t"``
            - ``scale``: float — std for normal, scale parameter for t
            - ``loc``: float, optional (default 0)
            - ``df``: float, required when ``distribution="t"``

        n_simulations : int

        Returns
        -------
        FeatureNoiseResult

        Examples
        --------
        >>> result = gen.generate_with_feature_noise(
        ...     X,
        ...     feature_noise={
        ...         "USDRUB": {"distribution": "normal", "scale": 0.01},
        ...         "OIL":    {"distribution": "t", "scale": 0.02, "df": 4},
        ...     },
        ...     n_simulations=500,
        ... )
        >>> result.simulations.shape
        (500, n_samples)
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

        feature_cols = [c for c in X.columns if c != self.date_col]
        for fname in feature_noise:
            if fname not in feature_cols:
                raise ValueError(
                    f"Feature {fname!r} not found in X. "
                    f"Available features: {feature_cols}."
                )

        n = len(X)
        rng = np.random.default_rng(self.random_seed)

        # Sample all feature noises at once: (n_simulations, n_samples) per feature.
        # scale=0 returns zeros without consuming RNG state, keeping eps_Y reproducible.
        feat_noise_arrays: dict[str, NDArray] = {
            fname: _sample_feature_noise(spec, size=(n_simulations, n), rng=rng)
            for fname, spec in feature_noise.items()
        }

        # Sample Y noise
        eps_Y = self._noise_model.sample_rng(size=(n_simulations, n), rng=rng)

        X_np = X.drop(self.date_col).to_numpy().astype(np.float64)
        feat_idx = {fname: feature_cols.index(fname) for fname in feature_noise}

        f_X_mean = self._model.predict(X_np)

        # Noisy feature values: X̃[j] = X[j] + ε_X[j], shape (n_simulations, n_samples)
        X_simulations: dict[str, NDArray] = {
            fname: X_np[:, feat_idx[fname]] + noise_arr
            for fname, noise_arr in feat_noise_arrays.items()
        }

        simulations = np.empty((n_simulations, n))
        for i in range(n_simulations):
            X_tilde = X_np.copy()
            for fname, idx in feat_idx.items():
                X_tilde[:, idx] = X_simulations[fname][i]
            simulations[i] = self._model.predict(X_tilde) + eps_Y[i]

        return FeatureNoiseResult(
            simulations=simulations,
            X_simulations=X_simulations,
            f_X_mean=f_X_mean,
        )

    # ---- diagnostics ----

    def diagnose(
        self,
        result: GenerationResult,
        X: pl.DataFrame | None = None,
        y_real: NDArray | None = None,
    ) -> DiagnosticReport:
        """Run full diagnostics on a generation result.

        Parameters
        ----------
        result : GenerationResult
        X : pl.DataFrame or None (defaults to fit data)
        y_real : array or None (defaults to fit data)

        Returns
        -------
        DiagnosticReport
        """
        if X is None:
            X = self._X_fit
        if y_real is None:
            y_real = self._y_fit
        y_real = np.asarray(y_real, dtype=np.float64)

        X_numeric = X.drop(self.date_col).to_numpy().astype(np.float64)
        synth_mean = result.simulations.mean(axis=0)

        ks_res = ks_test(y_real, synth_mean)
        corr_diag = feature_correlations(X_numeric, y_real, synth_mean)
        cov_diag = coverage(y_real, result.simulations)

        return DiagnosticReport(
            ks=ks_res,
            correlations=corr_diag,
            coverage=cov_diag,
            oos_metrics=self.oos_metrics,
        )

    # ---- plotting ----

    def plot_coverage(
        self,
        result: GenerationResult,
        y_real: NDArray | None = None,
        **kwargs,
    ):
        """Plot real trajectory vs synthetic CI (1 sigma, 2 sigma)."""
        y_real = self._y_fit if y_real is None else np.asarray(y_real, dtype=np.float64)
        return plot_coverage(y_real, result.simulations, **kwargs)

    def plot_correlations(
        self,
        result: GenerationResult,
        X: pl.DataFrame | None = None,
        y_real: NDArray | None = None,
        feature_names: list[str] | None = None,
    ):
        """Plot feature correlation comparison."""
        if X is None:
            X = self._X_fit
        if y_real is None:
            y_real = self._y_fit
        y_real = np.asarray(y_real, dtype=np.float64)
        X_numeric = X.drop(self.date_col).to_numpy().astype(np.float64)
        synth_mean = result.simulations.mean(axis=0)
        diags = feature_correlations(X_numeric, y_real, synth_mean)
        return plot_correlation_comparison(diags, feature_names=feature_names)
