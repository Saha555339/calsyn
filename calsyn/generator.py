"""CalibratedGenerator — main entry point for calsyn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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


class CalibratedGenerator:
    """Calibrated synthetic data generator.

    Data-generating process:  Y = effect(t) + f(X) + eps

    where f is a CatBoost model calibrated on real (X, y) data, effect(t)
    is a date-based treatment vector, and eps is sampled from a distribution
    fitted to the calibration residuals.

    Parameters
    ----------
    date_col : str
        Name of the datetime column in X.
    noise : {"auto", "normal", "t"}
        Noise distribution type.
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
    random_seed : int
        Global random seed.

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
        auto_tune: bool = False,
        n_trials: int = 50,
        val_fraction: float = 0.1,
        strat_freq: str = "M",
        catboost_params: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.date_col = date_col
        self.noise = noise
        self.auto_tune = auto_tune
        self.n_trials = n_trials
        self.val_fraction = val_fraction
        self.strat_freq = strat_freq
        self.catboost_params = catboost_params
        self.random_seed = random_seed

        self._model: CalibrationModel | None = None
        self._noise_model: NoiseModel | None = None
        self._X_fit: pd.DataFrame | None = None
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

    def fit(self, X: pd.DataFrame, y: NDArray) -> "CalibratedGenerator":
        """Fit calibration model and noise distribution.

        The model is trained only on the training split (90% by default).
        OOS metrics are computed on the held-out stratified monthly sample.
        Noise distribution is fitted to train-set residuals.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain date_col and numeric feature columns.
        y : array of shape (n_samples,)
            Target variable.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64)
        self._X_fit = X.copy()
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

        # noise from train-set residuals only
        residuals = self._model.residuals(X, y)
        self._noise_model = NoiseModel(kind=self.noise, random_seed=self.random_seed)
        self._noise_model.fit(residuals)

        return self

    # ---- generate ----

    def generate(
        self,
        X: pd.DataFrame,
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
        X : pd.DataFrame
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

        dates = pd.to_datetime(X[self.date_col])
        f_X = self._model.predict(X)

        effect_or_grid = build_treatment_vector(
            dates,
            treatment_start=treatment_start,
            treatment=treatment,
            tau=tau,
        )

        # mode B or mode A with single tau: effect_or_grid is NDArray
        if isinstance(effect_or_grid, np.ndarray):
            return self._generate_single(f_X, effect_or_grid, n_simulations)

        # mode A with tau grid: effect_or_grid is dict[float, NDArray]
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

    # ---- diagnostics ----

    def diagnose(
        self,
        result: GenerationResult,
        X: pd.DataFrame | None = None,
        y_real: NDArray | None = None,
    ) -> DiagnosticReport:
        """Run full diagnostics on a generation result.

        Parameters
        ----------
        result : GenerationResult
        X : pd.DataFrame or None (defaults to fit data)
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

        X_numeric = X.drop(columns=[self.date_col]).values.astype(np.float64)
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
        X: pd.DataFrame | None = None,
        y_real: NDArray | None = None,
        feature_names: list[str] | None = None,
    ):
        """Plot feature correlation comparison."""
        if X is None:
            X = self._X_fit
        if y_real is None:
            y_real = self._y_fit
        y_real = np.asarray(y_real, dtype=np.float64)
        X_numeric = X.drop(columns=[self.date_col]).values.astype(np.float64)
        synth_mean = result.simulations.mean(axis=0)
        diags = feature_correlations(X_numeric, y_real, synth_mean)
        return plot_correlation_comparison(diags, feature_names=feature_names)
