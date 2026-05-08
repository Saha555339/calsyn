"""Calibration model: CatBoost wrapper with stratified monthly OOS and optional Optuna."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from numpy.typing import NDArray


def _default_catboost_params() -> dict[str, Any]:
    return {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": 42,
    }


STRAT_FREQ = {
    "M": "M",       # month
    "d": "D",       # day
    "m": "min",     # minute
    "h": "h",       # hour
    "W": "W",       # week
}


def stratified_time_split(
    dates: pd.Series,
    val_fraction: float,
    random_seed: int,
    strat_freq: str = "M",
) -> tuple[NDArray, NDArray]:
    """Split indices so that val_fraction of each time period goes to OOS.

    Parameters
    ----------
    dates : pd.Series of datetime-like
    val_fraction : float in (0, 1)
    random_seed : int
    strat_freq : str
        Stratification period: "M" (month), "d" (day), "m" (minute),
        "h" (hour), "W" (week).

    Returns
    -------
    (train_idx, val_idx) — integer numpy arrays
    """
    if strat_freq not in STRAT_FREQ:
        raise ValueError(
            f"Unknown strat_freq={strat_freq!r}. Choose from {list(STRAT_FREQ.keys())}."
        )
    pd_freq = STRAT_FREQ[strat_freq]

    rng = np.random.default_rng(random_seed)
    periods = dates.dt.to_period(pd_freq)
    train_parts, val_parts = [], []

    for _, group_idx in dates.groupby(periods).groups.items():
        idx = np.array(group_idx)
        rng.shuffle(idx)
        n_val = int(len(idx) * val_fraction)
        if n_val == 0 or len(idx) < 2:
            train_parts.append(idx)
        else:
            val_parts.append(idx[:n_val])
            train_parts.append(idx[n_val:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
    return train_idx, val_idx


def _optuna_objective(
    trial,
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
    random_seed: int,
) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 300, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 32),
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": random_seed,
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    preds = model.predict(X_val)
    return float(np.mean((y_val - preds) ** 2))


class CalibrationModel:
    """CatBoost calibration model with stratified time-period OOS validation.

    The model is fit ONLY on the training split.  OOS metrics are computed
    on the held-out portion (val_fraction sampled from each time period).

    Parameters
    ----------
    date_col : str
        Name of the datetime column inside X (DataFrame).
    auto_tune : bool
        If True, run Optuna hyperparameter search.
    n_trials : int
        Number of Optuna trials.
    val_fraction : float
        Fraction of each time period held out for OOS.
    strat_freq : str
        Stratification period: "M" (month), "d" (day), "m" (minute),
        "h" (hour), "W" (week).
    catboost_params : dict or None
        Custom CatBoost parameters (ignored when auto_tune=True).
    random_seed : int
        Random seed.
    """

    def __init__(
        self,
        date_col: str = "date",
        auto_tune: bool = False,
        n_trials: int = 50,
        val_fraction: float = 0.1,
        strat_freq: str = "M",
        catboost_params: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.date_col = date_col
        self.auto_tune = auto_tune
        self.n_trials = n_trials
        self.val_fraction = val_fraction
        self.strat_freq = strat_freq
        self.random_seed = random_seed
        self.catboost_params = catboost_params or _default_catboost_params()
        self.model_: CatBoostRegressor | None = None
        self.best_params_: dict[str, Any] | None = None
        self.oos_metrics_: dict[str, float] | None = None
        self.train_idx_: NDArray | None = None
        self.val_idx_: NDArray | None = None

    @staticmethod
    def _compute_metrics(y_true: NDArray, y_pred: NDArray) -> dict[str, float]:
        residuals = y_true - y_pred
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        bias = float(np.mean(residuals))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"r2": r2, "rmse": rmse, "bias": bias}

    def _feature_matrix(self, X: pd.DataFrame) -> NDArray:
        """Drop date column and return numeric array."""
        return X.drop(columns=[self.date_col]).values.astype(np.float64)

    def fit(self, X: pd.DataFrame, y: NDArray) -> "CalibrationModel":
        """Fit on train split, compute OOS metrics on val split.

        Parameters
        ----------
        X : pd.DataFrame with date_col and numeric feature columns
        y : array of shape (n_samples,)

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64)
        dates = pd.to_datetime(X[self.date_col])

        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1).")
        
        train_idx, val_idx = stratified_time_split(
            dates, self.val_fraction, self.random_seed, self.strat_freq
        )
        self.train_idx_ = train_idx
        self.val_idx_ = val_idx

        if len(train_idx) == 0:
            raise ValueError("Training split is empty.")

        if len(val_idx) == 0:
            raise ValueError(
                "Validation split is empty. Increase val_fraction or change strat_freq."
            )

        X_numeric = self._feature_matrix(X)
        X_train, y_train = X_numeric[train_idx], y[train_idx]
        X_val, y_val = X_numeric[val_idx], y[val_idx]

        # resolve params (optuna uses train/val split)
        if self.auto_tune:
            self.best_params_ = self._tune(X_train, y_train, X_val, y_val)
            params = {**self.best_params_, "verbose": 0, "random_seed": self.random_seed}
        else:
            params = {**self.catboost_params, "random_seed": self.random_seed}
            self.best_params_ = params

        # fit only on train
        self.model_ = CatBoostRegressor(**params)
        self.model_.fit(X_train, y_train)

        # OOS metrics
        oos_preds = self.model_.predict(X_val)
        self.oos_metrics_ = self._compute_metrics(y_val, oos_preds)

        return self

    def predict(self, X: pd.DataFrame | NDArray) -> NDArray:
        """Predict f(X)."""
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .predict().")
        if isinstance(X, pd.DataFrame):
            X = self._feature_matrix(X)
        return self.model_.predict(np.asarray(X, dtype=np.float64))

    def residuals(self, X: pd.DataFrame, y: NDArray) -> NDArray:
        """Compute residuals on train split: y_train - f(X_train)."""
        y = np.asarray(y, dtype=np.float64)
        X_numeric = self._feature_matrix(X)
        idx = self.train_idx_
        preds = self.model_.predict(X_numeric[idx])
        return y[idx] - preds

    def _tune(
        self, X_train: NDArray, y_train: NDArray, X_val: NDArray, y_val: NDArray
    ) -> dict[str, Any]:
        import optuna

        study = optuna.create_study(direction="minimize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            lambda trial: _optuna_objective(
                trial, X_train, y_train, X_val, y_val, self.random_seed
            ),
            n_trials=self.n_trials,
        )
        return study.best_params
