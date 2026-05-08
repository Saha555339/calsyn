"""Tests for calsyn core functionality."""

import numpy as np
import pandas as pd
import pytest

from calsyn import CalibratedGenerator


@pytest.fixture
def fast_catboost_params():
    return {
        "iterations": 30,
        "learning_rate": 0.1,
        "depth": 4,
        "loss_function": "RMSE",
        "verbose": 0,
    }


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    X = pd.DataFrame({
        "date": dates,
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "f3": rng.standard_normal(n),
    })
    y = np.sin(X["f1"].values) + 0.5 * X["f2"].values ** 2 + rng.normal(0, 0.3, n)
    return X, y


# --- fit & generate shapes ---

def test_fit_generate_mode_a(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    result = gen.generate(X, treatment_start="2023-07-01", tau=0.0, n_simulations=50)
    assert result.simulations.shape == (50, len(y))
    assert len(result.f_X) == len(y)
    assert len(result.effect) == len(y)


def test_fit_generate_mode_b(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    result = gen.generate(
        X,
        treatment=[("2023-03-01", "2023-03-15", 0.05), ("2023-06-01", "2023-06-10", -0.01)],
        n_simulations=30,
    )
    assert result.simulations.shape == (30, len(y))


# --- tau grid ---

def test_tau_grid(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    results = gen.generate(X, treatment_start="2023-07-01", tau=[0.0, 0.01, 0.05], n_simulations=10)
    assert isinstance(results, dict)
    assert set(results.keys()) == {0.0, 0.01, 0.05}
    for tau_val, res in results.items():
        assert res.simulations.shape == (10, len(y))


# --- effect shifts mean ---

def test_effect_shifts_mean(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    r0 = gen.generate(X, treatment_start="2023-07-01", tau=0.0, n_simulations=200)
    r1 = gen.generate(X, treatment_start="2023-07-01", tau=1.0, n_simulations=200)
    treated = pd.to_datetime(X["date"]) >= pd.Timestamp("2023-07-01")
    mean_diff = r1.simulations[:, treated].mean() - r0.simulations[:, treated].mean()
    assert abs(mean_diff - 1.0) < 0.15


# --- OOS metrics ---

def test_oos_metrics(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    m = gen.oos_metrics
    assert "r2" in m
    assert "rmse" in m
    assert "bias" in m


# --- noise ---

def test_noise_auto(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="auto",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    params = gen.noise_params
    assert params["distribution"] in ("normal", "t")


def test_noise_t(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="t",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    params = gen.noise_params
    assert params["distribution"] == "t"
    assert "df" in params


# --- diagnostics ---

def test_diagnose(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    result = gen.generate(X, treatment_start="2023-07-01", tau=0.0, n_simulations=100)
    report = gen.diagnose(result)
    assert report.ks.pvalue >= 0
    n_features = X.shape[1] - 1  # minus date col
    assert len(report.correlations) == n_features
    assert 0 <= report.coverage.coverage_1sigma <= 1
    assert 0 <= report.coverage.coverage_2sigma <= 1
    assert report.coverage.coverage_2sigma >= report.coverage.coverage_1sigma
    assert "r2" in report.oos_metrics


# --- treatment windows effect ---

def test_treatment_windows_effect(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    result = gen.generate(
        X,
        treatment=[("2023-03-01", "2023-03-31", 5.0)],
        n_simulations=100,
    )
    dates = pd.to_datetime(X["date"])
    in_window = (dates >= "2023-03-01") & (dates <= "2023-03-31")
    outside = ~in_window
    mean_in = result.simulations[:, in_window].mean()
    mean_out = result.simulations[:, outside].mean()
    assert mean_in - mean_out > 3.0


# --- errors ---

def test_not_fitted_raises():
    gen = CalibratedGenerator()
    X = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10), "f1": range(10)})
    with pytest.raises(RuntimeError):
        gen.generate(X, treatment_start="2023-01-05", tau=0.0)


def test_no_treatment_raises(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    with pytest.raises(ValueError):
        gen.generate(X, n_simulations=10)


# --- stratified monthly split ---

def test_stratified_split_covers_all_months(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        val_fraction=0.1,
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    train_idx = gen._model.train_idx_
    val_idx = gen._model.val_idx_
    dates = pd.to_datetime(X["date"])
    train_months = set(dates.iloc[train_idx].dt.to_period("M"))
    val_months = set(dates.iloc[val_idx].dt.to_period("M"))
    assert val_months == train_months


def test_strat_freq_daily(fast_catboost_params):
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    X = pd.DataFrame({
        "date": dates,
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
    })
    y = X["f1"].values + rng.normal(0, 0.3, n)
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        val_fraction=0.1,
        strat_freq="d",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    m = gen.oos_metrics
    assert "r2" in m
    assert len(gen._model.val_idx_) > 0


def test_strat_freq_weekly(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        val_fraction=0.3,
        strat_freq="W",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    assert len(gen._model.val_idx_) > 0


def test_strat_freq_invalid(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        strat_freq="Z",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    with pytest.raises(ValueError):
        gen.fit(X, y)
