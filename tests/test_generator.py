"""Tests for calsyn core functionality."""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
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
    start = date(2023, 1, 1)
    dates = pl.Series([start + timedelta(days=i) for i in range(n)])
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    f3 = rng.standard_normal(n)
    X = pl.DataFrame({"date": dates, "f1": f1, "f2": f2, "f3": f3})
    y = np.sin(f1) + 0.5 * f2 ** 2 + rng.normal(0, 0.3, n)
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
    for _, res in results.items():
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
    treated = (X["date"].cast(pl.Date) >= date(2023, 7, 1)).to_numpy()
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


def test_noise_scale_override(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        noise_scale=10.0,
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    assert abs(gen.noise_params["scale"] - 10.0) < 1e-9


# --- random_seed=None produces variability ---

def test_random_seed_none_varies(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=None,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    r1 = gen.generate(X, treatment_start="2023-07-01", tau=0.0, n_simulations=10)
    r2 = gen.generate(X, treatment_start="2023-07-01", tau=0.0, n_simulations=10)
    # Two runs with no seed should produce different simulations
    assert not np.allclose(r1.simulations, r2.simulations)


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
    in_window = (
        (X["date"].cast(pl.Date) >= date(2023, 3, 1))
        & (X["date"].cast(pl.Date) <= date(2023, 3, 31))
    ).to_numpy()
    outside = ~in_window
    mean_in = result.simulations[:, in_window].mean()
    mean_out = result.simulations[:, outside].mean()
    assert mean_in - mean_out > 3.0


# --- errors ---

def test_not_fitted_raises():
    gen = CalibratedGenerator()
    X = pl.DataFrame({
        "date": [date(2023, 1, 1) + timedelta(days=i) for i in range(10)],
        "f1": list(range(10)),
    })
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
    dates = X["date"]
    train_months = set(dates[train_idx].dt.truncate("1mo").to_list())
    val_months = set(dates[val_idx].dt.truncate("1mo").to_list())
    assert val_months == train_months


def test_strat_freq_daily(fast_catboost_params):
    rng = np.random.default_rng(0)
    n = 500
    start = datetime(2023, 1, 1)
    dates = pl.Series([start + timedelta(hours=i) for i in range(n)])
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    X = pl.DataFrame({"date": dates, "f1": f1, "f2": f2})
    y = f1 + rng.normal(0, 0.3, n)
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


# --- generate_with_feature_noise ---

@pytest.fixture
def fitted_gen(sample_data, fast_catboost_params):
    X, y = sample_data
    gen = CalibratedGenerator(
        date_col="date",
        noise="normal",
        random_seed=42,
        catboost_params=fast_catboost_params,
    )
    gen.fit(X, y)
    return gen, X, y


def test_feature_noise_shapes(fitted_gen):
    gen, X, y = fitted_gen
    result = gen.generate_with_feature_noise(
        X,
        feature_noise={
            "f1": {"distribution": "normal", "scale": 0.01},
            "f2": {"distribution": "t", "scale": 0.02, "df": 5},
        },
        n_simulations=30,
    )
    n = len(y)
    assert result.simulations.shape == (30, n)
    assert result.X_simulations["f1"].shape == (30, n)
    assert result.X_simulations["f2"].shape == (30, n)
    assert result.f_X_mean.shape == (n,)


def test_feature_noise_zero_scale(fitted_gen):
    gen, X, _ = fitted_gen
    result = gen.generate_with_feature_noise(
        X,
        feature_noise={
            "f1": {"distribution": "normal", "scale": 0.0},
            "f2": {"distribution": "normal", "scale": 0.0},
            "f3": {"distribution": "normal", "scale": 0.0},
        },
        n_simulations=50,
    )
    # scale=0 → no RNG consumed for feature noise → eps_Y drawn from same state as generate()
    r0 = gen.generate(X, treatment_start="2024-01-01", tau=0.0, n_simulations=50)
    np.testing.assert_allclose(result.simulations, r0.simulations)


def test_feature_noise_invalid_feature(fitted_gen):
    gen, X, _ = fitted_gen
    with pytest.raises(ValueError, match="not found in X"):
        gen.generate_with_feature_noise(
            X,
            feature_noise={"nonexistent_col": {"distribution": "normal", "scale": 0.01}},
            n_simulations=10,
        )


def test_feature_noise_invalid_distribution(fitted_gen):
    gen, X, _ = fitted_gen
    with pytest.raises(ValueError, match="Unknown distribution"):
        gen.generate_with_feature_noise(
            X,
            feature_noise={"f1": {"distribution": "exponential", "scale": 0.01}},
            n_simulations=10,
        )


def test_feature_noise_t_requires_df(fitted_gen):
    gen, X, _ = fitted_gen
    with pytest.raises(ValueError, match="'df' is required"):
        gen.generate_with_feature_noise(
            X,
            feature_noise={"f1": {"distribution": "t", "scale": 0.01}},
            n_simulations=10,
        )


def test_feature_noise_only_specified_features_in_X_simulations(fitted_gen):
    gen, X, _ = fitted_gen
    result = gen.generate_with_feature_noise(
        X,
        feature_noise={"f1": {"distribution": "normal", "scale": 0.01}},
        n_simulations=10,
    )
    assert set(result.X_simulations.keys()) == {"f1"}
    assert "f2" not in result.X_simulations
    assert "f3" not in result.X_simulations


def test_feature_noise_increases_y_variance(fitted_gen):
    gen, X, _ = fitted_gen
    r_small = gen.generate_with_feature_noise(
        X,
        feature_noise={"f1": {"distribution": "normal", "scale": 0.001}},
        n_simulations=200,
    )
    r_large = gen.generate_with_feature_noise(
        X,
        feature_noise={"f1": {"distribution": "normal", "scale": 2.0}},
        n_simulations=200,
    )
    assert r_large.simulations.var() > r_small.simulations.var()
