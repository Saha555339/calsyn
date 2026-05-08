# calsyn

**Cal**ibrated **Syn**thetic Data Generator for Monte Carlo simulations.

Generate realistic synthetic trajectories using a piecewise-linear data-generating process calibrated on real data:

```
Y = effect(t) + f(X) + ε
```

where:
- **f(X)** — CatBoost model trained on real features, capturing nonlinear structure
- **effect(t)** — date-based treatment effect (single start date or arbitrary windows with different τ)
- **ε** — noise sampled from a distribution fitted to calibration residuals

## Installation

```bash
pip install calsyn
```

With Optuna hyperparameter tuning:

```bash
pip install "calsyn[tune]"
```

## Quick Start

```python
import pandas as pd
from calsyn import CalibratedGenerator

# X is a DataFrame with a date column and numeric features
# y is the target variable (e.g. log-returns)

gen = CalibratedGenerator(date_col="date", noise="auto")
gen.fit(X, y)

# mode A: effect starts at a given date
result = gen.generate(X, treatment_start="2024-06-01", tau=0.02, n_simulations=500)
result.simulations.shape  # (500, n_samples)
```

## Input Format

`X` must be a `pd.DataFrame` with:
- a datetime column (name passed via `date_col`)
- numeric feature columns

```python
X = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=500, freq="B"),
    "GAZP": gazp_log_returns,
    "LKOH": lkoh_log_returns,
    "USDRUB": usdrub_log_returns,
})
y = sber_log_returns  # target
```

## Treatment Modes

### Mode A — single start date

Everything after `treatment_start` gets effect τ. Supports tau grid for sweep.

```python
# single tau
result = gen.generate(X, treatment_start="2024-06-01", tau=0.02)

# tau grid — returns dict[float, GenerationResult]
results = gen.generate(
    X, treatment_start="2024-06-01",
    tau=[0.0, 0.01, 0.02, 0.05],
    n_simulations=500,
)
for tau_val, res in results.items():
    print(f"tau={tau_val}: mean={res.simulations.mean():.4f}")
```

### Mode B — arbitrary windows with different effects

Each window is `(start_date, end_date, tau)`. Dates are inclusive. Outside all windows, effect is zero.

```python
result = gen.generate(
    X,
    treatment=[
        ("2024-03-01", "2024-03-15", 0.03),   # positive shock in early March
        ("2024-06-01", "2024-06-10", -0.01),   # negative shock in June
        ("2024-09-01", "2024-09-30", 0.05),    # larger effect in September
    ],
    n_simulations=500,
)
```

## Validation

The model is trained on ~90% of data (configurable via `val_fraction`). The held-out set is **stratified by time period** — from each period, `val_fraction` of observations go to OOS.

Stratification period is set via `strat_freq`:

| `strat_freq` | Period | Use case |
|---|---|---|
| `"M"` | Month | Daily/weekly financial data (default) |
| `"W"` | Week | Daily data with short history |
| `"d"` | Day | Intraday (hourly/minute) data |
| `"h"` | Hour | Sub-minute tick data |
| `"m"` | Minute | High-frequency tick data |

```python
# daily data — stratify by month (default)
gen = CalibratedGenerator(date_col="date", noise="auto", val_fraction=0.1)

# intraday data — stratify by day
gen = CalibratedGenerator(date_col="date", noise="auto", val_fraction=0.1, strat_freq="d")

gen.fit(X, y)
print(gen.oos_metrics)
# {'r2': 0.82, 'rmse': 0.017, 'bias': -0.0003}
```

## Noise Distribution

Three modes for the residual noise ε:

```python
# auto: pick normal or t by BIC on residuals
gen = CalibratedGenerator(date_col="date", noise="auto")

# force Gaussian
gen = CalibratedGenerator(date_col="date", noise="normal")

# force Student-t (heavier tails)
gen = CalibratedGenerator(date_col="date", noise="t")
```

After fitting, inspect the chosen distribution:

```python
gen.fit(X, y)
print(gen.noise_params)
# {'distribution': 't', 'df': 4.2, 'loc': 0.001, 'scale': 0.012}
```

## Diagnostics

```python
result_h0 = gen.generate(X, treatment_start="2024-06-01", tau=0.0, n_simulations=500)
report = gen.diagnose(result_h0)

print(report.oos_metrics)
# {'r2': 0.82, 'rmse': 0.017, 'bias': -0.0003}

print(report.ks)
# KSResult(statistic=0.04, pvalue=0.72)

print(report.coverage)
# CoverageDiagnostic(coverage_1sigma=0.68, coverage_2sigma=0.95, n_points=500)

for c in report.correlations:
    print(f"  X{c.feature_index}: real={c.corr_real:.3f} synth={c.corr_synthetic:.3f}")
```

## Plots

```python
# coverage plot: real trajectory vs 1σ/2σ bands
gen.plot_coverage(result_h0)

# feature correlation comparison
gen.plot_correlations(result_h0, feature_names=["GAZP", "LKOH", "USDRUB"])
```

## Optuna Tuning

```python
gen = CalibratedGenerator(date_col="date", noise="auto", auto_tune=True, n_trials=100)
gen.fit(X, y)
```

## API Reference

### `CalibratedGenerator`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date_col` | `str` | `"date"` | Name of datetime column in X |
| `noise` | `"auto" \| "normal" \| "t"` | `"auto"` | Noise distribution |
| `auto_tune` | `bool` | `False` | Optuna hyperparameter search |
| `n_trials` | `int` | `50` | Optuna trials |
| `val_fraction` | `float` | `0.1` | OOS fraction per time period |
| `strat_freq` | `str` | `"M"` | Stratification: "M", "W", "d", "h", "m" |
| `catboost_params` | `dict \| None` | `None` | Custom CatBoost params |
| `random_seed` | `int` | `42` | Random seed |

**Methods:**

- `.fit(X, y)` — fit calibration model (train-only) and noise distribution
- `.generate(X, treatment_start=..., tau=..., n_simulations=...)` — mode A
- `.generate(X, treatment=[...], n_simulations=...)` — mode B
- `.diagnose(result)` — KS, correlations, coverage, OOS metrics
- `.plot_coverage(result)` — coverage plot with 1σ/2σ bands
- `.plot_correlations(result, feature_names=...)` — correlation bar chart
- `.oos_metrics` — dict with R², RMSE, bias
- `.noise_params` — fitted noise distribution parameters

## License

MIT
