"""Plotting utilities for calsyn diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def plot_coverage(
    y_real: NDArray,
    simulations: NDArray,
    title: str = "Coverage: real trajectory vs synthetic CI",
    xlabel: str = "Observation index",
    ylabel: str = "Target",
    ax: Any | None = None,
) -> Any:
    """Plot real trajectory against 1σ and 2σ bands from simulations.

    Parameters
    ----------
    y_real : array of shape (n,)
    simulations : array of shape (n_simulations, n)
    title, xlabel, ylabel : str
    ax : matplotlib Axes or None
        If None, a new figure is created.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    y_real = np.asarray(y_real, dtype=np.float64)
    simulations = np.asarray(simulations, dtype=np.float64)

    means = simulations.mean(axis=0)
    stds = simulations.std(axis=0, ddof=1)
    x = np.arange(len(y_real))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(
        x, means - 2 * stds, means + 2 * stds,
        alpha=0.2, color="steelblue", label="±2σ",
    )
    ax.fill_between(
        x, means - stds, means + stds,
        alpha=0.35, color="steelblue", label="±1σ",
    )
    ax.plot(x, means, color="steelblue", linewidth=1, label="Mean synthetic")
    ax.plot(x, y_real, color="black", linewidth=1.2, label="Real")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")

    return ax


def plot_correlation_comparison(
    corr_diagnostics: list,
    feature_names: list[str] | None = None,
    ax: Any | None = None,
) -> Any:
    """Bar chart comparing real vs synthetic feature correlations.

    Parameters
    ----------
    corr_diagnostics : list of CorrelationDiagnostic
    feature_names : list of str or None
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    n = len(corr_diagnostics)
    if feature_names is None:
        feature_names = [f"X{d.feature_index}" for d in corr_diagnostics]

    real_corrs = [d.corr_real for d in corr_diagnostics]
    synth_corrs = [d.corr_synthetic for d in corr_diagnostics]

    x = np.arange(n)
    width = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 5))

    ax.bar(x - width / 2, real_corrs, width, label="Real", color="black", alpha=0.7)
    ax.bar(x + width / 2, synth_corrs, width, label="Synthetic", color="steelblue", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Correlation with target")
    ax.set_title("Feature correlations: real vs synthetic")
    ax.legend()
    ax.axhline(0, color="grey", linewidth=0.5)

    return ax
