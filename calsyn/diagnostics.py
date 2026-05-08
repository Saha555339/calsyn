"""Diagnostics: KS test, feature correlation comparison, coverage analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class KSResult:
    """Kolmogorov-Smirnov test result for real vs synthetic."""

    statistic: float
    pvalue: float


@dataclass
class CorrelationDiagnostic:
    """Per-feature correlation comparison."""

    feature_index: int
    corr_real: float
    corr_synthetic: float
    abs_diff: float


@dataclass
class CoverageDiagnostic:
    """Coverage of real values by synthetic confidence intervals."""

    coverage_1sigma: float  # fraction of real points inside mean ± 1σ
    coverage_2sigma: float  # fraction of real points inside mean ± 2σ
    n_points: int


def ks_test(y_real: NDArray, y_synthetic: NDArray) -> KSResult:
    """Two-sample KS test between real and synthetic target distributions.

    Parameters
    ----------
    y_real : array of shape (n,)
    y_synthetic : array of shape (m,)
        Pooled synthetic values (all simulations flattened) or a single simulation.

    Returns
    -------
    KSResult
    """
    stat, pval = stats.ks_2samp(y_real, y_synthetic)
    return KSResult(statistic=float(stat), pvalue=float(pval))


def feature_correlations(
    X: NDArray, y_real: NDArray, y_synthetic: NDArray
) -> list[CorrelationDiagnostic]:
    """Compare corr(X_j, y_real) vs corr(X_j, y_synthetic) for each feature.

    y_synthetic should have the same length as y_real (e.g. mean across
    simulations at each time step).

    Parameters
    ----------
    X : array of shape (n, p)
    y_real : array of shape (n,)
    y_synthetic : array of shape (n,)

    Returns
    -------
    list of CorrelationDiagnostic, one per feature
    """
    X = np.asarray(X, dtype=np.float64)
    y_real = np.asarray(y_real, dtype=np.float64)
    y_synthetic = np.asarray(y_synthetic, dtype=np.float64)

    results = []
    for j in range(X.shape[1]):
        cr = float(np.corrcoef(X[:, j], y_real)[0, 1])
        cs = float(np.corrcoef(X[:, j], y_synthetic)[0, 1])
        results.append(
            CorrelationDiagnostic(
                feature_index=j,
                corr_real=cr,
                corr_synthetic=cs,
                abs_diff=abs(cr - cs),
            )
        )
    return results


def coverage(
    y_real: NDArray, simulations: NDArray
) -> CoverageDiagnostic:
    """Compute coverage of real trajectory by synthetic confidence intervals.

    Parameters
    ----------
    y_real : array of shape (n,)
        Real target values.
    simulations : array of shape (n_simulations, n)
        Synthetic trajectories.

    Returns
    -------
    CoverageDiagnostic
    """
    y_real = np.asarray(y_real, dtype=np.float64)
    simulations = np.asarray(simulations, dtype=np.float64)

    means = simulations.mean(axis=0)
    stds = simulations.std(axis=0, ddof=1)

    n = len(y_real)
    in_1s = np.sum(np.abs(y_real - means) <= stds)
    in_2s = np.sum(np.abs(y_real - means) <= 2 * stds)

    return CoverageDiagnostic(
        coverage_1sigma=float(in_1s / n),
        coverage_2sigma=float(in_2s / n),
        n_points=n,
    )
