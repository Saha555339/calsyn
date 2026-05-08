"""Treatment assignment: build the effect vector τ·D from date-based specifications."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def build_treatment_vector(
    dates: pd.Series,
    treatment_start: str | None = None,
    treatment: list[tuple[str, str, float]] | None = None,
    tau: float | list[float] | None = None,
) -> NDArray | dict[float, NDArray]:
    """Build the additive effect vector(s) aligned to dates.

    Two modes:

    **Mode A — treatment_start + tau:**
        D[i] = 1 for dates[i] >= treatment_start, else 0.
        Returns tau * D for a single tau, or {t: t*D for t in tau} for a grid.

    **Mode B — treatment (list of windows):**
        Each element is (start_date, end_date, tau_window).
        D is zero everywhere; for each window, dates in [start, end] get
        the corresponding tau_window added.
        Returns the combined effect vector.  (tau argument is ignored.)

    Parameters
    ----------
    dates : pd.Series of datetime-like
    treatment_start : str or None
        ISO date string for mode A.
    treatment : list of (start, end, tau) or None
        Date windows with per-window effect sizes for mode B.
    tau : float, list of float, or None
        Effect size(s) for mode A.

    Returns
    -------
    NDArray (single effect vector) or dict[float, NDArray] (tau grid, mode A only)
    """
    dates = pd.to_datetime(dates)

    if treatment is not None:
        return _build_from_windows(dates, treatment)

    if treatment_start is not None:
        return _build_from_start(dates, treatment_start, tau)

    raise ValueError("Provide either treatment_start or treatment.")


def _build_from_start(
    dates: pd.Series, treatment_start: str, tau: float | list[float] | None
) -> NDArray | dict[float, NDArray]:
    if tau is None:
        raise ValueError("tau is required when using treatment_start.")
    start = pd.Timestamp(treatment_start)
    D = (dates >= start).astype(np.float64).values

    if isinstance(tau, (list, tuple, np.ndarray)):
        return {float(t): t * D for t in tau}
    return float(tau) * D


def _build_from_windows(
    dates: pd.Series, treatment: list[tuple[str, str, float]]
) -> NDArray:
    effect = np.zeros(len(dates), dtype=np.float64)
    for start_str, end_str, tau_w in treatment:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        mask = (dates >= start) & (dates <= end)
        effect[mask.values] += tau_w
    return effect
