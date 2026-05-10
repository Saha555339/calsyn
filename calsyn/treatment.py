"""Treatment assignment: build the effect vector τ·D from date-based specifications."""

from __future__ import annotations

from datetime import date as _date

import numpy as np
import polars as pl
from numpy.typing import NDArray


def build_treatment_vector(
    dates: pl.Series,
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
        treatment = [(start_date, end_date, tau_window), ...].
        D is zero everywhere; for each window, dates in [start, end] get
        the corresponding tau_window added.
        Returns the combined effect vector.  (tau argument is ignored.)

    Parameters
    ----------
    dates : pl.Series of Date or Datetime
    treatment_start : str or None
        ISO date string for mode A (e.g. "2024-06-01").
    treatment : list of (start, end, tau) or None
        Date windows with per-window effect sizes for mode B.
    tau : float, list of float, or None
        Effect size(s) for mode A.

    Returns
    -------
    NDArray (single effect vector) or dict[float, NDArray] (tau grid, mode A only)
    """
    dates = _as_date(dates)

    if treatment is not None:
        return _build_from_windows(dates, treatment)

    if treatment_start is not None:
        return _build_from_start(dates, treatment_start, tau)

    raise ValueError("Provide either treatment_start or treatment.")


def _as_date(s: pl.Series) -> pl.Series:
    """Normalize a polars Series to pl.Date."""
    if s.dtype == pl.Date:
        return s
    if s.dtype in (pl.Utf8, pl.String):
        return s.str.to_date()
    # Datetime (with or without timezone) → strip time component
    return s.dt.date()


def _build_from_start(
    dates: pl.Series, treatment_start: str, tau: float | list[float] | None
) -> NDArray | dict[float, NDArray]:
    if tau is None:
        raise ValueError("tau is required when using treatment_start.")
    start = _date.fromisoformat(treatment_start[:10])
    D = (dates >= start).cast(pl.Float64).to_numpy()

    if isinstance(tau, (list, tuple, np.ndarray)):
        return {float(t): t * D for t in tau}
    return float(tau) * D


def _build_from_windows(
    dates: pl.Series, treatment: list[tuple[str, str, float]]
) -> NDArray:
    effect = np.zeros(len(dates), dtype=np.float64)
    for start_str, end_str, tau_w in treatment:
        start = _date.fromisoformat(start_str[:10])
        end = _date.fromisoformat(end_str[:10])
        mask = ((dates >= start) & (dates <= end)).to_numpy()
        effect[mask] += tau_w
    return effect
