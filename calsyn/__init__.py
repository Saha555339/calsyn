"""calsyn — Calibrated Synthetic Data Generator.

Generate Monte Carlo trajectories of a target variable using a piecewise-linear
data-generating process:  Y = τ·D + f(X) + ε, where f is a CatBoost model
calibrated on real data.
"""

from calsyn._version import __version__
from calsyn.generator import (
    CalibratedGenerator,
    GenerationResult,
    DiagnosticReport,
    FeatureNoiseResult,
)

__all__ = [
    "CalibratedGenerator",
    "GenerationResult",
    "DiagnosticReport",
    "FeatureNoiseResult",
    "__version__",
]
