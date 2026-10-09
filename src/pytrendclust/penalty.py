"""Penalties on the number of change points and the noise estimate they are scaled by.

A penalty is measured in units of the noise variance ``sigma**2``: SSE grows with
the square of the series' units, so a penalty in the same units makes the
trade-off independent of whether memory is counted in bytes or gigabytes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

#: ``penalty(n_change_points, n_samples, noise_variance) -> float``.
#: A custom penalty must be non-decreasing in ``n_change_points``.
PenaltyFunc = Callable[[int, int, float], float]

# Consistency constant turning a median absolute deviation into a standard
# deviation for Gaussian noise.
_MAD_TO_STD = 1.482602218505602


def default_scale(n_samples: int) -> float:
    """Default price of the first change point, in noise variances: ``6 * ln(n)``.

    BIC would charge ``ln(n)`` per parameter.  A change point is dearer than a
    parameter: its position is the best of ``n`` candidates, and a monotone fit
    adapts to noise more than a line does.  The constant 6 was chosen on
    simulated series (see the accompanying article): with it a spurious change
    point appears on pure noise in about 2 % of 100-point series, while noisy
    parabolas, V-shapes and five-period sine waves are segmented correctly.
    """
    return 6.0 * math.log(max(n_samples, 2))


def exponential_penalty(n_change_points: int, scale: float, growth: float) -> float:
    """``scale * (exp(growth * K) - 1) / (exp(growth) - 1)``, in noise variances.

    The first change point costs exactly ``scale``; every next one costs
    ``exp(growth)`` times more than the previous, so the model grows ever more
    reluctant to add change points.  ``growth = 0`` degenerates to the linear
    penalty ``scale * K``.
    """
    if n_change_points == 0:
        return 0.0
    if growth == 0.0:
        return scale * n_change_points
    return scale * math.expm1(growth * n_change_points) / math.expm1(growth)


def linear_penalty(n_change_points: int, scale: float) -> float:
    """Classic constant price per change point: ``scale * K``, in noise variances."""
    return scale * n_change_points


def estimate_noise_std(y: NDArray[np.float64]) -> float:
    """Robust estimate of the noise standard deviation around a piecewise-linear trend.

    Second differences cancel any linear trend, leaving ``eps[i] - 2 eps[i+1] +
    eps[i+2]`` with variance ``6 sigma**2``; a few kinks between segments are
    outliers that the median absolute deviation ignores.  Falls back to the plain
    standard deviation of the second differences when more than half of them
    coincide (for example, on a quantised series), and returns 0 for series too
    short to tell.
    """
    if len(y) < 4:
        return 0.0
    d2 = np.diff(y, n=2)
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    if mad > 0.0:
        return _MAD_TO_STD * mad / math.sqrt(6.0)
    return float(d2.std()) / math.sqrt(6.0)
