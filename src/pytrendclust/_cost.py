"""Least-squares cost of fitting a straight line to a contiguous run of points.

Prefix sums make the residual sum of squares (SSE) of any run ``[start, stop)``
an O(1) computation, which is what keeps the dynamic programme in
:mod:`pytrendclust._dp` affordable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LineFit:
    """Ordinary least-squares line ``y = intercept + slope * t`` over one run."""

    slope: float
    intercept: float
    sse: float
    #: Centred sum of squares of ``t``; zero when the run has a single point.
    stt: float


def fit_line(t: FloatArray, y: FloatArray) -> LineFit:
    """Fit a straight line to ``(t, y)`` by ordinary least squares."""
    t_mean = float(t.mean())
    y_mean = float(y.mean())
    dt = t - t_mean
    dy = y - y_mean
    stt = float(dt @ dt)
    slope = float(dt @ dy) / stt if stt > 0.0 else 0.0
    intercept = y_mean - slope * t_mean
    residual = dy - slope * dt
    return LineFit(slope=slope, intercept=intercept, sse=float(residual @ residual), stt=stt)


class LinearCost:
    """SSE of the best straight line on any run ``[start, stop)`` of a series."""

    def __init__(self, t: FloatArray, y: FloatArray) -> None:
        # Centring both axes keeps the prefix-sum differences well conditioned.
        tc = t - t.mean()
        yc = y - y.mean()
        zero = np.zeros(1)
        self._n = np.arange(len(y) + 1, dtype=np.float64)
        self._st = np.concatenate([zero, np.cumsum(tc)])
        self._sy = np.concatenate([zero, np.cumsum(yc)])
        self._stt = np.concatenate([zero, np.cumsum(tc * tc)])
        self._sty = np.concatenate([zero, np.cumsum(tc * yc)])
        self._syy = np.concatenate([zero, np.cumsum(yc * yc)])
        #: Total sum of squares of the series around its mean — the cost scale.
        self.total_ss = float(self._syy[-1])

    def sse(self, starts: IntArray, stop: int) -> FloatArray:
        """SSE of the runs ``[s, stop)`` for every ``s`` in ``starts`` (vectorised)."""
        n = self._n[stop] - self._n[starts]
        st = self._st[stop] - self._st[starts]
        sy = self._sy[stop] - self._sy[starts]
        ctt = (self._stt[stop] - self._stt[starts]) - st * st / n
        cty = (self._sty[stop] - self._sty[starts]) - st * sy / n
        cyy = (self._syy[stop] - self._syy[starts]) - sy * sy / n
        explained = np.divide(cty * cty, ctt, out=np.zeros_like(ctt), where=ctt > 0.0)
        sse: FloatArray = np.maximum(cyy - explained, 0.0)
        return sse
