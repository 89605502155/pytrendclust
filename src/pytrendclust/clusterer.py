"""Trend clustering of a time series with an adaptive number of clusters."""

from __future__ import annotations

import itertools
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytrendclust._cost import LinearCost, fit_line
from pytrendclust._dp import segment
from pytrendclust.penalty import (
    PenaltyFunc,
    default_scale,
    estimate_noise_std,
    exponential_penalty,
    linear_penalty,
)
from pytrendclust.result import Segment, Trend, TrendClusteringResult

PenaltyKind = Literal["exponential", "linear", "none"]

# Floor of the noise scale relative to the series' spread: on a perfectly clean
# series the noise estimate is 0, and a zero penalty would make every
# segmentation with zero SSE a tie.
_NOISE_FLOOR = 1e-6


class TrendClusterer:
    """Split a time series into consecutive trend clusters (rising, falling, flat).

    The series is first cut into linear pieces, each with its own least-squares
    line.  The number of pieces is not given in advance: the clusterer minimises

        SSE(K) + sigma**2 * penalty(K)

    over the number of change points ``K``, where ``SSE(K)`` is the best total
    squared error of a piecewise-linear fit with ``K`` change points and
    ``sigma`` is the noise level.  The default exponential penalty makes every
    extra change point dearer than the previous one, so noise such as a
    "picket fence" (up, down, up, down, ...) is recognised as a single flat
    cluster instead of ten tiny trends.

    Every piece is then labelled rising, falling or flat, and adjacent pieces
    with the same direction are merged into one trend cluster: a cluster is a
    maximal run with one character of monotonicity.  A noisy parabola thus
    gives two clusters, even though its curved halves need several lines.
    Flat pieces are never merged, so a step between two plateaus stays a
    boundary.

    Parameters
    ----------
    penalty:
        ``"exponential"`` (default) — the ``K``-th change point costs
        ``penalty_scale * exp(penalty_growth * (K - 1))`` noise variances;
        ``"linear"`` — every change point costs ``penalty_scale``;
        ``"none"`` — no regularisation (only ``max_change_points`` limits K);
        or a callable ``f(n_change_points, n_samples, noise_variance)`` returning
        the penalty in the units of SSE, non-decreasing in ``n_change_points``.
    penalty_scale:
        Price of the first change point in noise variances.  Defaults to
        ``6 * ln(n)`` (see :func:`~pytrendclust.default_scale`).  Larger means
        fewer change points.
    penalty_growth:
        Exponential growth rate of the price of every next change point: the
        ``K``-th one costs ``exp(penalty_growth * (K - 1))`` times the first.
    min_size:
        Minimum number of points in a cluster; at least 2 (a line needs two
        points, and three are needed before a line can be wrong).
    max_change_points:
        Hard upper bound on ``K``; ``None`` means only the penalty limits it.
    noise_std:
        Noise standard deviation in the units of the series.  ``None`` estimates
        it robustly from second differences of the series.
    flat_threshold:
        A cluster is ``flat`` when ``|slope| / standard_error(slope)`` is below
        this value (2 is roughly the 95 % significance level).
    merge_same_trend:
        Merge adjacent rising (falling) pieces into one cluster, so that cluster
        boundaries are exactly the changes of monotonicity.  ``False`` keeps
        every linear piece as a cluster of its own.

    Examples
    --------
    >>> y = [10, 9, 8, 7, 6, 5, 4, 5, 6, 7]
    >>> result = TrendClusterer().fit(y).result_
    >>> result.change_points
    [7]
    >>> [s.trend.value for s in result.segments]
    ['down', 'up']
    >>> result.select()[1]
    array([5., 6., 7.])
    """

    def __init__(
        self,
        *,
        penalty: PenaltyKind | PenaltyFunc = "exponential",
        penalty_scale: float | None = None,
        penalty_growth: float = 0.1,
        min_size: int = 3,
        max_change_points: int | None = None,
        noise_std: float | None = None,
        flat_threshold: float = 2.0,
        merge_same_trend: bool = True,
    ) -> None:
        self.penalty = penalty
        self.penalty_scale = penalty_scale
        self.penalty_growth = penalty_growth
        self.min_size = min_size
        self.max_change_points = max_change_points
        self.noise_std = noise_std
        self.flat_threshold = flat_threshold
        self.merge_same_trend = merge_same_trend
        self._validate()
        self.result_: TrendClusteringResult | None = None

    # -- parameters --------------------------------------------------------

    _PARAM_NAMES = (
        "penalty",
        "penalty_scale",
        "penalty_growth",
        "min_size",
        "max_change_points",
        "noise_std",
        "flat_threshold",
        "merge_same_trend",
    )

    def get_params(self) -> dict[str, Any]:
        """Current parameters, as accepted by the constructor and :meth:`set_params`."""
        return {name: getattr(self, name) for name in self._PARAM_NAMES}

    def set_params(self, **params: Any) -> TrendClusterer:
        """Change parameters in place; returns ``self`` for chaining."""
        unknown = set(params) - set(self._PARAM_NAMES)
        if unknown:
            msg = f"unknown parameter(s): {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        old = self.get_params()
        for name, value in params.items():
            setattr(self, name, value)
        try:
            self._validate()
        except (TypeError, ValueError):
            for name, value in old.items():
                setattr(self, name, value)
            raise
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _validate(self) -> None:
        if not callable(self.penalty) and self.penalty not in ("exponential", "linear", "none"):
            msg = (
                "penalty must be 'exponential', 'linear', 'none' or a callable, "
                f"got {self.penalty!r}"
            )
            raise ValueError(msg)
        if self.penalty_scale is not None and not (
            math.isfinite(self.penalty_scale) and self.penalty_scale >= 0
        ):
            msg = f"penalty_scale must be a finite number >= 0, got {self.penalty_scale!r}"
            raise ValueError(msg)
        if not (math.isfinite(self.penalty_growth) and self.penalty_growth >= 0):
            msg = f"penalty_growth must be a finite number >= 0, got {self.penalty_growth!r}"
            raise ValueError(msg)
        if isinstance(self.min_size, bool) or not isinstance(self.min_size, int):
            msg = f"min_size must be an int, got {self.min_size!r}"
            raise TypeError(msg)
        if self.min_size < 2:
            msg = f"min_size must be >= 2, got {self.min_size}"
            raise ValueError(msg)
        if self.max_change_points is not None and (
            isinstance(self.max_change_points, bool)
            or not isinstance(self.max_change_points, int)
            or self.max_change_points < 0
        ):
            msg = f"max_change_points must be None or an int >= 0, got {self.max_change_points!r}"
            raise ValueError(msg)
        if self.noise_std is not None and not (
            math.isfinite(self.noise_std) and self.noise_std >= 0
        ):
            msg = f"noise_std must be None or a finite number >= 0, got {self.noise_std!r}"
            raise ValueError(msg)
        if not (math.isfinite(self.flat_threshold) and self.flat_threshold >= 0):
            msg = f"flat_threshold must be a finite number >= 0, got {self.flat_threshold!r}"
            raise ValueError(msg)
        if not isinstance(self.merge_same_trend, bool):
            msg = f"merge_same_trend must be a bool, got {self.merge_same_trend!r}"
            raise ValueError(msg)

    # -- fitting -----------------------------------------------------------

    def fit(self, y: ArrayLike, t: ArrayLike | None = None) -> TrendClusterer:
        """Cluster the series ``y`` observed at times ``t`` (default ``0, 1, 2, ...``).

        The result is stored in :attr:`result_`; returns ``self``.
        """
        y_arr, t_arr = _as_series(y, t)
        n = len(y_arr)

        sigma = self._noise_std(y_arr)
        sigma2 = sigma * sigma
        penalty = self._penalty_in_sse_units(n, sigma2)
        largest_k = n // self.min_size - 1
        if self.max_change_points is not None:
            largest_k = min(largest_k, self.max_change_points)

        seg = segment(LinearCost(t_arr, y_arr), n, self.min_size, max(largest_k, 0), penalty)

        bounds = [0, *seg.change_points, n]
        pieces = tuple(
            self._describe(index=i, start=start, stop=stop, t=t_arr, y=y_arr, sigma=sigma)
            for i, (start, stop) in enumerate(itertools.pairwise(bounds))
        )
        segments = self._merge(pieces, t_arr, y_arr, sigma) if self.merge_same_trend else pieces
        self.result_ = TrendClusteringResult(
            t=t_arr,
            y=y_arr,
            segments=segments,
            pieces=pieces,
            noise_std=sigma,
            criterion=seg.criterion,
            params=self._exportable_params(),
        )
        return self

    def fit_predict(self, y: ArrayLike, t: ArrayLike | None = None) -> NDArray[np.int64]:
        """Fit and return the cluster (segment) index of every point."""
        result = self.fit(y, t).result_
        assert result is not None  # noqa: S101 - set by fit()
        return result.labels

    def _noise_std(self, y: NDArray[np.float64]) -> float:
        sigma = estimate_noise_std(y) if self.noise_std is None else float(self.noise_std)
        spread = float(np.ptp(y)) if len(y) else 0.0
        return max(sigma, _NOISE_FLOOR * spread)

    def _penalty_in_sse_units(self, n: int, sigma2: float) -> Any:
        if callable(self.penalty):
            func = self.penalty
            return lambda k: float(func(k, n, sigma2))
        if self.penalty == "none":
            return lambda k: 0.0
        scale = default_scale(n) if self.penalty_scale is None else self.penalty_scale
        if self.penalty == "linear":
            return lambda k: sigma2 * linear_penalty(k, scale)
        growth = self.penalty_growth
        return lambda k: sigma2 * exponential_penalty(k, scale, growth)

    def _merge(
        self,
        pieces: tuple[Segment, ...],
        t: NDArray[np.float64],
        y: NDArray[np.float64],
        sigma: float,
    ) -> tuple[Segment, ...]:
        """Join adjacent rising (or falling) pieces into one trend cluster each."""
        groups: list[list[Segment]] = []
        for piece in pieces:
            last = groups[-1][-1] if groups else None
            if last is not None and piece.trend is last.trend and piece.trend is not Trend.FLAT:
                groups[-1].append(piece)
            else:
                groups.append([piece])
        return tuple(
            self._describe(
                index=i,
                start=group[0].start,
                stop=group[-1].stop,
                t=t,
                y=y,
                sigma=sigma,
                trend=group[0].trend,
            )
            for i, group in enumerate(groups)
        )

    def _describe(
        self,
        *,
        index: int,
        start: int,
        stop: int,
        t: NDArray[np.float64],
        y: NDArray[np.float64],
        sigma: float,
        trend: Trend | None = None,
    ) -> Segment:
        """Line fit and trend of ``[start, stop)``; ``trend`` overrides the t-test."""
        line = fit_line(t[start:stop], y[start:stop])
        if line.stt == 0.0 or line.slope == 0.0:
            t_stat = 0.0
        elif sigma == 0.0:
            t_stat = math.inf
        else:
            t_stat = abs(line.slope) * math.sqrt(line.stt) / sigma
        if trend is None:
            if t_stat < self.flat_threshold:
                trend = Trend.FLAT
            else:
                trend = Trend.UP if line.slope > 0 else Trend.DOWN
        return Segment(
            index=index,
            start=start,
            stop=stop,
            trend=trend,
            slope=line.slope,
            intercept=line.intercept,
            sse=line.sse,
            t_statistic=t_stat,
        )

    def _exportable_params(self) -> dict[str, Any]:
        params = self.get_params()
        if callable(params["penalty"]):
            params["penalty"] = getattr(params["penalty"], "__name__", repr(params["penalty"]))
        return params


def cluster_trends(
    y: ArrayLike, t: ArrayLike | None = None, **params: Any
) -> TrendClusteringResult:
    """One-call shortcut: ``TrendClusterer(**params).fit(y, t).result_``."""
    result = TrendClusterer(**params).fit(y, t).result_
    assert result is not None  # noqa: S101 - set by fit()
    return result


def _as_series(
    y: ArrayLike, t: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_arr = np.array(y, dtype=np.float64)
    if y_arr.ndim != 1:
        msg = f"y must be one-dimensional, got shape {y_arr.shape}"
        raise ValueError(msg)
    if len(y_arr) == 0:
        msg = "y is empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(y_arr)):
        msg = "y contains NaN or infinite values"
        raise ValueError(msg)
    if t is None:
        return y_arr, np.arange(len(y_arr), dtype=np.float64)
    t_arr = np.array(t, dtype=np.float64)
    if t_arr.shape != y_arr.shape:
        msg = f"t and y must have the same shape, got {t_arr.shape} and {y_arr.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(t_arr)):
        msg = "t contains NaN or infinite values"
        raise ValueError(msg)
    if np.any(np.diff(t_arr) <= 0):
        msg = "t must be strictly increasing"
        raise ValueError(msg)
    return y_arr, t_arr
