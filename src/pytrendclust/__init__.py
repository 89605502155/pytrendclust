"""Trend clustering of time series with an adaptive number of clusters.

Splits a series into consecutive rising, falling and flat runs, choosing the
number of change points by a penalised least-squares criterion whose penalty
grows exponentially with every extra change point.

>>> from pytrendclust import cluster_trends
>>> result = cluster_trends([10, 9, 8, 7, 6, 5, 4, 5, 6, 7])
>>> result.last_segment.trend.value, result.indices().tolist()
('up', [7, 8, 9])
"""

from pytrendclust.clusterer import TrendClusterer, cluster_trends
from pytrendclust.penalty import (
    PenaltyFunc,
    default_scale,
    estimate_noise_std,
    exponential_penalty,
    linear_penalty,
)
from pytrendclust.result import CriterionRow, Segment, Trend, TrendClusteringResult

__all__ = [
    "CriterionRow",
    "PenaltyFunc",
    "Segment",
    "Trend",
    "TrendClusterer",
    "TrendClusteringResult",
    "cluster_trends",
    "default_scale",
    "estimate_noise_std",
    "exponential_penalty",
    "linear_penalty",
]

__version__ = "0.1.0"
