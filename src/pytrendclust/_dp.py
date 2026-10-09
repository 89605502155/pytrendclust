"""Exact penalised segmentation by dynamic programming over the number of segments.

For every number of change points ``K`` the programme finds the partition of the
series into ``K + 1`` runs with the smallest total SSE (the "segment
neighbourhood" recursion).  The penalty may be any non-decreasing function of
``K`` — in particular an exponential one, which an additive per-segment
recursion such as PELT cannot express.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pytrendclust._cost import IntArray, LinearCost
from pytrendclust.result import CriterionRow


@dataclass(frozen=True, slots=True)
class Segmentation:
    change_points: tuple[int, ...]
    criterion: tuple[CriterionRow, ...]


def segment(
    cost: LinearCost,
    n: int,
    min_size: int,
    max_change_points: int,
    penalty: Callable[[int], float],
) -> Segmentation:
    """Return the change points minimising ``SSE(K) + penalty(K)``.

    Change points are indices ``c`` such that a new segment starts at ``c``.
    The search over ``K`` stops early once ``penalty(K)`` alone exceeds the best
    objective seen so far: SSE is non-negative and the penalty non-decreasing,
    so no larger ``K`` can win.
    """
    inf = np.inf
    all_stops = np.arange(n + 1)
    # best[j]: minimal SSE of splitting the prefix [0, j) into k + 1 segments.
    best = np.full(n + 1, inf)
    best[min_size:] = [cost.sse(np.array([0]), j)[0] for j in all_stops[min_size:]]
    back: list[IntArray] = []
    tie_tol = 1e-10 * cost.total_ss

    criterion = [CriterionRow(0, float(best[n]), penalty(0))]
    best_k, best_obj = 0, criterion[0].objective

    for k in range(1, max_change_points + 1):
        pen = penalty(k)
        if pen >= best_obj:
            break
        new = np.full(n + 1, inf)
        arg = np.zeros(n + 1, dtype=np.int64)
        for j in range((k + 1) * min_size, n + 1):
            starts = np.arange(k * min_size, j - min_size + 1)
            total = best[starts] + cost.sse(starts, j)
            # On ties take the latest start: a point lying on both lines (the
            # vertex of a V) stays with the older regime, keeping the newest
            # segment free of ambiguous points.
            i = int(np.flatnonzero(total <= total.min() + tie_tol)[-1])
            new[j] = total[i]
            arg[j] = starts[i]
        best = new
        back.append(arg)
        point = CriterionRow(k, float(best[n]), pen)
        criterion.append(point)
        # Strict improvement beyond rounding noise: ties go to fewer change points.
        if point.objective < best_obj - tie_tol:
            best_k, best_obj = k, point.objective

    change_points: list[int] = []
    j = n
    for k in range(best_k, 0, -1):
        j = int(back[k - 1][j])
        change_points.append(j)
    return Segmentation(tuple(reversed(change_points)), tuple(criterion))
