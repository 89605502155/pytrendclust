import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pytrendclust import Trend, TrendClusterer, cluster_trends
from pytrendclust._cost import LinearCost, fit_line


def trends(result):
    return [s.trend for s in result.segments]


# -- the scenarios that motivate the library ---------------------------------


def test_down_then_up_keeps_only_the_fresh_part():
    result = cluster_trends([10, 9, 8, 7, 6, 5, 4, 5, 6, 7])
    assert result.change_points == [7]
    assert trends(result) == [Trend.DOWN, Trend.UP]
    assert result.indices().tolist() == [7, 8, 9]


def test_picket_fence_is_one_flat_cluster():
    fence = [5, 6] * 5
    result = cluster_trends(fence)
    assert result.n_change_points == 0
    assert trends(result) == [Trend.FLAT]


def test_without_regularisation_the_fence_is_split():
    fence = [5, 6] * 5
    assert cluster_trends(fence, penalty="none").n_change_points > 0


def test_noisy_v_shape_is_found_near_the_true_break():
    rng = np.random.default_rng(1)
    y = np.r_[np.linspace(100, 60, 85), np.linspace(60, 75, 15)] + rng.normal(0, 1.5, 100)
    result = cluster_trends(y)
    assert result.n_change_points == 1
    assert abs(result.change_points[0] - 85) <= 3
    assert result.last_segment.trend is Trend.UP


def test_pure_noise_has_no_change_points_and_is_mostly_flat():
    results = [
        cluster_trends(50 + np.random.default_rng(seed).normal(0, 3, 120)) for seed in range(40)
    ]
    assert all(r.n_change_points == 0 for r in results)
    # "flat" is a significance test at |t| < 2, so ~5 % false "up"/"down" are expected.
    assert sum(r.last_segment.trend is Trend.FLAT for r in results) >= 34


def test_three_regimes():
    rng = np.random.default_rng(7)
    y = np.r_[np.linspace(0, 30, 40), np.full(40, 30.0), np.linspace(30, 0, 40)]
    result = cluster_trends(y + rng.normal(0, 1, 120))
    assert trends(result) == [Trend.UP, Trend.FLAT, Trend.DOWN]


def test_level_shift_between_two_flat_regimes():
    rng = np.random.default_rng(3)
    y = np.r_[np.full(50, 10.0), np.full(30, 40.0)] + rng.normal(0, 1, 80)
    result = cluster_trends(y)
    assert result.change_points == [50]


def test_parabola_is_two_clusters_although_it_needs_more_lines():
    rng = np.random.default_rng(4)
    t = np.arange(80)
    y = 0.02 * (t - 50.0) ** 2 + rng.normal(0, 0.5, 80)
    result = cluster_trends(y)
    assert len(result.pieces) > 2
    assert trends(result) == [Trend.DOWN, Trend.UP]
    assert abs(result.change_points[0] - 50) <= 4


def test_merging_can_be_switched_off():
    rng = np.random.default_rng(4)
    t = np.arange(80)
    y = 0.02 * (t - 50.0) ** 2 + rng.normal(0, 0.5, 80)
    result = cluster_trends(y, merge_same_trend=False)
    assert result.segments == result.pieces
    assert result.n_segments > 2


def test_sine_is_told_apart_from_the_fence():
    rng = np.random.default_rng(0)
    t = np.arange(300)
    sine = cluster_trends(np.sin(2 * np.pi * 5 * t / 300) + rng.normal(0, 0.15, 300))
    directed = [s.trend for s in sine.segments if s.trend is not Trend.FLAT]
    # 5 periods from phase 0: a quarter up, 9 half-periods, a quarter up = 11 runs.
    assert len(directed) == 11
    assert all(a is not b for a, b in itertools.pairwise(directed))
    fence = cluster_trends(np.tile([0.0, 1.0], 150) + rng.normal(0, 0.05, 300))
    assert trends(fence) == [Trend.FLAT]


def test_flat_pieces_are_never_merged():
    y = np.r_[np.full(30, 1.0), np.full(30, 5.0)]
    result = cluster_trends(y + np.random.default_rng(1).normal(0, 0.2, 60))
    assert trends(result) == [Trend.FLAT, Trend.FLAT]


def test_result_is_invariant_to_units():
    rng = np.random.default_rng(5)
    y = np.r_[np.linspace(10, 0, 60), np.linspace(0, 8, 40)] + rng.normal(0, 0.7, 100)
    in_mb = cluster_trends(y)
    in_bytes = cluster_trends(y * 2**20 + 7e9)
    assert in_mb.change_points == in_bytes.change_points


def test_irregular_time_axis():
    t = np.cumsum(np.r_[1.0, np.full(9, 2.0)])
    y = 3.0 - 0.5 * t
    result = cluster_trends(y, t=t)
    assert result.n_change_points == 0
    assert result.last_segment.slope == pytest.approx(-0.5)
    assert result.select()[0].tolist() == t.tolist()


# -- regularisation knobs -----------------------------------------------------


def test_larger_penalty_never_adds_change_points():
    rng = np.random.default_rng(11)
    y = np.cumsum(rng.normal(0, 1, 150))
    counts = [cluster_trends(y, penalty_scale=s).n_change_points for s in (1, 5, 20, 100, 1000)]
    assert counts == sorted(counts, reverse=True)


def test_exponential_penalty_is_stricter_than_linear_at_the_same_scale():
    rng = np.random.default_rng(12)
    y = np.cumsum(rng.normal(0, 1, 200))
    exp_k = cluster_trends(y, penalty="exponential").n_change_points
    lin_k = cluster_trends(y, penalty="linear").n_change_points
    assert exp_k <= lin_k


def test_max_change_points_caps_k():
    y = [5, 6] * 15
    assert cluster_trends(y, penalty="none", max_change_points=2).n_change_points == 2


def test_custom_penalty_callable():
    calls = []

    def huge(k, n, sigma2):
        calls.append((k, n))
        return 1e12 * k

    result = cluster_trends([10, 9, 8, 7, 6, 5, 4, 5, 6, 7], penalty=huge)
    assert result.n_change_points == 0
    assert calls[0] == (0, 10)
    assert result.params["penalty"] == "huge"


def test_explicit_noise_std_overrides_the_estimate():
    result = cluster_trends([10, 9, 8, 7, 6, 5, 4, 5, 6, 7], noise_std=100.0)
    assert result.noise_std == 100.0
    assert result.n_change_points == 0
    assert result.last_segment.trend is Trend.FLAT


def test_min_size_is_respected():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(0, 1, 60))
    result = cluster_trends(y, penalty="none", min_size=5)
    assert min(s.n_points for s in result.segments) >= 5


# -- exactness: dynamic programme vs brute force ----------------------------


def brute_force_objective(y, min_size, penalty):
    n = len(y)
    t = np.arange(n, dtype=float)
    best = math.inf
    for k in range(n):
        for cps in itertools.combinations(range(1, n), k):
            bounds = [0, *cps, n]
            if any(b - a < min_size for a, b in itertools.pairwise(bounds)):
                continue
            sse = sum(fit_line(t[a:b], y[a:b]).sse for a, b in itertools.pairwise(bounds))
            best = min(best, sse + penalty(k))
    return best


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(-20, 20), min_size=4, max_size=11),
    st.integers(2, 4),
    st.sampled_from([0.0, 0.3, 2.0, 8.0]),
)
def test_dynamic_programme_finds_the_global_optimum(values, min_size, scale):
    y = np.asarray(values, dtype=float)
    clusterer = TrendClusterer(min_size=min_size, penalty_scale=scale, noise_std=1.0)
    result = clusterer.fit(y).result_
    expected = brute_force_objective(y, min_size, clusterer._penalty_in_sse_units(len(y), 1.0))
    assert result.objective == pytest.approx(expected, abs=1e-7)
    # The reported objective is the actual cost of the reported segmentation.
    pen = clusterer._penalty_in_sse_units(len(y), 1.0)
    sse = sum(s.sse for s in result.pieces)
    assert result.objective == pytest.approx(sse + pen(len(result.pieces) - 1), abs=1e-7)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=40))
def test_prefix_sum_cost_matches_direct_fit(values):
    y = np.asarray(values)
    t = np.arange(len(y), dtype=float)
    cost = LinearCost(t, y)
    starts = np.arange(len(y) - 1)
    direct = [fit_line(t[s:], y[s:]).sse for s in starts]
    np.testing.assert_allclose(cost.sse(starts, len(y)), direct, atol=1e-6 * (1 + np.ptp(y)) ** 2)


# -- shapes and edge cases ----------------------------------------------------


@pytest.mark.parametrize("y", [[1.0], [1.0, 2.0], [3.0, 3.0, 3.0], [1.0, 5.0, 2.0, 8.0, 0.0]])
def test_short_series_give_one_segment(y):
    result = cluster_trends(y)
    assert result.n_segments == 1
    assert result.labels.tolist() == [0] * len(y)


def test_constant_series_is_flat():
    result = cluster_trends(np.full(30, 4.2))
    assert result.n_change_points == 0
    assert result.last_segment.trend is Trend.FLAT


def test_segments_cover_the_series_without_gaps():
    rng = np.random.default_rng(9)
    result = cluster_trends(np.cumsum(rng.normal(0, 1, 300)))
    bounds = [(s.start, s.stop) for s in result.segments]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 300
    assert all(a[1] == b[0] for a, b in itertools.pairwise(bounds))
    assert len(result.labels) == len(result.trends) == len(result.fitted) == 300


def test_fit_predict_returns_labels():
    labels = TrendClusterer().fit_predict([10, 9, 8, 7, 6, 5, 4, 5, 6, 7])
    assert labels.tolist() == [0] * 7 + [1] * 3


@pytest.mark.parametrize(
    ("y", "t", "message"),
    [
        ([], None, "empty"),
        ([[1, 2], [3, 4]], None, "one-dimensional"),
        ([1, np.nan, 3], None, "NaN"),
        ([1, 2, 3], [0, 1], "same shape"),
        ([1, 2, 3], [0, 2, 1], "strictly increasing"),
        ([1, 2, 3], [0, np.inf, 5], "NaN or infinite"),
    ],
)
def test_bad_input_is_rejected(y, t, message):
    with pytest.raises(ValueError, match=message):
        cluster_trends(y, t=t)


@pytest.mark.parametrize(
    "params",
    [
        {"penalty": "quadratic"},
        {"penalty_scale": -1.0},
        {"penalty_growth": math.nan},
        {"min_size": 1},
        {"max_change_points": -1},
        {"noise_std": -0.1},
        {"flat_threshold": math.inf},
        {"merge_same_trend": 1},
    ],
)
def test_bad_parameters_are_rejected(params):
    with pytest.raises(ValueError, match=next(iter(params))):
        TrendClusterer(**params)


def test_min_size_must_be_an_int():
    with pytest.raises(TypeError):
        TrendClusterer(min_size=2.5)


def test_get_and_set_params():
    clusterer = TrendClusterer(penalty_scale=4.0)
    assert clusterer.get_params()["penalty_scale"] == 4.0
    assert clusterer.set_params(min_size=5) is clusterer
    assert clusterer.min_size == 5
    with pytest.raises(ValueError, match="unknown"):
        clusterer.set_params(window=3)


def test_failed_set_params_leaves_the_clusterer_unchanged():
    clusterer = TrendClusterer()
    with pytest.raises(ValueError, match="min_size"):
        clusterer.set_params(penalty_scale=2.0, min_size=0)
    assert clusterer.penalty_scale is None
    assert clusterer.min_size == 3


def test_repr_round_trips():
    clusterer = TrendClusterer(penalty="linear", min_size=4)
    assert "penalty='linear'" in repr(clusterer)
