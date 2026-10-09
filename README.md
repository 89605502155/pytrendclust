# pytrendclust

[![PyPI](https://img.shields.io/pypi/v/pytrendclust)](https://pypi.org/project/pytrendclust/)
[![Python](https://img.shields.io/pypi/pyversions/pytrendclust)](https://pypi.org/project/pytrendclust/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/89605502155/pytrendclust/blob/main/LICENSE)

Source code: <https://github.com/89605502155/pytrendclust>

Trend clustering of time series with an **adaptive number of clusters**.

`pytrendclust` splits a series into consecutive runs that rise, fall or stay
flat. You do not pass the number of clusters. It is chosen by a penalised
least-squares criterion in which every extra change point costs exponentially
more than the previous one. Real changes of direction are found, and noise is
not mistaken for a trend.

Typical use: keep only the most recent regime of a series as the training set
for a regression model. For example, a series of RAM consumption first falls
and then grows. Fitting a line to the whole history mixes two regimes; fitting
it to the last cluster does not.

```python
from pytrendclust import TrendClusterer

y = [10, 9, 8, 7, 6, 5, 4, 5, 6, 7]

result = TrendClusterer().fit(y).result_
print(result.summary())
# 10 points, 1 change point(s), noise std 0.27
#   #0: [0, 7)    7 pts  down  slope -1
#   #1: [7, 10)    3 pts  up    slope +1

t_recent, y_recent = result.select()       # points of the last (newest) cluster
result.export("result.json")               # or "points.csv"
```

A "picket fence" `5, 6, 5, 6, ...` is one flat cluster, not ten tiny trends:

```python
>>> from pytrendclust import cluster_trends
>>> cluster_trends([5, 6] * 5).summary()
'10 points, 0 change point(s), noise std 1.211\n  #0: [0, 10)   10 pts  flat  slope +0.0303'
>>> cluster_trends([5, 6] * 5, penalty="none").n_change_points   # no regularisation
2

```

## Installation

```bash
pip install pytrendclust            # numpy only
pip install "pytrendclust[pandas]"  # + DataFrame export
```

## Method

For every possible number of change points `K`, a dynamic programme finds the
exact partition into `K + 1` runs with the smallest total squared error of
per-run least-squares lines, `SSE(K)`. The clusterer then picks

```
K* = argmin_K  SSE(K) + sigma² · pen(K)
pen(K) = c · (exp(g·K) − 1) / (exp(g) − 1)
```

* `c` (`penalty_scale`): price of the first change point in noise variances.
  The default is `5·ln(n)`.
* `g` (`penalty_growth`, default 0.5): each next change point costs `e^g`
  times more than the previous one. `g = 0` gives the classic linear penalty.
* `sigma` (`noise_std`): noise level. By default it is estimated robustly
  from second differences of the series, which cancel any linear trend. This
  makes the result independent of units: bytes and gigabytes give the same
  clusters.

The search over `K` stops as soon as the penalty alone exceeds the best value
found. The exponential penalty makes this happen after a few steps. The
solution is the global optimum of the criterion; the test suite checks it
against brute-force enumeration.

Each cluster is labelled `up`, `down` or `flat` with a t-test on its slope:
`|slope| / se(slope) < flat_threshold` (default 2) means `flat`.

## API

| Object | Purpose |
|---|---|
| `TrendClusterer(**params)` | set parameters; `fit(y, t=None)`, `fit_predict`, `get_params`, `set_params` |
| `cluster_trends(y, t=None, **params)` | one-call shortcut returning the result |
| `TrendClusteringResult` | `segments`, `change_points`, `labels`, `trends`, `fitted`, `last_segment`, `select()`, `indices()`, `criterion` |
| export | `to_dict()`, `to_json(path)`, `to_csv(path)`, `to_dataframe()`, `export(path)`; `from_dict`, `from_json` |

Parameters: `penalty` (`"exponential"`, `"linear"`, `"none"`, or a callable
`f(K, n, sigma²)`), `penalty_scale`, `penalty_growth`, `min_size` (default 3),
`max_change_points`, `noise_std`, `flat_threshold`.

`result.criterion` lists `SSE(K)`, the penalty and their sum for every `K` the
search evaluated. Use it to justify the chosen number of clusters.

Complexity is `O(K · n²)` time and `O(K · n)` memory: about 0.4 s for 1000
points and 6 s for 5000.

## Кратко по-русски

Трендовая кластеризация временного ряда с адаптивным числом классов. Ряд
делится на участки роста, спада и плато. Число точек разладки выбирается
автоматически: к ошибке модели добавляется штраф, растущий экспоненциально с
каждой новой точкой. Так шумовые колебания («заборчик») не принимаются за смену
тренда. `result.select()` возвращает точки последнего, самого свежего участка.
На них обучается регрессия.

## Development

```bash
git clone https://github.com/89605502155/pytrendclust.git
cd pytrendclust
uv sync --all-extras
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
uv build
```

## License

MIT, see [LICENSE](https://github.com/89605502155/pytrendclust/blob/main/LICENSE).
