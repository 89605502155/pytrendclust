"""Train a regression only on the most recent trend cluster of a RAM-usage series.

The series falls for 85 steps and then grows for 15.  A line fitted to the whole
history points down; a line fitted to the last cluster follows the current growth.
"""

import numpy as np

from pytrendclust import TrendClusterer

rng = np.random.default_rng(0)
ram_mb = np.r_[np.linspace(4000, 2500, 85), np.linspace(2500, 3100, 15)]
ram_mb += rng.normal(0, 30, ram_mb.size)

result = TrendClusterer().fit(ram_mb).result_
print(result.summary())

t, y = result.select()  # only the newest regime
next_t = len(ram_mb)
whole = np.polyval(np.polyfit(np.arange(len(ram_mb)), ram_mb, 1), next_t)
recent = np.polyval(np.polyfit(t, y, 1), next_t)
print(f"forecast for t={next_t}: whole history {whole:.0f} MB, last cluster {recent:.0f} MB")

result.export("ram_clusters.json")
