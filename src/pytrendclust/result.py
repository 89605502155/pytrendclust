"""Result of a trend clustering and its export to dict, JSON, CSV and pandas."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

Level = Literal["cluster", "piece"]

FORMAT_VERSION = 2
_READABLE_VERSIONS = (1, 2)


class Trend(StrEnum):
    """Direction of a segment's linear trend."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class Segment:
    """One trend cluster: a contiguous run of points ``[start, stop)`` and its line."""

    index: int
    start: int
    stop: int
    trend: Trend
    slope: float
    intercept: float
    sse: float
    #: ``|slope| / standard_error(slope)``; ``inf`` for a noise-free series.
    t_statistic: float

    @property
    def n_points(self) -> int:
        return self.stop - self.start

    def predict(self, t: Any) -> NDArray[np.float64]:
        """Value of the segment's line at time(s) ``t`` — also beyond the segment."""
        return self.intercept + self.slope * np.asarray(t, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "stop": self.stop,
            "n_points": self.n_points,
            "trend": self.trend.value,
            "slope": self.slope,
            "intercept": self.intercept,
            "sse": self.sse,
            "t_statistic": _json_float(self.t_statistic),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(
            index=int(data["index"]),
            start=int(data["start"]),
            stop=int(data["stop"]),
            trend=Trend(data["trend"]),
            slope=float(data["slope"]),
            intercept=float(data["intercept"]),
            sse=float(data["sse"]),
            t_statistic=float(data["t_statistic"]),
        )


@dataclass(frozen=True, slots=True)
class CriterionRow:
    """Penalised criterion for one candidate number of change points."""

    n_change_points: int
    sse: float
    penalty: float

    @property
    def objective(self) -> float:
        return self.sse + self.penalty


@dataclass(frozen=True)
class TrendClusteringResult:
    """Which point belongs to which trend cluster, plus everything needed to audit it.

    ``segments`` are the trend clusters: ordered in time, covering the series
    without gaps, each with one character of monotonicity.  ``segments[-1]`` is
    the most recent regime, the one to train a regression on.  ``pieces`` are
    the linear pieces found by the penalised segmentation before adjacent
    pieces with the same direction were merged into clusters.
    """

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    segments: tuple[Segment, ...]
    noise_std: float
    #: Criterion for every number of change points the search evaluated.
    criterion: tuple[CriterionRow, ...]
    #: Linear pieces before merging; empty means "same as ``segments``".
    pieces: tuple[Segment, ...] = ()
    #: Parameters of the :class:`~pytrendclust.TrendClusterer` that produced this result.
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pieces:
            object.__setattr__(self, "pieces", self.segments)

    # -- structure ---------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.y)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_change_points(self) -> int:
        return len(self.segments) - 1

    @property
    def change_points(self) -> list[int]:
        """Indices where a new segment starts (the first segment's start, 0, excluded)."""
        return [s.start for s in self.segments[1:]]

    @property
    def labels(self) -> NDArray[np.int64]:
        """Segment index of every point."""
        return np.repeat(
            np.arange(self.n_segments, dtype=np.int64), [s.n_points for s in self.segments]
        )

    @property
    def trends(self) -> list[Trend]:
        """Trend of the segment every point belongs to."""
        return [s.trend for s in self.segments for _ in range(s.n_points)]

    @property
    def fitted(self) -> NDArray[np.float64]:
        """Piecewise-linear fit: every point's value on the line of its own piece."""
        return np.concatenate([s.predict(self.t[s.start : s.stop]) for s in self.pieces])

    @property
    def piece_labels(self) -> NDArray[np.int64]:
        """Index of the linear piece every point belongs to."""
        return np.repeat(
            np.arange(len(self.pieces), dtype=np.int64), [s.n_points for s in self.pieces]
        )

    @property
    def last_segment(self) -> Segment:
        """The most recent trend cluster of the series."""
        return self.segments[-1]

    @property
    def last_piece(self) -> Segment:
        """The most recent linear piece: the tail of :attr:`last_segment` with one slope."""
        return self.pieces[-1]

    @property
    def objective(self) -> float:
        """Value of the penalised criterion at the chosen segmentation (of ``pieces``)."""
        chosen = len(self.pieces) - 1
        return next(r.objective for r in self.criterion if r.n_change_points == chosen)

    # -- selection ---------------------------------------------------------

    def _pick(self, segment: int, level: Level) -> Segment:
        if level == "cluster":
            return self.segments[segment]
        if level == "piece":
            return self.pieces[segment]
        msg = f"level must be 'cluster' or 'piece', got {level!r}"
        raise ValueError(msg)

    def indices(self, segment: int = -1, *, level: Level = "cluster") -> NDArray[np.int64]:
        """Indices of the points of one cluster (or linear piece); the last one by default."""
        s = self._pick(segment, level)
        return np.arange(s.start, s.stop, dtype=np.int64)

    def select(
        self, segment: int = -1, *, level: Level = "cluster"
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(t, y)`` of one cluster (or linear piece); the last (most recent) one by default.

        To train a linear regression on the current regime, ``level="piece"``
        is usually the better choice: a trend cluster may join pieces of one
        direction but different slopes, while the last piece follows one line.
        """
        s = self._pick(segment, level)
        return self.t[s.start : s.stop].copy(), self.y[s.start : s.stop].copy()

    # -- export ------------------------------------------------------------

    def to_dict(self, *, include_points: bool = True) -> dict[str, Any]:
        """Plain-Python representation, round-trippable through :meth:`from_dict`."""
        data: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "n_points": self.n_points,
            "n_segments": self.n_segments,
            "n_change_points": self.n_change_points,
            "change_points": self.change_points,
            "noise_std": self.noise_std,
            "objective": self.objective,
            "params": self.params,
            "segments": [s.to_dict() for s in self.segments],
            "pieces": [s.to_dict() for s in self.pieces],
            "criterion": [
                {
                    "n_change_points": r.n_change_points,
                    "sse": r.sse,
                    "penalty": r.penalty,
                    "objective": r.objective,
                }
                for r in self.criterion
            ],
        }
        if include_points:
            data["points"] = {"t": self.t.tolist(), "y": self.y.tolist()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendClusteringResult:
        """Rebuild a result exported with ``to_dict(include_points=True)``."""
        if data.get("format_version") not in _READABLE_VERSIONS:
            msg = f"unsupported format_version: {data.get('format_version')!r}"
            raise ValueError(msg)
        if "points" not in data:
            msg = "the export has no points; export it with include_points=True"
            raise ValueError(msg)
        return cls(
            t=np.asarray(data["points"]["t"], dtype=np.float64),
            y=np.asarray(data["points"]["y"], dtype=np.float64),
            segments=tuple(Segment.from_dict(s) for s in data["segments"]),
            noise_std=float(data["noise_std"]),
            criterion=tuple(
                CriterionRow(int(r["n_change_points"]), float(r["sse"]), float(r["penalty"]))
                for r in data["criterion"]
            ),
            pieces=tuple(Segment.from_dict(s) for s in data.get("pieces", ())),
            params=dict(data["params"]),
        )

    def to_json(self, path: str | Path | None = None, *, indent: int | None = 2) -> str:
        """Serialise to JSON; also write it to ``path`` when given."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> TrendClusteringResult:
        """Load a result from a JSON file path or a JSON string."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def point_table(self) -> list[dict[str, Any]]:
        """One row per point: index, t, y, cluster, trend, piece, fitted value, residual."""
        labels = self.labels
        pieces = self.piece_labels
        trends = self.trends
        fitted = self.fitted
        return [
            {
                "index": i,
                "t": float(self.t[i]),
                "y": float(self.y[i]),
                "segment": int(labels[i]),
                "trend": trends[i].value,
                "piece": int(pieces[i]),
                "fitted": float(fitted[i]),
                "residual": float(self.y[i] - fitted[i]),
                "is_last_segment": bool(labels[i] == self.n_segments - 1),
            }
            for i in range(self.n_points)
        ]

    def to_csv(self, path: str | Path | None = None) -> str:
        """Per-point table as CSV; also write it to ``path`` when given."""
        rows = self.point_table()
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        text = buffer.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_dataframe(self) -> pd.DataFrame:
        """Per-point table as a :class:`pandas.DataFrame` (needs ``pytrendclust[pandas]``)."""
        try:
            import pandas as pd
        except ImportError as exc:
            msg = "to_dataframe() needs pandas: pip install 'pytrendclust[pandas]'"
            raise ImportError(msg) from exc
        return pd.DataFrame(self.point_table())

    def export(self, path: str | Path) -> None:
        """Write the result to ``path``; the format follows the suffix (``.json``/``.csv``)."""
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            self.to_json(path)
        elif suffix == ".csv":
            self.to_csv(path)
        else:
            msg = f"unsupported export format {suffix!r}; use .json or .csv"
            raise ValueError(msg)

    def summary(self) -> str:
        """Human-readable one-line-per-segment description."""
        lines = [
            (
                f"{self.n_points} points, {self.n_change_points} change point(s), "
                f"noise std {self.noise_std:.4g}"
            )
        ]
        lines.extend(
            f"  #{s.index}: [{s.start}, {s.stop}) {s.n_points:>4} pts  "
            f"{s.trend.value:<4}  slope {s.slope:+.4g}"
            for s in self.segments
        )
        return "\n".join(lines)


def _json_float(value: float) -> float | str:
    """JSON has no infinity; spell it as a string that ``float()`` parses back."""
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value
