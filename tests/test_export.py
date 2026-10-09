import csv
import io
import json

import numpy as np
import pytest

from pytrendclust import TrendClusteringResult, cluster_trends

Y = [10, 9, 8, 7, 6, 5, 4, 5, 6, 7]


@pytest.fixture
def result():
    return cluster_trends(Y)


def test_dict_round_trip(result):
    restored = TrendClusteringResult.from_dict(result.to_dict())
    assert restored.segments == result.segments
    assert restored.criterion == result.criterion
    assert restored.pieces == result.pieces
    np.testing.assert_array_equal(restored.y, result.y)
    assert restored.params == result.params


def test_json_file_round_trip(result, tmp_path):
    path = tmp_path / "result.json"
    result.export(path)
    restored = TrendClusteringResult.from_json(path)
    assert restored.change_points == [7]
    assert TrendClusteringResult.from_json(path.read_text()).segments == result.segments


def test_json_content(result):
    data = json.loads(result.to_json())
    assert data["change_points"] == [7]
    assert [s["trend"] for s in data["segments"]] == ["down", "up"]
    assert data["params"]["penalty"] == "exponential"
    assert data["criterion"][0]["n_change_points"] == 0


def test_noise_free_t_statistic_survives_json():
    result = cluster_trends([0.0, 1.0, 2.0, 3.0, 4.0], noise_std=0.0)
    restored = TrendClusteringResult.from_json(result.to_json())
    assert restored.segments == result.segments


def test_csv_has_one_row_per_point(result, tmp_path):
    path = tmp_path / "points.csv"
    result.export(path)
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    assert len(rows) == len(Y)
    assert [r["segment"] for r in rows] == ["0"] * 7 + ["1"] * 3
    assert rows[-1]["trend"] == "up"
    assert rows[-1]["is_last_segment"] == "True"
    assert float(rows[0]["fitted"]) == pytest.approx(10.0)


def test_dataframe(result):
    pd = pytest.importorskip("pandas")
    frame = result.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert frame["segment"].tolist() == result.labels.tolist()


def test_unknown_export_format(result, tmp_path):
    with pytest.raises(ValueError, match="unsupported export format"):
        result.export(tmp_path / "result.xlsx")


def test_from_dict_without_points(result):
    with pytest.raises(ValueError, match="include_points"):
        TrendClusteringResult.from_dict(result.to_dict(include_points=False))


def test_version_1_exports_are_still_readable(result):
    data = result.to_dict()
    data["format_version"] = 1
    del data["pieces"]
    restored = TrendClusteringResult.from_dict(data)
    assert restored.pieces == restored.segments


def test_from_dict_rejects_other_versions(result):
    data = result.to_dict()
    data["format_version"] = 99
    with pytest.raises(ValueError, match="format_version"):
        TrendClusteringResult.from_dict(data)


def test_select_feeds_a_regression(result):
    t, y = result.select()
    slope, intercept = np.polyfit(t, y, 1)
    assert slope == pytest.approx(1.0)
    assert result.last_segment.predict(10) == pytest.approx(intercept + slope * 10)


def test_summary_mentions_every_segment(result):
    text = result.summary()
    assert "1 change point(s)" in text
    assert text.count("\n") == result.n_segments


def test_select_last_piece_of_a_two_slope_rise():
    rng = np.random.default_rng(3)
    y = np.r_[np.arange(40) * 1.0, 40 + np.arange(40) * 4.0] + rng.normal(0, 0.5, 80)
    result = cluster_trends(y)
    assert result.n_segments == 1
    assert result.last_piece.start == pytest.approx(40, abs=2)
    t, _ = result.select(level="piece")
    assert t[0] == result.last_piece.start
    assert result.indices(level="piece").tolist() == list(range(result.last_piece.start, 80))
    with pytest.raises(ValueError, match="level"):
        result.select(level="window")
