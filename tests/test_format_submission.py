"""Tests for the ``ddkast format-submission`` stage.

The stage slices "tomorrow" (UTC) out of the predictions written by ``predict``,
so each test constructs a predictions Parquet whose index spans tomorrow and
points the stage at it via the test config + an ``out_dir`` fixture.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.pipeline import format_submission


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "submissions" / "test-team"


def _tomorrow_index() -> pd.DatetimeIndex:
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    start = pd.Timestamp(tomorrow, tz="UTC")
    return pd.date_range(start=start, periods=24, freq="1h")


def _write_predictions(config: Config, series: pd.Series[float]) -> None:
    processed = ParquetStore(config.processed_dir)
    processed.write(config.processed_predictions, series.to_frame(config.model_target))


def test_writes_valid_csv(config: Config, out_dir: Path) -> None:
    idx = _tomorrow_index()
    values = np.linspace(40_000.0, 60_000.0, 24)
    _write_predictions(config, pd.Series(values, index=idx))

    format_submission.run(config, out_dir)

    forecast_date = idx[0].date().isoformat()
    out_path = out_dir / f"{forecast_date}.csv"
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert list(df.columns) == ["timestamp_utc", "forecast_mw"]
    assert len(df) == 24
    assert df["timestamp_utc"].iloc[0] == f"{forecast_date}T00:00:00Z"
    assert df["timestamp_utc"].iloc[-1] == f"{forecast_date}T23:00:00Z"
    assert (df["forecast_mw"] > 0).all()
    assert df["forecast_mw"].dtype == float


def test_rejects_short_window(config: Config, out_dir: Path) -> None:
    idx = _tomorrow_index()[:23]
    _write_predictions(config, pd.Series(np.full(23, 50_000.0), index=idx))

    with pytest.raises(ValueError, match="Expected 24"):
        format_submission.run(config, out_dir)


def test_rejects_wrong_start_timestamp(config: Config, out_dir: Path) -> None:
    idx = _tomorrow_index() + pd.Timedelta(hours=1)
    _write_predictions(config, pd.Series(np.full(24, 50_000.0), index=idx))

    with pytest.raises(ValueError, match="Expected 24|does not match"):
        format_submission.run(config, out_dir)


def test_rejects_nan(config: Config, out_dir: Path) -> None:
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[5] = np.nan
    _write_predictions(config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="NaN"):
        format_submission.run(config, out_dir)


def test_rejects_non_positive(config: Config, out_dir: Path) -> None:
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[10] = 0.0
    _write_predictions(config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="non-positive"):
        format_submission.run(config, out_dir)


def test_rejects_infinite(config: Config, out_dir: Path) -> None:
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[7] = np.inf
    _write_predictions(config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="non-finite"):
        format_submission.run(config, out_dir)


def test_skips_file_on_validation_failure(config: Config, out_dir: Path) -> None:
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[3] = -1.0
    _write_predictions(config, pd.Series(values, index=idx))

    with pytest.raises(ValueError):
        format_submission.run(config, out_dir)

    assert not out_dir.exists() or not any(out_dir.iterdir())
