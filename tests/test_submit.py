"""Tests for the ``ddkast submit`` stage.

The stage slices "tomorrow" (UTC) out of the predictions written by ``predict``,
so each test constructs a predictions Parquet whose index spans today + tomorrow
and points the stage at it via the ``submit`` test config.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.pipeline import submit


@pytest.fixture
def submit_config(config: Config, tmp_path: Path) -> Config:
    return config.model_copy(
        update={
            "team_id": "test-team",
            "submissions_dir": tmp_path / "submissions",
        }
    )


def _tomorrow_index() -> pd.DatetimeIndex:
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    start = pd.Timestamp(tomorrow, tz="UTC")
    return pd.date_range(start=start, periods=24, freq="1h")


def _write_predictions(config: Config, series: pd.Series[float]) -> None:
    processed = ParquetStore(config.processed_dir)
    processed.write(config.processed_predictions, series.to_frame(config.model_target))


def test_submit_writes_valid_csv(submit_config: Config) -> None:
    idx = _tomorrow_index()
    values = np.linspace(40_000.0, 60_000.0, 24)
    _write_predictions(submit_config, pd.Series(values, index=idx))

    submit.run(submit_config)

    forecast_date = idx[0].date().isoformat()
    out_path = (
        submit_config.submissions_dir / submit_config.team_id / f"{forecast_date}.csv"
    )
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert list(df.columns) == ["timestamp_utc", "forecast_mw"]
    assert len(df) == 24
    assert df["timestamp_utc"].iloc[0] == f"{forecast_date}T00:00:00Z"
    assert df["timestamp_utc"].iloc[-1] == f"{forecast_date}T23:00:00Z"
    assert (df["forecast_mw"] > 0).all()
    assert df["forecast_mw"].dtype == float


def test_submit_rejects_short_window(submit_config: Config) -> None:
    idx = _tomorrow_index()[:23]
    _write_predictions(submit_config, pd.Series(np.full(23, 50_000.0), index=idx))

    with pytest.raises(ValueError, match="Expected 24"):
        submit.run(submit_config)


def test_submit_rejects_wrong_start_timestamp(submit_config: Config) -> None:
    idx = _tomorrow_index() + pd.Timedelta(hours=1)
    _write_predictions(submit_config, pd.Series(np.full(24, 50_000.0), index=idx))

    with pytest.raises(ValueError, match="Expected 24|does not match"):
        submit.run(submit_config)


def test_submit_rejects_nan(submit_config: Config) -> None:
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[5] = np.nan
    _write_predictions(submit_config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="NaN"):
        submit.run(submit_config)


def test_submit_rejects_non_positive(submit_config: Config) -> None:
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[10] = 0.0
    _write_predictions(submit_config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="non-positive"):
        submit.run(submit_config)


def test_submit_rejects_infinite(submit_config: Config) -> None:
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[7] = np.inf
    _write_predictions(submit_config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="non-finite"):
        submit.run(submit_config)


def test_submit_skips_file_on_validation_failure(submit_config: Config) -> None:
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[3] = -1.0
    _write_predictions(submit_config, pd.Series(values, index=idx))

    with pytest.raises(ValueError):
        submit.run(submit_config)

    team_dir = submit_config.submissions_dir / submit_config.team_id
    assert not team_dir.exists() or not any(team_dir.iterdir())
