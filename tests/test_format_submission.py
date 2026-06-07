"""Tests for the ``ddkast format-submission`` stage.

The stage rebuilds the rolling-origin folds from the configured window, picks the one
fold whose block covers tomorrow (UTC), and slices that fold's persisted forecast. So
each test pins a fold's ``forecast_start`` at tomorrow 00:00 UTC — making its half-open
block exactly tomorrow — seeds a ``processed_load`` with enough history before it, and
writes that fold's ``predictions/<id>`` file.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.folds import build_folds
from ddkast.pipeline import format_submission


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "submissions" / "test-team"


@pytest.fixture
def sub_config(config: Config) -> Config:
    """One fold; its forecast_start is tomorrow 00:00 UTC, so its block is tomorrow."""
    origin = _tomorrow_0000()
    return config.model_copy(
        update={
            "lags": 24,
            "horizon": 24,
            "n_forecasts": 1,
            "forecasts_start": origin,
            "forecasts_end": origin,
        }
    )


def _tomorrow_index() -> pd.DatetimeIndex:
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    start = pd.Timestamp(tomorrow, tz="UTC")
    return pd.date_range(start=start, periods=24, freq="1h")


def _today_2300() -> pd.Timestamp:
    today = datetime.now(UTC).date()
    return pd.Timestamp(today, tz="UTC") + pd.Timedelta(hours=23)


def _tomorrow_0000() -> pd.Timestamp:
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    return pd.Timestamp(tomorrow, tz="UTC")


def _seed_load(config: Config, last_ts: pd.Timestamp, periods: int = 24 * 3) -> None:
    """Write a flat ``processed_load`` ending at ``last_ts`` — values unused."""
    idx = pd.date_range(end=last_ts, periods=periods, freq="1h")
    ParquetStore(config.processed_dir).write(
        config.processed_load,
        pd.DataFrame({config.model_target: np.full(periods, 50_000.0)}, index=idx),
    )


def _write_fold_predictions(config: Config, series: pd.Series[float]) -> None:
    """Persist ``series`` as the latest fold's ``predictions/<fold_id>`` forecast."""
    processed = ParquetStore(config.processed_dir)
    folds = build_folds(processed.read(config.processed_load).index, config)  # type: ignore[arg-type]
    processed.write(
        f"{config.predictions_subdir}/{folds[-1].fold_id}",
        series.to_frame(config.model_target),
    )


def test_writes_valid_csv(sub_config: Config, out_dir: Path) -> None:
    _seed_load(sub_config, _today_2300())
    idx = _tomorrow_index()
    values = np.linspace(40_000.0, 60_000.0, 24)
    _write_fold_predictions(sub_config, pd.Series(values, index=idx))

    format_submission.run(sub_config, out_dir)

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


def test_rounds_to_two_decimals(sub_config: Config, out_dir: Path) -> None:
    _seed_load(sub_config, _today_2300())
    idx = _tomorrow_index()
    # Values with >2 decimals; the stage must round for leaderboard parity.
    _write_fold_predictions(
        sub_config, pd.Series(np.full(24, 50_000.123456), index=idx)
    )

    format_submission.run(sub_config, out_dir)

    forecast_date = idx[0].date().isoformat()
    df = pd.read_csv(out_dir / f"{forecast_date}.csv")
    assert (df["forecast_mw"] == df["forecast_mw"].round(2)).all()
    assert (df["forecast_mw"] == 50_000.12).all()


def test_rejects_no_covering_fold(config: Config, out_dir: Path) -> None:
    # The only fold's origin is two days ago → its block is yesterday, not tomorrow.
    origin = _today_2300() - pd.Timedelta(hours=48)
    cfg = config.model_copy(
        update={
            "lags": 24,
            "horizon": 24,
            "n_forecasts": 1,
            "forecasts_start": origin,
            "forecasts_end": origin,
        }
    )
    _seed_load(cfg, origin)

    with pytest.raises(ValueError, match="No fold covers"):
        format_submission.run(cfg, out_dir)


def test_rejects_ambiguous_folds(config: Config, out_dir: Path) -> None:
    # horizon (48) > stride (24) makes two consecutive 48h blocks both cover tomorrow.
    end = _tomorrow_0000()
    cfg = config.model_copy(
        update={
            "lags": 24,
            "horizon": 48,
            "n_forecasts": 2,
            "forecasts_start": end - pd.Timedelta(hours=24),
            "forecasts_end": end,
        }
    )
    _seed_load(cfg, end, periods=24 * 4)

    with pytest.raises(ValueError, match="expected exactly one"):
        format_submission.run(cfg, out_dir)


def test_rejects_short_window(sub_config: Config, out_dir: Path) -> None:
    _seed_load(sub_config, _today_2300())
    idx = _tomorrow_index()[:23]
    _write_fold_predictions(sub_config, pd.Series(np.full(23, 50_000.0), index=idx))

    with pytest.raises(ValueError, match="Expected 24"):
        format_submission.run(sub_config, out_dir)


def test_rejects_wrong_start_timestamp(sub_config: Config, out_dir: Path) -> None:
    _seed_load(sub_config, _today_2300())
    idx = _tomorrow_index() + pd.Timedelta(hours=1)
    _write_fold_predictions(sub_config, pd.Series(np.full(24, 50_000.0), index=idx))

    with pytest.raises(ValueError, match="Expected 24|does not match"):
        format_submission.run(sub_config, out_dir)


def test_rejects_nan(sub_config: Config, out_dir: Path) -> None:
    _seed_load(sub_config, _today_2300())
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[5] = np.nan
    _write_fold_predictions(sub_config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="NaN"):
        format_submission.run(sub_config, out_dir)


def test_rejects_non_positive(sub_config: Config, out_dir: Path) -> None:
    _seed_load(sub_config, _today_2300())
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[10] = 0.0
    _write_fold_predictions(sub_config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="non-positive"):
        format_submission.run(sub_config, out_dir)


def test_rejects_infinite(sub_config: Config, out_dir: Path) -> None:
    _seed_load(sub_config, _today_2300())
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[7] = np.inf
    _write_fold_predictions(sub_config, pd.Series(values, index=idx))

    with pytest.raises(ValueError, match="non-finite"):
        format_submission.run(sub_config, out_dir)


def test_skips_file_on_validation_failure(sub_config: Config, out_dir: Path) -> None:
    _seed_load(sub_config, _today_2300())
    idx = _tomorrow_index()
    values = np.full(24, 50_000.0)
    values[3] = -1.0
    _write_fold_predictions(sub_config, pd.Series(values, index=idx))

    with pytest.raises(ValueError):
        format_submission.run(sub_config, out_dir)

    assert not out_dir.exists() or not any(out_dir.iterdir())
