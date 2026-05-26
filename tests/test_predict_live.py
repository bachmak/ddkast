"""Tests for pipeline/predict.py live path (--target-date)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.pipeline import predict, train

_WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]


@pytest.fixture
def live_config(config: Config) -> Config:
    return config.model_copy(update={"lags": 5, "test_days": 2, "horizon": 24})


@pytest.fixture
def _write_synthetic(live_config: Config) -> None:
    """Populate processed store with clean load and weather (offline)."""
    processed = ParquetStore(live_config.processed_dir)
    rng = np.random.default_rng(42)

    # 240 h of load: 2024-01-01 00:00 → 2024-01-10 23:00 UTC (inclusive)
    idx = pd.date_range("2024-01-01", periods=240, freq="1h", tz="UTC")
    load_df = pd.DataFrame(
        {live_config.model_target: rng.uniform(40_000, 60_000, len(idx))}, index=idx
    )
    processed.write(live_config.processed_load, load_df)

    weather_df = pd.DataFrame(
        rng.uniform(0, 20, (len(idx), len(_WEATHER_COLS))),
        index=idx,
        columns=_WEATHER_COLS,
    )
    processed.write(live_config.processed_weather, weather_df)

    # DAF needs to cover the test window (last test_days*24 h)
    daf_df = pd.DataFrame(
        {"forecast_mw": rng.uniform(40_000, 60_000, len(idx))}, index=idx
    )
    processed.write(live_config.processed_entso_forecast, daf_df)

    # Train model so predict can load it
    train.run(live_config)


def test_predict_live_raises_when_load_gap(
    _write_synthetic: None, live_config: Config
) -> None:
    """Requesting a target_date that's not D+1 from last load hour raises ValueError."""
    # Last load hour is 2024-01-10 23:00; target_date two days ahead creates a gap
    target = date(2024, 1, 12)
    with pytest.raises(ValueError, match="ddkast download"):
        predict.run(live_config, target_date=target)


def _make_forecast_weather(start: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(start, periods=24, freq="1h")
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.uniform(0, 20, (24, len(_WEATHER_COLS))),
        index=idx,
        columns=_WEATHER_COLS,
    )


def test_predict_live_produces_24_rows(
    _write_synthetic: None, live_config: Config
) -> None:
    """With correct load continuity and mocked weather, predict returns 24 rows."""
    # Last load hour is 2024-01-10 23:00; target_date is 2024-01-11
    target = date(2024, 1, 11)
    target_start = pd.Timestamp(target, tz="UTC")
    mock_weather = _make_forecast_weather(target_start)

    with patch("ddkast.pipeline.predict.fetch_weather", return_value=mock_weather):
        predict.run(live_config, target_date=target)

    processed = ParquetStore(live_config.processed_dir)
    preds = processed.read(live_config.processed_predictions)
    assert len(preds) == 24
