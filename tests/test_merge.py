"""Tests for pipeline/merge.py — common-index trim, hourly DAF, no NaN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.pipeline import merge

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
def merge_config(config: Config) -> Config:
    return config.model_copy(update={"outlier_iqr_multiplier": 10.0})


@pytest.fixture
def raw_data(merge_config: Config) -> None:
    """Write synthetic raw artifacts with different tail lengths to test trim."""
    raw = ParquetStore(merge_config.raw_dir)
    rng = np.random.default_rng(55)

    # Actual load: 100 h at 15-min resolution (400 rows), UTC-aware
    load_idx_15min = pd.date_range("2024-01-01", periods=400, freq="15min", tz="UTC")
    load_df = pd.DataFrame(
        {"load_mw": rng.uniform(40_000, 60_000, 400)}, index=load_idx_15min
    )
    raw.write(merge_config.raw_load_actual, load_df)

    # DAF: 98 h at 15-min resolution (392 rows) — 2 h shorter tail than load
    daf_idx_15min = pd.date_range("2024-01-01", periods=392, freq="15min", tz="UTC")
    daf_df = pd.DataFrame(
        {"forecast_mw": rng.uniform(40_000, 60_000, 392)}, index=daf_idx_15min
    )
    raw.write(merge_config.raw_load_forecast, daf_df)

    # Weather: 96 h at 1h resolution — 4 h shorter tail than load
    weather_idx_1h = pd.date_range("2024-01-01", periods=96, freq="1h", tz="UTC")
    weather_df = pd.DataFrame(
        rng.uniform(0, 20, (96, len(_WEATHER_COLS))),
        index=weather_idx_1h,
        columns=_WEATHER_COLS,
    )
    raw.write(merge_config.raw_weather, weather_df)


def test_merge_load_daf_share_index_weather_is_subset(
    raw_data: None, merge_config: Config
) -> None:
    merge.run(merge_config)
    processed = ParquetStore(merge_config.processed_dir)

    load = processed.read(merge_config.processed_load)
    daf = processed.read(merge_config.processed_entso_forecast)
    weather = processed.read(merge_config.processed_weather)

    assert load.index.equals(daf.index), "load and DAF indexes differ"
    # Weather has a separate trim (Open-Meteo lag) — subset of load
    assert weather.index.isin(load.index).all(), "weather not a subset of load"
    assert len(weather) <= len(load), "weather should not be longer than load"


def test_merge_daf_is_hourly(raw_data: None, merge_config: Config) -> None:
    merge.run(merge_config)
    processed = ParquetStore(merge_config.processed_dir)
    daf = processed.read(merge_config.processed_entso_forecast)

    # All consecutive gaps must be exactly 1 hour
    gaps = daf.index.to_series().diff().dropna()
    assert (gaps == pd.Timedelta("1h")).all(), "DAF index is not uniformly hourly"


def test_merge_no_nan_in_any_artifact(raw_data: None, merge_config: Config) -> None:
    merge.run(merge_config)
    processed = ParquetStore(merge_config.processed_dir)

    for key in [
        merge_config.processed_load,
        merge_config.processed_entso_forecast,
        merge_config.processed_weather,
    ]:
        df = processed.read(key)
        assert df.isna().sum().sum() == 0, f"{key} contains NaN after merge"


def test_merge_trimmed_length_matches_shortest(
    raw_data: None, merge_config: Config
) -> None:
    """Load/DAF are trimmed to their mutual coverage; weather is trimmed separately."""
    merge.run(merge_config)
    processed = ParquetStore(merge_config.processed_dir)

    load = processed.read(merge_config.processed_load)
    weather = processed.read(merge_config.processed_weather)
    # Load is trimmed to load ∩ DAF (98h); weather trimmed to its own coverage (96h)
    assert len(weather) <= len(load), "weather should not exceed load length"
    assert len(weather) <= 96, "weather cannot exceed its raw 96-row input"
