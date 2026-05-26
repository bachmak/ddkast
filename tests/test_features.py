from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.preprocessing.features import build_exog_builder, build_exog_matrix

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


def _synthetic_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="1h")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        rng.uniform(0, 20, (len(idx), len(_WEATHER_COLS))),
        index=idx,
        columns=_WEATHER_COLS,
    )


def test_build_exog_builder_returns_exog_builder(config: Config) -> None:
    from spotforecast2_safe import ExogBuilder

    builder = build_exog_builder(config)
    assert isinstance(builder, ExogBuilder)


def test_exog_output_shape(config: Config) -> None:
    builder = build_exog_builder(config)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-07 23:00:00", tz="UTC")
    exog = builder.build(start, end)

    expected_rows = 7 * 24  # one week of hourly data
    assert len(exog) == expected_rows

    # RBF columns + holidays + is_weekend
    expected_cols = (
        config.rbf_periods_hour + config.rbf_periods_dow + config.rbf_periods_month + 2
    )
    assert exog.shape[1] == expected_cols


def test_exog_no_nan(config: Config) -> None:
    builder = build_exog_builder(config)
    start = pd.Timestamp("2024-06-01", tz="UTC")
    end = pd.Timestamp("2024-06-30 23:00:00", tz="UTC")
    exog = builder.build(start, end)
    assert exog.isna().sum().sum() == 0


def test_exog_has_expected_columns(config: Config) -> None:
    builder = build_exog_builder(config)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01 23:00:00", tz="UTC")
    exog = builder.build(start, end)

    assert all(f"hour_{i}" in exog.columns for i in range(config.rbf_periods_hour))
    assert all(f"dow_{i}" in exog.columns for i in range(config.rbf_periods_dow))
    assert all(f"month_{i}" in exog.columns for i in range(config.rbf_periods_month))
    assert "holidays" in exog.columns
    assert "is_weekend" in exog.columns


def test_exog_holiday_flag(config: Config) -> None:
    # Christmas Day 2024 (Dec 25) is a German public holiday
    builder = build_exog_builder(config)
    start = pd.Timestamp("2024-12-25", tz="UTC")
    end = pd.Timestamp("2024-12-25 23:00:00", tz="UTC")
    exog = builder.build(start, end)
    assert (exog["holidays"] == 1).all()


def test_exog_weekend_flag(config: Config) -> None:
    # 2024-01-06 is a Saturday
    builder = build_exog_builder(config)
    start = pd.Timestamp("2024-01-06", tz="UTC")
    end = pd.Timestamp("2024-01-06 23:00:00", tz="UTC")
    exog = builder.build(start, end)
    assert (exog["is_weekend"] == 1).all()


# --- build_exog_matrix tests ---


def test_build_exog_matrix_column_count(config: Config) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-07 23:00:00", tz="UTC")
    weather = _synthetic_weather(start, end)
    exog = build_exog_matrix(start, end, weather, config)

    expected_calendar = (
        config.rbf_periods_hour + config.rbf_periods_dow + config.rbf_periods_month + 2
    )
    expected_cols = expected_calendar + len(_WEATHER_COLS)
    assert exog.shape[1] == expected_cols


def test_build_exog_matrix_no_nan(config: Config) -> None:
    start = pd.Timestamp("2024-06-01", tz="UTC")
    end = pd.Timestamp("2024-06-30 23:00:00", tz="UTC")
    weather = _synthetic_weather(start, end)
    exog = build_exog_matrix(start, end, weather, config)
    assert exog.isna().sum().sum() == 0


def test_build_exog_matrix_index_matches_range(config: Config) -> None:
    start = pd.Timestamp("2024-03-01", tz="UTC")
    end = pd.Timestamp("2024-03-07 23:00:00", tz="UTC")
    weather = _synthetic_weather(start, end)
    exog = build_exog_matrix(start, end, weather, config)

    expected_index = pd.date_range(start, end, freq="1h")
    assert len(exog) == len(expected_index)
    assert exog.index[0] == start
    assert exog.index[-1] == end


def test_build_exog_matrix_raises_on_nan(config: Config) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-02 23:00:00", tz="UTC")
    weather = _synthetic_weather(start, end)
    weather.iloc[5, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN"):
        build_exog_matrix(start, end, weather, config)
