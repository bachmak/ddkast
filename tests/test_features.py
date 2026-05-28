from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.preprocessing.features import build_exog_matrix


def test_build_exog_matrix_returns_dataframe(
    config: Config, make_weather: Callable[..., pd.DataFrame]
) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01 23:00:00", tz="UTC")
    exog = build_exog_matrix(start, end, make_weather(start, end), config)
    assert isinstance(exog, pd.DataFrame)


def test_exog_output_shape(
    config: Config,
    make_weather: Callable[..., pd.DataFrame],
    weather_cols: list[str],
) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-07 23:00:00", tz="UTC")
    exog = build_exog_matrix(start, end, make_weather(start, end), config)

    expected_rows = 7 * 24  # one week of hourly data
    assert len(exog) == expected_rows

    # RBF columns + holidays + is_weekend + weather columns
    expected_cols = (
        config.rbf_periods_hour
        + config.rbf_periods_dow
        + config.rbf_periods_month
        + 2
        + len(weather_cols)
    )
    assert exog.shape[1] == expected_cols


def test_exog_no_nan(config: Config, make_weather: Callable[..., pd.DataFrame]) -> None:
    start = pd.Timestamp("2024-06-01", tz="UTC")
    end = pd.Timestamp("2024-06-30 23:00:00", tz="UTC")
    exog = build_exog_matrix(start, end, make_weather(start, end), config)
    assert exog.isna().sum().sum() == 0


def test_exog_has_expected_columns(
    config: Config,
    make_weather: Callable[..., pd.DataFrame],
    weather_cols: list[str],
) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01 23:00:00", tz="UTC")
    exog = build_exog_matrix(start, end, make_weather(start, end), config)

    assert all(f"hour_{i}" in exog.columns for i in range(config.rbf_periods_hour))
    assert all(f"dow_{i}" in exog.columns for i in range(config.rbf_periods_dow))
    assert all(f"month_{i}" in exog.columns for i in range(config.rbf_periods_month))
    assert "holidays" in exog.columns
    assert "is_weekend" in exog.columns
    assert all(col in exog.columns for col in weather_cols)


def test_exog_holiday_flag(
    config: Config, make_weather: Callable[..., pd.DataFrame]
) -> None:
    # Christmas Day 2024 (Dec 25) is a German public holiday
    start = pd.Timestamp("2024-12-25", tz="UTC")
    end = pd.Timestamp("2024-12-25 23:00:00", tz="UTC")
    exog = build_exog_matrix(start, end, make_weather(start, end), config)
    assert (exog["holidays"] == 1).all()


def test_exog_weekend_flag(
    config: Config, make_weather: Callable[..., pd.DataFrame]
) -> None:
    # 2024-01-06 is a Saturday
    start = pd.Timestamp("2024-01-06", tz="UTC")
    end = pd.Timestamp("2024-01-06 23:00:00", tz="UTC")
    exog = build_exog_matrix(start, end, make_weather(start, end), config)
    assert (exog["is_weekend"] == 1).all()


def test_exog_raises_on_nan(
    config: Config, make_weather: Callable[..., pd.DataFrame]
) -> None:
    # A NaN in the joined matrix must fail fast rather than reach the model.
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01 23:00:00", tz="UTC")
    weather = make_weather(start, end)
    weather.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_exog_matrix(start, end, weather, config)
