"""Unit tests for data/weather.py — mocked WeatherService, no network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ddkast.data.weather import fetch_weather

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


def _make_weather_df(start: pd.Timestamp, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.uniform(0, 20, (periods, len(_WEATHER_COLS))),
        index=idx,
        columns=_WEATHER_COLS,
    )


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "weather.parquet"


def test_fetch_weather_archive_returns_hourly_df(cache_path: Path) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-07 23:00:00", tz="UTC")
    expected_rows = 7 * 24

    mock_df = _make_weather_df(start, expected_rows)

    with patch("ddkast.data.weather.WeatherService") as mock_ws:
        instance = MagicMock()
        instance.get_dataframe.return_value = mock_df
        mock_ws.return_value = instance

        result = fetch_weather(
            start=start,
            end=end,
            latitude=50.1,
            longitude=8.7,
            cache_path=cache_path,
            use_forecast=False,
        )

    assert len(result) == expected_rows
    assert result.isna().sum().sum() == 0
    mock_ws.assert_called_once_with(
        latitude=50.1,
        longitude=8.7,
        cache_path=cache_path,
        use_forecast=False,
    )
    instance.get_dataframe.assert_called_once_with(
        start=start, end=end, freq="h", fill_missing=False
    )


def test_fetch_weather_forecast_returns_24_rows(cache_path: Path) -> None:
    start = pd.Timestamp("2024-06-01", tz="UTC")
    end = pd.Timestamp("2024-06-01 23:00:00", tz="UTC")
    mock_df = _make_weather_df(start, 24)

    with patch("ddkast.data.weather.WeatherService") as mock_ws:
        instance = MagicMock()
        instance.get_dataframe.return_value = mock_df
        mock_ws.return_value = instance

        result = fetch_weather(
            start=start,
            end=end,
            latitude=50.1,
            longitude=8.7,
            cache_path=cache_path,
            use_forecast=True,
        )

    assert len(result) == 24
    mock_ws.assert_called_once_with(
        latitude=50.1,
        longitude=8.7,
        cache_path=cache_path,
        use_forecast=True,
    )


def test_fetch_weather_archive_propagates_gap_error(cache_path: Path) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-07 23:00:00", tz="UTC")

    with patch("ddkast.data.weather.WeatherService") as mock_ws:
        instance = MagicMock()
        instance.get_dataframe.side_effect = ValueError("gaps detected")
        mock_ws.return_value = instance

        with pytest.raises(ValueError, match="gaps detected"):
            fetch_weather(
                start=start,
                end=end,
                latitude=50.1,
                longitude=8.7,
                cache_path=cache_path,
                use_forecast=False,
            )


def test_fetch_weather_forecast_propagates_gap_error(cache_path: Path) -> None:
    start = pd.Timestamp("2024-06-01", tz="UTC")
    end = pd.Timestamp("2024-06-01 23:00:00", tz="UTC")

    with patch("ddkast.data.weather.WeatherService") as mock_ws:
        instance = MagicMock()
        instance.get_dataframe.side_effect = ValueError("gaps detected")
        mock_ws.return_value = instance

        with pytest.raises(ValueError, match="gaps detected"):
            fetch_weather(
                start=start,
                end=end,
                latitude=50.1,
                longitude=8.7,
                cache_path=cache_path,
                use_forecast=True,
            )
