"""Tests for the DataSource abstraction and the generated smoke fixtures."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ddkast.config import Config
from ddkast.data.source import (
    ApiDataSource,
    FixtureDataSource,
    make_data_source,
)
from ddkast.data.weather import WEATHER_COLS


def test_make_data_source_defaults_to_api(config: Config) -> None:
    assert isinstance(make_data_source(config), ApiDataSource)


def test_make_data_source_selects_fixtures(fixtures_config: Config) -> None:
    assert isinstance(make_data_source(fixtures_config), FixtureDataSource)


def test_fixtures_have_expected_schema(fixtures_config: Config) -> None:
    source = make_data_source(fixtures_config)
    # Wide window: the full generated fixture (2023-12-31 → 2024-01-12).
    start, end = datetime(2023, 1, 1), datetime(2025, 1, 1)

    actual = source.load_actual(start, end)
    forecast = source.load_forecast(start, end)
    weather = source.weather(start, end)

    assert not actual.empty
    assert list(actual.columns) == ["load_mw"]
    assert not forecast.empty
    assert list(forecast.columns) == ["forecast_mw"]
    assert not weather.empty
    assert list(weather.columns) == WEATHER_COLS


def test_window_slice_trims(fixtures_config: Config) -> None:
    source = make_data_source(fixtures_config)

    full = source.load_actual(datetime(2023, 1, 1), datetime(2025, 1, 1))
    start, end = datetime(2024, 1, 2), datetime(2024, 1, 3)
    window = source.load_actual(start, end)

    assert 0 < len(window) < len(full)
    idx_utc = window.index.tz_convert("UTC")
    assert idx_utc.min() >= pd.Timestamp(start, tz="UTC")
    assert idx_utc.max() < pd.Timestamp(end, tz="UTC")
