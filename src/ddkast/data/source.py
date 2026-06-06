from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd

from ddkast.config import Config
from ddkast.data.fetch import fetch_load, fetch_load_forecast
from ddkast.data.store import ParquetStore
from ddkast.data.weather import fetch_weather


class DataSource(Protocol):
    """Where the download stage gets its raw frames from.

    Mirrors the DataStore protocol: a thin seam that lets the pipeline read
    either live APIs (production) or committed fixtures (offline CI smoke test)
    without the download stage knowing which.
    """

    def load_actual(self, start: datetime, end: datetime) -> pd.DataFrame: ...
    def load_forecast(self, start: datetime, end: datetime) -> pd.DataFrame: ...
    def weather(self, start: datetime, end: datetime) -> pd.DataFrame: ...


class ApiDataSource:
    """Live source: delegates to the existing fetch_* functions (production path)."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def load_actual(self, start: datetime, end: datetime) -> pd.DataFrame:
        return fetch_load(
            self._config.entsoe_api_key, self._config.country_code, start, end
        )

    def load_forecast(self, start: datetime, end: datetime) -> pd.DataFrame:
        return fetch_load_forecast(
            self._config.entsoe_api_key, self._config.country_code, start, end
        )

    def weather(self, start: datetime, end: datetime) -> pd.DataFrame:
        return fetch_weather(
            start=start,
            end=end,
            latitude=self._config.weather_latitude,
            longitude=self._config.weather_longitude,
            horizon=self._config.horizon,
        )


class FixtureDataSource:
    """Offline source: reads committed parquet fixtures, slices to [start, end).

    Honours the same DOWNLOAD_START / DOWNLOAD_END knobs as the live source so
    those env vars stay meaningful in fixture mode. Fixtures live under the same
    Config filename keys as the live raw store (raw_load_actual, …).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._store = ParquetStore(config.fixtures_dir)

    def load_actual(self, start: datetime, end: datetime) -> pd.DataFrame:
        return _slice(
            self._store.read(self._config.raw_load_actual),
            start,
            end,
            name=self._config.raw_load_actual,
        )

    def load_forecast(self, start: datetime, end: datetime) -> pd.DataFrame:
        return _slice(
            self._store.read(self._config.raw_load_forecast),
            start,
            end,
            name=self._config.raw_load_forecast,
        )

    def weather(self, start: datetime, end: datetime) -> pd.DataFrame:
        # Mirror the live source's forecast tail: fetch_weather appends the next
        # ``horizon`` hours past the archive end so the latest origin predict forecasts
        # has future exog. Extend the fixture slice the same way (the committed fixture
        # is generated wide enough to cover it).
        return _slice(
            self._store.read(self._config.raw_weather),
            start,
            end + timedelta(hours=self._config.horizon),
            name=self._config.raw_weather,
        )


def _slice(
    df: pd.DataFrame, start: datetime, end: datetime, *, name: str = ""
) -> pd.DataFrame:
    """Return rows whose timestamp falls in [start, end), comparing in UTC.

    Works for both tz-aware index resolutions in our fixtures (Europe/Berlin
    load, UTC weather): the index is normalised to UTC before comparison while
    the original index (and its tz) is preserved on the returned frame.
    """
    assert isinstance(df.index, pd.DatetimeIndex)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    idx_utc = df.index.tz_convert("UTC")
    mask = (idx_utc >= start_ts) & (idx_utc < end_ts)  # type: ignore[operator]
    result = df.loc[mask]
    if result.empty:
        label = f" '{name}'" if name else ""
        raise ValueError(
            f"Fixture{label} has no rows in [{start_ts}, {end_ts}). "
            f"Fixture range: {df.index.min()} – {df.index.max()}"
        )
    return result


def make_data_source(config: Config) -> DataSource:
    """Pick the source the download stage uses, driven by config.data_source."""
    if config.data_source == "fixtures":
        return FixtureDataSource(config)
    return ApiDataSource(config)
