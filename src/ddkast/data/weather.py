from __future__ import annotations

from pathlib import Path

import pandas as pd
from spotforecast2_safe.weather import WeatherService


def fetch_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    latitude: float,
    longitude: float,
    use_forecast: bool,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch weather from Open-Meteo without WeatherService's internal cache.

    cache_path=None (default) disables WeatherService's file cache entirely —
    every call hits the API directly. Callers that want caching must manage it
    themselves (e.g. via DataStore / ParquetStore).

    use_forecast=False  reanalysis archive (full historical coverage).
    use_forecast=True   forecast endpoint (prospective, max 7 days ahead).
    fill_missing=False  raises ValueError on any gap (CR-3 fail-safe).
    """
    ws = WeatherService(
        latitude=latitude,
        longitude=longitude,
        cache_path=cache_path,
        use_forecast=use_forecast,
    )
    return ws.get_dataframe(start=start, end=end, freq="h", fill_missing=False)
