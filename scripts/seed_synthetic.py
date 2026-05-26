"""Populate the raw data store with realistic synthetic load and weather data.

Bypasses the ENTSO-E API and Open-Meteo so the full pipeline can be exercised
without any API key or network access.
Run from the project root:

    uv run python scripts/seed_synthetic.py

Then run the remaining stages normally:

    uv run ddkast merge
    uv run ddkast train
    uv run ddkast predict
    uv run ddkast evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console

from ddkast.config import load
from ddkast.data.store import ParquetStore

_console = Console()

_RNG = np.random.default_rng(42)

# Realistic Germany load range (MW)
_BASE_MW = 52_000.0
_DAILY_AMP = 8_000.0  # peak-to-trough swing within a day
_WEEKLY_AMP = 5_000.0  # workday vs. weekend drop
_NOISE_STD = 800.0
_FORECAST_ERROR_STD = 600.0  # DAF error relative to actual

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


def _synthetic_load(index: pd.DatetimeIndex) -> np.ndarray:
    hours = index.hour.to_numpy(dtype=float)
    dow = index.dayofweek.to_numpy(dtype=float)  # 0=Mon … 6=Sun
    doy = index.dayofyear.to_numpy(dtype=float)

    # Daily pattern: low at night, two peaks (morning + evening)
    daily = (
        -np.cos(2 * np.pi * hours / 24) * _DAILY_AMP * 0.6
        + np.sin(2 * np.pi * hours / 24) * _DAILY_AMP * 0.4
    )

    # Weekly pattern: weekends ~10 % lower
    weekly = np.where(dow >= 5, -_WEEKLY_AMP, 0.0)

    # Annual pattern: higher in winter, lower in summer
    annual = -np.cos(2 * np.pi * doy / 365) * 4_000.0

    noise = _RNG.normal(0.0, _NOISE_STD, len(index))

    return _BASE_MW + daily + weekly + annual + noise


def main() -> None:
    config = load()
    store = ParquetStore(config.raw_dir)

    start = pd.Timestamp(config.download_start, tz="UTC")
    end = pd.Timestamp(config.download_end, tz="UTC") + pd.Timedelta(hours=23)
    index = pd.date_range(start=start, end=end, freq="1h")

    _console.print(
        f"[bold]seed[/bold] generating {len(index):,} hours of synthetic load "
        f"({config.download_start} → {config.download_end})"
    )

    actual_values = _synthetic_load(index)
    actual_df = pd.DataFrame({"load_mw": actual_values}, index=index)
    store.write(config.raw_load_actual, actual_df)
    _console.print(
        f"  [green]✓[/green] actual load  {len(actual_df):,} rows "
        f"→ {config.raw_dir / config.raw_load_actual}.parquet"
    )

    forecast_values = actual_values + _RNG.normal(0.0, _FORECAST_ERROR_STD, len(index))
    forecast_df = pd.DataFrame({"forecast_mw": forecast_values}, index=index)
    store.write(config.raw_load_forecast, forecast_df)
    _console.print(
        f"  [green]✓[/green] DAF forecast  {len(forecast_df):,} rows "
        f"→ {config.raw_dir / config.raw_load_forecast}.parquet"
    )

    weather_data = {
        "temperature_2m": 10.0 + 8.0 * np.sin(2 * np.pi * index.dayofyear / 365),
        "relative_humidity_2m": _RNG.uniform(50, 90, len(index)),
        "precipitation": np.clip(_RNG.exponential(0.1, len(index)), 0, 10),
        "rain": np.clip(_RNG.exponential(0.05, len(index)), 0, 5),
        "snowfall": np.clip(_RNG.exponential(0.01, len(index)), 0, 2),
        "weather_code": _RNG.integers(0, 100, len(index)).astype(float),
        "pressure_msl": _RNG.normal(1013.0, 10.0, len(index)),
        "surface_pressure": _RNG.normal(1008.0, 10.0, len(index)),
        "cloud_cover": _RNG.uniform(0, 100, len(index)),
        "cloud_cover_low": _RNG.uniform(0, 50, len(index)),
        "cloud_cover_mid": _RNG.uniform(0, 50, len(index)),
        "cloud_cover_high": _RNG.uniform(0, 50, len(index)),
        "wind_speed_10m": _RNG.uniform(0, 20, len(index)),
        "wind_direction_10m": _RNG.uniform(0, 360, len(index)),
        "wind_gusts_10m": _RNG.uniform(0, 30, len(index)),
    }
    weather_df = pd.DataFrame(weather_data, index=index)
    store.write(config.raw_weather, weather_df)
    _console.print(
        f"  [green]✓[/green] weather       {len(weather_df):,} rows "
        f"→ {config.raw_dir / config.raw_weather}.parquet"
    )

    _console.print()
    _console.print("Now run:")
    _console.print("  [cyan]uv run ddkast merge[/cyan]")
    _console.print("  [cyan]uv run ddkast train[/cyan]")
    _console.print("  [cyan]uv run ddkast predict[/cyan]")
    _console.print("  [cyan]uv run ddkast evaluate[/cyan]")


if __name__ == "__main__":
    main()
