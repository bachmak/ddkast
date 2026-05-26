from __future__ import annotations

import logging

import pandas as pd
from rich.console import Console
from spotforecast2_safe.preprocessing import agg_and_resample_data

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.preprocessing.clean import clean

_console = Console()
_logger = logging.getLogger(__name__)


def run(config: Config) -> None:
    """Read raw data, clean / resample all three artifacts, trim, write."""
    raw = ParquetStore(config.raw_dir)
    processed = ParquetStore(config.processed_dir)

    # 1. Clean actual load
    _console.print("[bold]merge[/bold] cleaning actual load…")
    actual = raw.read(config.raw_load_actual)
    clean_load = clean(actual, config)

    # 2. Resample DAF to hourly (arrives at 15-min resolution from ENTSO-E)
    _console.print("  resampling ENTSO-E day-ahead forecast to hourly…")
    raw_daf = raw.read(config.raw_load_forecast)
    if isinstance(raw_daf.index, pd.DatetimeIndex):
        raw_daf = (
            raw_daf.tz_convert("UTC")
            if raw_daf.index.tz is not None
            else raw_daf.tz_localize("UTC")
        )
    daf_hourly = agg_and_resample_data(raw_daf, rule=config.resolution)

    # Fill short gaps (same policy as actual load)
    daf_hourly = daf_hourly.interpolate(
        method="linear", limit=config.max_interpolation_hours
    )

    # CR-3: warn (do not fail) if NaN remain — DAF gaps do not invalidate load data
    nan_count = int(daf_hourly.isna().sum().sum())
    if nan_count > 0:
        _logger.warning(
            "DAF has %d NaN after resampling/interpolation "
            "(max_interpolation_hours=%d); rows will be dropped.",
            nan_count,
            config.max_interpolation_hours,
        )
        daf_hourly = daf_hourly.dropna()

    # 3. Resample raw weather to hourly and ensure UTC
    _console.print("  processing weather…")
    raw_weather = raw.read(config.raw_weather)
    raw_idx: pd.DatetimeIndex = raw_weather.index  # type: ignore[assignment]
    if raw_idx.tz is None:
        raw_weather.index = raw_idx.tz_localize("UTC")
    else:
        raw_weather.index = raw_idx.tz_convert("UTC")
    weather_processed = raw_weather.resample("1h").mean()

    # 4. Trim load and DAF to their mutual coverage
    load_daf_index = clean_load.index.intersection(daf_hourly.index)
    clean_load_trimmed = clean_load.loc[load_daf_index]
    daf_trimmed = daf_hourly.loc[load_daf_index]

    # 5. Trim weather separately to its own coverage within the load range
    #    (Open-Meteo archive has a publication lag — last few days may be missing)
    weather_index = clean_load_trimmed.index.intersection(weather_processed.index)
    weather_trimmed = weather_processed.loc[weather_index]

    # 6. NaN check on load and DAF (hard fail); weather is best-effort
    for name, df in [("clean_load", clean_load_trimmed), ("daf", daf_trimmed)]:
        nan_count = int(df.isna().sum().sum())
        if nan_count > 0:
            raise ValueError(
                f"{name} has {nan_count} NaN after trimming to common index"
            )

    weather_nan = int(weather_trimmed.isna().sum().sum())
    if weather_nan > 0:
        _logger.warning(
            "weather has %d NaN after trimming; affected rows dropped.", weather_nan
        )
        weather_trimmed = weather_trimmed.dropna()

    # 7. Write all three
    processed.write(config.processed_load, clean_load_trimmed)
    _console.print(
        f"  [green]✓[/green] {len(clean_load_trimmed):,} clean rows → "
        f"{config.processed_dir / config.processed_load}.parquet"
    )

    processed.write(config.processed_entso_forecast, daf_trimmed)
    _console.print(
        f"  [green]✓[/green] {len(daf_trimmed):,} DAF rows → "
        f"{config.processed_dir / config.processed_entso_forecast}.parquet"
    )

    processed.write(config.processed_weather, weather_trimmed)
    _console.print(
        f"  [green]✓[/green] {len(weather_trimmed):,} weather rows → "
        f"{config.processed_dir / config.processed_weather}.parquet"
    )
