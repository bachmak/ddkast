from __future__ import annotations

import json
from datetime import datetime, timedelta

from rich.console import Console

from ddkast.config import Config
from ddkast.data.source import make_data_source
from ddkast.data.store import ParquetStore

_console = Console()

_FINGERPRINT_NAME = "download.fingerprint.json"


def _cache_key(config: Config) -> dict[str, object]:
    """The config values that determine the download output (the cache key)."""
    return {
        "country_code": config.country_code,
        "data_source": config.data_source,
        "download_end": config.download_end.isoformat(),
        "download_start": config.download_start.isoformat(),
        "horizon": config.horizon,
        "weather_latitude": config.weather_latitude,
        "weather_longitude": config.weather_longitude,
    }


def _is_cached(config: Config, key: dict[str, object]) -> bool:
    """True if all raw outputs exist and the stored fingerprint matches ``key``."""
    outputs = (config.raw_load_actual, config.raw_load_forecast, config.raw_weather)
    if not all((config.raw_dir / f"{name}.parquet").exists() for name in outputs):
        return False
    fingerprint = config.raw_dir / _FINGERPRINT_NAME
    if not fingerprint.exists():
        return False
    return json.loads(fingerprint.read_text()) == key


def run(config: Config, *, force: bool = False) -> None:
    """Fetch raw load data from ENTSO-E and weather from Open-Meteo."""
    key = _cache_key(config)
    if not force and _is_cached(config, key):
        _console.print("[green]✓[/green] download cache hit — skipping fetch")
        return

    store = ParquetStore(config.raw_dir)
    source = make_data_source(config)

    start = datetime(
        config.download_start.year,
        config.download_start.month,
        config.download_start.day,
    )
    # Add one day so the full download_end date is included
    end = datetime(
        config.download_end.year,
        config.download_end.month,
        config.download_end.day,
    ) + timedelta(days=1)

    _console.print(
        f"[bold]download[/bold] {config.download_start} → "
        f"{config.download_end} ({config.country_code})"
    )

    _console.print("  fetching actual load…")
    actual = source.load_actual(start, end)
    store.write(config.raw_load_actual, actual)
    _console.print(
        f"  [green]✓[/green] actual load  {len(actual):,} rows → "
        f"{config.raw_dir / config.raw_load_actual}.parquet"
    )

    _console.print("  fetching day-ahead forecast…")
    forecast = source.load_forecast(start, end)
    store.write(config.raw_load_forecast, forecast)
    _console.print(
        f"  [green]✓[/green] DAF forecast  {len(forecast):,} rows → "
        f"{config.raw_dir / config.raw_load_forecast}.parquet"
    )

    _console.print("  fetching weather archive (Open-Meteo)…")
    weather = source.weather(start, end)
    store.write(config.raw_weather, weather)
    _console.print(
        f"  [green]✓[/green] weather       {len(weather):,} rows → "
        f"{config.raw_dir / config.raw_weather}.parquet"
    )

    (config.raw_dir / _FINGERPRINT_NAME).write_text(
        json.dumps(key, indent=2, sort_keys=True)
    )
