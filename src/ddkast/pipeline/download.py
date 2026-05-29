from __future__ import annotations

from datetime import datetime, timedelta

from rich.console import Console

from ddkast.config import Config
from ddkast.data.source import make_data_source
from ddkast.data.store import ParquetStore

_console = Console()


def run(config: Config) -> None:
    """Fetch raw load data from ENTSO-E and weather from Open-Meteo."""
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
