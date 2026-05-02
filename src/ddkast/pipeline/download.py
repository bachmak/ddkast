from __future__ import annotations

from datetime import datetime, timedelta

from rich.console import Console

from ddkast.config import Config
from ddkast.data.fetch import fetch_load, fetch_load_forecast
from ddkast.data.store import ParquetStore

_console = Console()


def run(config: Config) -> None:
    """Fetch raw load data from ENTSO-E and write to the raw data store."""
    store = ParquetStore(config.raw_dir)

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
    actual = fetch_load(config.entsoe_api_key, config.country_code, start, end)
    store.write(config.raw_load_actual, actual)
    _console.print(
        f"  [green]✓[/green] actual load  {len(actual):,} rows → "
        f"{config.raw_dir / config.raw_load_actual}.parquet"
    )

    _console.print("  fetching day-ahead forecast…")
    forecast = fetch_load_forecast(
        config.entsoe_api_key, config.country_code, start, end
    )
    store.write(config.raw_load_forecast, forecast)
    _console.print(
        f"  [green]✓[/green] DAF forecast  {len(forecast):,} rows → "
        f"{config.raw_dir / config.raw_load_forecast}.parquet"
    )
