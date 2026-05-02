from __future__ import annotations

from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.preprocessing.clean import clean

_console = Console()


def run(config: Config) -> None:
    """Read raw data, clean it, and write a single processed dataset."""
    raw = ParquetStore(config.raw_dir)
    processed = ParquetStore(config.processed_dir)

    _console.print("[bold]merge[/bold] cleaning actual load…")
    actual = raw.read(config.raw_load_actual)
    cleaned = clean(actual, config)
    processed.write(config.processed_load, cleaned)
    _console.print(
        f"  [green]✓[/green] {len(cleaned):,} clean rows → "
        f"{config.processed_dir / config.processed_load}.parquet"
    )

    _console.print("  passing through ENTSO-E day-ahead forecast…")
    forecast = raw.read(config.raw_load_forecast)
    processed.write(config.processed_entso_forecast, forecast)
    _console.print(
        f"  [green]✓[/green] {len(forecast):,} rows → "
        f"{config.processed_dir / config.processed_entso_forecast}.parquet"
    )
