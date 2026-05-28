from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.preprocessing.clean import clean

_console = Console()


def _passthrough(
    raw: ParquetStore,
    processed: ParquetStore,
    label: str,
    raw_key: str,
    processed_key: str,
    processed_dir: Path,
) -> None:
    _console.print(f"  passing through {label}…")
    data = raw.read(raw_key)
    processed.write(processed_key, data)
    _console.print(
        f"  [green]✓[/green] {len(data):,} rows → "
        f"{processed_dir / processed_key}.parquet"
    )


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

    _passthrough(
        raw,
        processed,
        "ENTSO-E day-ahead forecast",
        config.raw_load_forecast,
        config.processed_entso_forecast,
        config.processed_dir,
    )

    _passthrough(
        raw,
        processed,
        "weather",
        config.raw_weather,
        config.processed_weather,
        config.processed_dir,
    )
