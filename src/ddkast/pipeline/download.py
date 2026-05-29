from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.incremental import append_tail, plan_fetch
from ddkast.data.source import make_data_source
from ddkast.data.store import ParquetStore

_console = Console()

Fetcher = Callable[[datetime, datetime], pd.DataFrame]


def _download_source(
    store: ParquetStore,
    name: str,
    label: str,
    fetch: Fetcher,
    start: datetime,
    end: datetime,
    *,
    full: bool,
) -> None:
    """Fetch one source incrementally (or fully) and persist it via the store."""
    existing = store.read(name) if not full and store.exists(name) else None
    plan = plan_fetch(existing, start, end)

    if plan.mode == "uptodate" and existing is not None:
        _console.print(
            f"  [green]✓[/green] {label:<12} up to date  ({len(existing):,} rows)"
        )
        return

    fetched = fetch(plan.start, plan.end)

    if plan.mode == "tail" and existing is not None:
        result = append_tail(existing, fetched)
        added = len(result) - len(existing)
        if added == 0:
            _console.print(
                f"  [green]✓[/green] {label:<12} up to date  ({len(existing):,} rows)"
            )
            return
        summary = f"+{added:,} rows (now {len(result):,})"
    else:
        result = fetched
        summary = f"{len(result):,} rows"

    store.write(name, result)
    _console.print(
        f"  [green]✓[/green] {label:<12} {summary} → {store.base_dir / name}.parquet"
    )


def run(config: Config, *, full: bool = False) -> None:
    """Fetch raw load data from ENTSO-E and weather from Open-Meteo.

    By default only the missing forward tail of each source is fetched and
    appended to what is already stored. Pass ``full=True`` to ignore existing
    data and re-fetch the entire configured range.
    """
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

    mode = " [yellow](--full)[/yellow]" if full else ""
    _console.print(
        f"[bold]download[/bold]{mode} {config.download_start} → "
        f"{config.download_end} ({config.country_code})"
    )

    _download_source(
        store,
        config.raw_load_actual,
        "actual load",
        source.load_actual,
        start,
        end,
        full=full,
    )
    _download_source(
        store,
        config.raw_load_forecast,
        "DAF forecast",
        source.load_forecast,
        start,
        end,
        full=full,
    )
    _download_source(
        store,
        config.raw_weather,
        "weather",
        source.weather,
        start,
        end,
        full=full,
    )
