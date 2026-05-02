from __future__ import annotations

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.models.forecaster import fit

_console = Console()


def run(config: Config) -> None:
    """Split processed data, fit the forecasting model, and persist the test split."""
    processed = ParquetStore(config.processed_dir)
    df = processed.read(config.processed_load)

    cutoff: pd.Timestamp = df.index[-1] - pd.Timedelta(days=config.test_days)  # type: ignore[assignment]
    train_df = df.loc[df.index <= cutoff]
    test_df = df.loc[df.index > cutoff]

    _console.print(
        f"[bold]train[/bold]  {train_df.index[0].date()} → {train_df.index[-1].date()} "
        f"({len(train_df):,} rows)  |  test: {len(test_df):,} rows"
    )

    fit(train_df, config)
    processed.write(config.processed_test, test_df)
    _console.print(
        f"  [green]✓[/green] model saved  |  test split → "
        f"{config.processed_dir / config.processed_test}.parquet"
    )
