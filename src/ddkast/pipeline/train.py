from __future__ import annotations

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.models.forecaster import fit
from ddkast.preprocessing.features import build_exog_matrix

_console = Console()


def run(config: Config) -> None:
    """Split processed data, fit the forecasting model, and persist the test split."""
    processed = ParquetStore(config.processed_dir)
    df = processed.read(config.processed_load)

    cutoff: pd.Timestamp = df.index[-1] - pd.Timedelta(days=config.test_days)  # type: ignore[assignment]
    train_df = df.loc[df.index <= cutoff]
    test_df = df.loc[df.index > cutoff]

    # Build full exog over the entire load range so predict can reuse it
    weather_df = processed.read(config.processed_weather)
    load_start: pd.Timestamp = df.index.min()  # type: ignore[assignment]
    exog_full = build_exog_matrix(
        start=load_start,
        end=weather_df.index.max(),
        weather_df=weather_df,
        config=config,
    )
    processed.write(config.processed_exog, exog_full)

    # Clip train to exog coverage (weather may end before load due to publication lag)
    train_df = train_df.loc[exog_full.index.min() : exog_full.index.max()]
    train_df = train_df.loc[train_df.index <= cutoff]
    exog_train = exog_full.loc[train_df.index]

    _console.print(
        f"[bold]train[/bold]  {train_df.index[0].date()} → {train_df.index[-1].date()} "
        f"({len(train_df):,} rows)  |  test: {len(test_df):,} rows  |  "
        f"exog: {exog_train.shape[1]} cols"
    )

    fit(train_df, config, exog_train)
    processed.write(config.processed_test, test_df)
    _console.print(
        f"  [green]✓[/green] model saved  |  test split → "
        f"{config.processed_dir / config.processed_test}.parquet"
    )
