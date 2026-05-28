from __future__ import annotations

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.models.forecaster import forecast

_console = Console()


def run(config: Config) -> None:
    """Load the trained model and generate a forecast for the test window."""
    processed = ParquetStore(config.processed_dir)
    df = processed.read(config.processed_load)
    weather = processed.read(config.processed_weather)

    # Trim to the same training portion used in train.py
    cutoff: pd.Timestamp = df.index[-1] - pd.Timedelta(days=config.test_days)  # type: ignore[assignment]
    train_df = df.loc[df.index <= cutoff]

    _console.print(
        f"[bold]predict[/bold]  forecasting {config.horizon}h from {cutoff.date()}…"
    )

    predictions = forecast(train_df, weather, config)
    predictions_df = predictions.to_frame(name=config.model_target)
    processed.write(config.processed_predictions, predictions_df)

    _console.print(
        f"  [green]✓[/green] {len(predictions)} predictions "
        f"({predictions.index[0]} → {predictions.index[-1]}) "
        f"→ {config.processed_dir / config.processed_predictions}.parquet"
    )
