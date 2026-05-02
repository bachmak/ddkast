from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.evaluation.metrics import report

_console = Console()

_WEEK_HOURS = 168


def run(config: Config) -> None:
    """Compute metrics for model, naive baseline, and ENTSO-E DAF; print report."""
    processed = ParquetStore(config.processed_dir)

    predictions: pd.Series[float] = processed.read(config.processed_predictions)[
        config.model_target
    ]
    load = processed.read(config.processed_load)
    entso = processed.read(config.processed_entso_forecast)

    # Actual values aligned to forecast timestamps
    actual: pd.Series[float] = load[config.model_target].reindex(predictions.index)
    if actual.isna().any():
        raise ValueError(
            "Some forecast timestamps have no matching actual values. "
            "Ensure the test period is within the downloaded data range."
        )

    # 7-day naive: look up each forecast timestamp 168 hours earlier
    assert isinstance(predictions.index, pd.DatetimeIndex)
    naive_idx: pd.DatetimeIndex = predictions.index - pd.Timedelta(hours=_WEEK_HOURS)
    naive_values = load[config.model_target].reindex(naive_idx).values
    naive = pd.Series(naive_values, index=predictions.index, dtype=float)
    if naive.isna().any():
        raise ValueError(
            "Some naive baseline timestamps fall outside the downloaded data range. "
            "Extend download_start or reduce test_days in config.toml."
        )

    # ENTSO-E DAF aligned to forecast timestamps
    entso_series: pd.Series[float] = entso["forecast_mw"].reindex(predictions.index)
    entso_available = not entso_series.isna().any()

    model_metrics = report(actual, predictions)
    naive_metrics = report(actual, naive)

    title = (
        f"Evaluation  ({predictions.index[0].date()} → {predictions.index[-1].date()})"
    )
    table = Table(title=title)
    table.add_column("Metric", style="bold")
    table.add_column("7-day Naive", justify="right")
    if entso_available:
        table.add_column("ENTSO-E DAF", justify="right")
    table.add_column("Model", justify="right", style="green")

    for metric in ("MAE", "RMSE", "MAPE", "SMAPE"):
        unit = " %" if metric in ("MAPE", "SMAPE") else " MW"
        fmt = f"{naive_metrics[metric]:,.1f}{unit}"
        row = [metric, fmt]
        if entso_available:
            entso_metrics = report(actual, entso_series)
            row.append(f"{entso_metrics[metric]:,.1f}{unit}")
        row.append(f"{model_metrics[metric]:,.1f}{unit}")
        table.add_row(*row)

    _console.print(table)
