from __future__ import annotations

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.folds import Fold, build_folds
from ddkast.models.forecaster import forecast
from ddkast.pipeline.train import fold_model_dir

_console = Console()


def run(config: Config) -> None:
    """Emit each fold's forecast from its model — never trains, reads no actuals.

    Writes one forecast per fold to ``predictions/<fold_id>``, treating every fold the
    same: a fold is just a rolling origin and the block it anchors, whether that block
    lands in the past or the future. Predict picks no operational fold — that selection
    belongs downstream: ``format-submission`` takes the fold whose block covers the
    submission day, and ``evaluate`` scores the folds the actuals already cover, each
    deciding it from the data. Determinism (CR-2): fold order is fixed and every fold is
    forecast identically.
    """
    processed = ParquetStore(config.processed_dir)
    load_df = processed.read(config.processed_load)
    weather_df = processed.read(config.processed_weather)

    folds = build_folds(load_df.index, config)  # type: ignore[arg-type]
    _print_folds(folds)

    for fold in folds:
        _predict_fold(fold, processed, weather_df, config)

    _report_forecasts(folds, config)


def _print_folds(folds: list[Fold]) -> None:
    """Announce how many folds will be forecast before prediction begins."""
    _console.print(f"[bold]predict[/bold]  forecasting {len(folds)} folds…")


def _predict_fold(
    fold: Fold, processed: ParquetStore, weather_df: pd.DataFrame, config: Config
) -> None:
    """Forecast one fold from its persisted model and persist it to ``predictions/``."""
    predictions = forecast(
        weather_df,
        config,
        fold_model_dir(config, fold),
        fold.forecast_start,
        fold.forecast_end,
    )
    predictions_df = predictions.to_frame(name=config.model_target)
    processed.write(f"{config.predictions_subdir}/{fold.fold_id}", predictions_df)


def _report_forecasts(folds: list[Fold], config: Config) -> None:
    """Report where the per-fold forecasts landed."""
    _console.print(
        f"  [green]✓[/green] {len(folds)} forecasts → "
        f"{config.processed_dir / config.predictions_subdir}/"
    )
