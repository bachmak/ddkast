from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.folds import Fold, build_folds
from ddkast.models.forecaster import fit

_console = Console()


def fold_model_dir(config: Config, fold: Fold) -> Path:
    """Per-fold model directory: ``models/folds/<fold_id>`` (shared with predict)."""
    return config.models_dir / "folds" / fold.fold_id


def run(config: Config) -> None:
    """Fit one forecaster per rolling-origin fold; persist each under models/folds/."""
    processed = ParquetStore(config.processed_dir)
    load_df = processed.read(config.processed_load)
    weather_df = processed.read(config.processed_weather)

    folds = build_folds(load_df.index, config)  # type: ignore[arg-type]
    _print_folds(folds)

    for fold in folds:
        _fit_fold(fold, load_df, weather_df, config)

    _report_saved(folds, config)


def _print_folds(folds: list[Fold]) -> None:
    """Announce how many rolling-origin folds will be fitted."""
    _console.print(f"[bold]train[/bold]  fitting {len(folds)} rolling-origin folds…")


def _fit_fold(
    fold: Fold, load_df: pd.DataFrame, weather_df: pd.DataFrame, config: Config
) -> None:
    """Fit and persist one fold's forecaster on its expanding training window."""
    train_df = _training_window(load_df, fold)
    fit(train_df, weather_df, config, fold_model_dir(config, fold))


def _training_window(load_df: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """The expanding window a fold trains on: every row up to (incl.) its origin."""
    return load_df.loc[load_df.index < fold.forecast_start]  # type: ignore[misc,return-value]


def _report_saved(folds: list[Fold], config: Config) -> None:
    """Report how many fold models were persisted and the latest origin."""
    _console.print(
        f"  [green]✓[/green] {len(folds)} models saved → "
        f"{config.models_dir / 'folds'}  |  latest origin {folds[-1].forecast_start}"
    )
