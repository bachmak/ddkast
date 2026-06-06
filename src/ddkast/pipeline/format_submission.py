from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.folds import Fold, build_folds

_console = Console()


def run(config: Config, out_dir: Path) -> None:
    """Write tomorrow's hourly forecast to a leaderboard CSV under ``out_dir``.

    Rebuilds the rolling-origin folds from the load index, picks the one fold whose
    block covers the submission day (CR-3: raises if none or several do), then slices
    that fold's persisted forecast — predict no longer flags an operational fold for us.
    """
    processed = ParquetStore(config.processed_dir)
    load_df = processed.read(config.processed_load)

    folds = build_folds(load_df.index, config)  # type: ignore[arg-type]
    start, end = _tomorrow_window()
    fold = _select_submission_fold(folds, start, end)

    predictions = _read_fold_forecast(processed, fold, config)
    window = _slice_window(predictions, start, end)

    submission = _build_submission(window)
    _write_submission(submission, window, fold, out_dir)


def _tomorrow_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """The submission window, ``[start, end)``: tomorrow 00:00 UTC up to 23:00 UTC."""
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    start = pd.Timestamp(tomorrow, tz="UTC")
    return start, start + pd.Timedelta(hours=24)


def _select_submission_fold(
    folds: list[Fold], start: pd.Timestamp, end: pd.Timestamp
) -> Fold:
    """The single fold whose forecast block covers the submission window — or raise."""
    covering = [f for f in folds if f.forecast_start <= start and f.forecast_end >= end]
    if len(covering) == 1:
        return covering[0]
    if not covering:
        raise ValueError(
            f"No fold covers the submission window {start} → {end}: the latest fold "
            "ends earlier. Re-run predict on fresher data or raise config.horizon."
        )
    ids = ", ".join(f.fold_id for f in covering)
    raise ValueError(
        f"{len(covering)} folds cover the submission window {start} → {end} ({ids}); "
        "expected exactly one — overlapping folds (stride < horizon) are ambiguous."
    )


def _read_fold_forecast(
    processed: ParquetStore, fold: Fold, config: Config
) -> pd.Series[float]:
    """Read the selected fold's persisted forecast series from ``predictions/``."""
    return processed.read(f"{config.predictions_subdir}/{fold.fold_id}")[
        config.model_target
    ]


def _slice_window(
    predictions: pd.Series[float], start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series[float]:
    """Slice and validate the submission window out of the selected fold's forecast."""
    idx = predictions.index
    window: pd.Series[float] = predictions[(idx >= start) & (idx < end)]  # type: ignore[misc,assignment]

    if len(window) != 24:
        raise ValueError(
            f"Expected 24 hourly predictions for {start.date()} UTC, "
            f"got {len(window)}. "
            f"Predictions span {predictions.index[0]} → {predictions.index[-1]}; "
            "the selected fold's forecast has a gap in the submission window."
        )
    first_ts: pd.Timestamp = window.index[0]  # type: ignore[assignment]
    if first_ts != start:
        raise ValueError(
            f"First prediction timestamp {first_ts.isoformat()} does not match "
            f"expected start {start.isoformat()}."
        )

    values = window.to_numpy()
    if pd.isna(values).any():
        raise ValueError("Forecast contains NaN values; cannot submit.")
    if not np.isfinite(values).all():
        raise ValueError("Forecast contains non-finite values; cannot submit.")
    if not (values > 0).all():
        raise ValueError(
            "Forecast contains non-positive values; "
            "leaderboard requires strictly positive MW."
        )
    return window


def _build_submission(window: pd.Series[float]) -> pd.DataFrame:
    """Assemble the leaderboard frame: UTC timestamp strings and rounded MW values."""
    index: pd.DatetimeIndex = window.index  # type: ignore[assignment]
    return pd.DataFrame(
        {
            "timestamp_utc": index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            # Round to 2 decimals for leaderboard parity (lecture §13.5.3 y0.round(2)).
            "forecast_mw": window.to_numpy().round(2).astype(float),
        }
    )


def _write_submission(
    submission: pd.DataFrame, window: pd.Series[float], fold: Fold, out_dir: Path
) -> None:
    """Write the submission CSV named for its forecast date and report MW coverage."""
    index: pd.DatetimeIndex = window.index  # type: ignore[assignment]
    forecast_date = index[0].date()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{forecast_date.isoformat()}.csv"

    _console.print(
        f"[bold]format-submission[/bold]  fold {fold.fold_id} → forecast for "
        f"{forecast_date} (UTC) → {out_path}…"
    )
    submission.to_csv(out_path, index=False)

    values = window.to_numpy()
    _console.print(
        f"  [green]✓[/green] {len(submission)} rows "
        f"(MW range {values.min():.2f} → {values.max():.2f})"
    )
