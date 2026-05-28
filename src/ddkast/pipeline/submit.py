from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore

_console = Console()


def _slice_tomorrow_utc(predictions: pd.Series[float]) -> pd.Series[float]:
    """Return predictions for tomorrow 00:00–23:00 UTC (24 hourly rows)."""
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    start = pd.Timestamp(tomorrow, tz="UTC")
    end = start + pd.Timedelta(hours=23)
    window: pd.Series[float] = predictions.loc[start:end]  # type: ignore[misc]

    if len(window) != 24:
        raise ValueError(
            f"Expected 24 hourly predictions for {tomorrow} UTC, got {len(window)}. "
            f"Predictions span {predictions.index[0]} → {predictions.index[-1]}. "
            "Check config.horizon (must reach tomorrow 23:00 UTC)."
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


def run(config: Config) -> None:
    """Write tomorrow's hourly forecast to a leaderboard-shaped CSV."""
    processed = ParquetStore(config.processed_dir)
    predictions: pd.Series[float] = processed.read(config.processed_predictions)[
        config.model_target
    ]

    window = _slice_tomorrow_utc(predictions)
    index: pd.DatetimeIndex = window.index  # type: ignore[assignment]
    forecast_date = index[0].date()

    submission = pd.DataFrame(
        {
            "timestamp_utc": index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_mw": window.to_numpy().astype(float),
        }
    )

    out_dir = config.submissions_dir / config.team_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{forecast_date.isoformat()}.csv"

    _console.print(
        f"[bold]submit[/bold]  writing forecast for {forecast_date} (UTC) → {out_path}…"
    )
    submission.to_csv(out_path, index=False)

    values = window.to_numpy()
    _console.print(
        f"  [green]✓[/green] {len(submission)} rows "
        f"(MW range {values.min():.2f} → {values.max():.2f})"
    )
