"""Helpers for pinning rolling-origin fold windows onto the synthetic test data.

``write_processed`` (see ``conftest``) seeds an hourly index starting at ``DATA_START``,
so a test that wants ``n`` scored historical folds plus the live origin at the data tail
can derive the matching ``forecasts_start``/``forecasts_end``/``n_forecasts`` from
the row count.
"""

from __future__ import annotations

import pandas as pd

# write_processed seeds an hourly index starting here.
DATA_START = pd.Timestamp("2024-01-01", tz="UTC")


def last_ts(periods: int) -> pd.Timestamp:
    """Final timestamp of the synthetic hourly index of ``periods`` rows."""
    return DATA_START + pd.Timedelta(hours=periods - 1)


def fold_window(
    periods: int, historical_folds: int, stride_hours: int = 24
) -> dict[str, object]:
    """Config update for ``historical_folds`` realized folds plus the live one.

    Half-open geometry: a block is ``[forecast_start, forecast_start + horizon)``. The
    live fold's ``forecast_start`` is the first timestamp past the data tail (hourly
    grid), so its block is entirely future (skipped by evaluate); the earlier
    ``historical_folds`` blocks land fully in the actuals, the latest ending exactly on
    the tail. ``forecasts_start``/``forecasts_end`` are the earliest and live starts.
    """
    live_start = last_ts(periods) + pd.Timedelta(hours=1)  # first future stamp
    first_start = live_start - pd.Timedelta(hours=historical_folds * stride_hours)
    return {
        "n_forecasts": historical_folds + 1,
        "forecasts_start": first_start,
        "forecasts_end": live_start,
    }
