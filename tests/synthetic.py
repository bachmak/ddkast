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
    """Config update for ``historical_folds`` historical origins plus the live tail.

    Reproduces the previous index-anchored geometry: origins step ``stride_hours``
    apart, ending on the data tail, so ``historical_folds`` blocks land in the actuals
    and the last runs past them.
    """
    end = last_ts(periods)
    start = end - pd.Timedelta(hours=historical_folds * stride_hours)
    return {
        "n_forecasts": historical_folds + 1,
        "forecasts_start": start,
        "forecasts_end": end,
    }
