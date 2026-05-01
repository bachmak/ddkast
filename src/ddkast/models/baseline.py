from __future__ import annotations

import pandas as pd

_WEEK_IN_HOURS = 24 * 7


def predict(series: pd.Series, horizon: int = 24) -> pd.Series:  # type: ignore[type-arg]
    """7-day seasonal naive baseline: predict = same hour from one week ago."""
    return series.shift(_WEEK_IN_HOURS).iloc[-horizon:]
