from __future__ import annotations

import pandas as pd


def add_features(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Add lag, rolling, Fourier cyclical, calendar, and holiday features."""
    raise NotImplementedError
