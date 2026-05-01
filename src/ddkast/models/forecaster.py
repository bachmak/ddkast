from __future__ import annotations

import pandas as pd

from ddkast.config import Config


def fit(df: pd.DataFrame, config: Config) -> None:
    """Fit a recursive forecaster using spotforecast2-safe and persist it."""
    raise NotImplementedError


def forecast(df: pd.DataFrame, config: Config) -> pd.Series:  # type: ignore[type-arg]
    """Load the persisted model and return a forecast of length config.horizon."""
    raise NotImplementedError
