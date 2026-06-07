from __future__ import annotations

import pandas as pd
from spotforecast2_safe import ExogBuilder
from spotforecast2_safe.data.data import Period

from ddkast.config import Config


def _build_exog_builder(config: Config) -> ExogBuilder:
    """Return a configured ExogBuilder: RBF cyclical, holidays, weekend flag.

    Private to this module — callers should use build_exog_matrix, which joins
    the calendar features produced here with the weather features.
    """
    periods = [
        Period(
            name="hour",
            n_periods=config.rbf_periods_hour,
            column="hour",
            input_range=(0, 23),
        ),
        Period(
            name="dow",
            n_periods=config.rbf_periods_dow,
            column="dayofweek",
            input_range=(0, 6),
        ),
        Period(
            name="month",
            n_periods=config.rbf_periods_month,
            column="month",
            input_range=(1, 12),
        ),
    ]
    return ExogBuilder(periods=periods, country_code=config.holiday_country_code)


def build_exog_matrix(
    start: pd.Timestamp,
    end: pd.Timestamp,
    weather_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Build full exog matrix: calendar (RBF + holidays) + weather features."""

    # needed because the 'build' library function works with fully closed intervals
    before_end = end - pd.Timedelta(config.resolution)
    exog_cal = _build_exog_builder(config).build(start, before_end)

    exog = exog_cal.join(weather_df, how="left")

    nan_count = int(exog.isna().sum().sum())
    if nan_count > 0:
        raise ValueError(
            f"Exog matrix has {nan_count} NaN values after calendar+weather join"
        )

    return exog
