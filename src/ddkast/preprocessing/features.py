from __future__ import annotations

from spotforecast2_safe import ExogBuilder
from spotforecast2_safe.data.data import Period

from ddkast.config import Config


def build_exog_builder(config: Config) -> ExogBuilder:
    """Return a configured ExogBuilder: RBF cyclical, holidays, weekend flag."""
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
