from __future__ import annotations

import pandas as pd
from spotforecast2_safe.preprocessing import agg_and_resample_data
from spotforecast2_safe.preprocessing.outlier import manual_outlier_removal

from ddkast.config import Config


def clean(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Curate raw load data: resample, remove outliers, interpolate short gaps.

    Raises:
        ValueError: if any gap longer than config.max_interpolation_hours remains
                    after outlier removal — fail-safe, no silent imputation.
    """
    # Normalise to UTC so DST transitions don't create duplicate/missing timestamps
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.tz_convert("UTC") if df.index.tz is not None else df.tz_localize("UTC")

    # Resample to configured resolution; averages sub-hourly data and duplicates
    df = agg_and_resample_data(df, rule=config.resolution)

    # IQR-based outlier detection — deterministic and interpretable
    col = config.model_target
    series: pd.Series[float] = df[col]
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - config.outlier_iqr_multiplier * iqr
    upper = q3 + config.outlier_iqr_multiplier * iqr
    df, _ = manual_outlier_removal(
        df, column=col, lower_threshold=lower, upper_threshold=upper
    )

    # Interpolate gaps up to max_interpolation_hours; longer gaps remain as NaN
    df[col] = df[col].interpolate(method="linear", limit=config.max_interpolation_hours)

    # Fail-safe: any remaining NaN means a gap was too long to impute
    remaining = int(df[col].isna().sum())
    if remaining:
        raise ValueError(
            f"{remaining} unresolvable missing value(s) after interpolating gaps "
            f"up to {config.max_interpolation_hours} hours. "
            "Inspect the source data or increase `max_interpolation_hours`."
        )

    return df
