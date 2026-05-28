from __future__ import annotations

import pandas as pd
from joblib import (  # pyright: ignore[reportMissingTypeStubs]
    load as joblib_load,  # pyright: ignore[reportUnknownVariableType]
)
from lightgbm import LGBMRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.persistence import save_forecaster

from ddkast.config import Config
from ddkast.preprocessing.features import build_exog_matrix


def _with_freq(series: pd.Series[float]) -> pd.Series[float]:
    """Ensure the series DatetimeIndex has freq — Parquet and loc-slicing strip it."""
    if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
        inferred = pd.infer_freq(series.index)
        if inferred is not None:
            series = series.copy()
            dti: pd.DatetimeIndex = series.index  # type: ignore[assignment]
            dti.freq = inferred  # type: ignore[assignment]
    return series


def fit(df: pd.DataFrame, weather_df: pd.DataFrame, config: Config) -> None:
    """Fit a recursive forecaster on df and persist it to config.models_dir."""
    series: pd.Series[float] = _with_freq(df[config.model_target])
    exog = build_exog_matrix(series.index.min(), series.index.max(), weather_df, config)

    forecaster: ForecasterRecursive = ForecasterRecursive(
        estimator=LGBMRegressor(
            n_jobs=-1,
            verbose=-1,
            random_state=42,
            deterministic=True,
            force_col_wise=True,
        ),
        lags=config.lags,
    )
    forecaster.fit(y=series, exog=exog)
    save_forecaster(forecaster, config.models_dir, target=config.model_target)


def forecast(
    df: pd.DataFrame, weather_df: pd.DataFrame, config: Config
) -> pd.Series[float]:
    """Load the persisted model and return a forecast of length config.horizon."""
    # save_forecaster stores files as forecaster_{target}.joblib
    model_path = config.models_dir / f"forecaster_{config.model_target}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `ddkast train` first."
        )
    forecaster: ForecasterRecursive = joblib_load(model_path)  # type: ignore[assignment]
    last_ts: pd.Timestamp = df.index[-1]  # type: ignore[assignment]
    forecast_start = last_ts + pd.Timedelta(hours=1)
    forecast_end = last_ts + pd.Timedelta(hours=config.horizon)

    exog_future = build_exog_matrix(forecast_start, forecast_end, weather_df, config)

    return forecaster.predict(steps=config.horizon, exog=exog_future)  # type: ignore[return-value]
