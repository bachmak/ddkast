from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import (  # pyright: ignore[reportMissingTypeStubs]
    load as joblib_load,  # pyright: ignore[reportUnknownVariableType]
)
from lightgbm import LGBMRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.persistence import save_forecaster

from ddkast.config import Config
from ddkast.preprocessing.features import build_exog_matrix

# Fixed seed pinned into the model for determinism (CR-2).
RANDOM_STATE = 42


def _with_freq(series: pd.Series[float]) -> pd.Series[float]:
    """Ensure the series DatetimeIndex has freq — Parquet and loc-slicing strip it."""
    if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
        inferred = pd.infer_freq(series.index)
        if inferred is not None:
            series = series.copy()
            dti: pd.DatetimeIndex = series.index  # type: ignore[assignment]
            dti.freq = inferred  # type: ignore[assignment]
    return series


def _model_path(model_dir: Path, config: Config) -> Path:
    """Where ``save_forecaster`` writes / ``forecast`` reads a fold's model."""
    return model_dir / f"forecaster_{config.model_target}.joblib"


def fit(
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    config: Config,
    model_dir: Path,
) -> None:
    """Fit a recursive forecaster on ``load_df`` and persist it under ``model_dir``.

    ``load_df`` is the training window the caller already sliced to ``<= origin``; the
    fitted forecaster bakes in its trailing ``lags`` values, so ``forecast`` needs no
    actuals afterwards. Per-fold isolation comes purely from ``model_dir``.
    """
    series: pd.Series[float] = _with_freq(load_df[config.model_target])
    exog = build_exog_matrix(
        series.index.min(),
        series.index.max() + pd.Timedelta(config.resolution),
        weather_df,
        config,
    )

    forecaster: ForecasterRecursive = ForecasterRecursive(
        estimator=LGBMRegressor(
            n_jobs=-1,
            verbose=-1,
            random_state=RANDOM_STATE,
            deterministic=True,
            force_col_wise=True,
        ),
        lags=config.lags,
    )
    forecaster.fit(y=series, exog=exog)
    save_forecaster(forecaster, model_dir, target=config.model_target)


def forecast(
    weather_df: pd.DataFrame,
    config: Config,
    model_dir: Path,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> pd.Series[float]:
    """Load the fold's model and forecast over ``[forecast_start, forecast_end]``.

    Reads no actuals — the lag window lives inside the fitted forecaster; only the
    future exog (calendar + weather) for the forecast block is built here.
    """
    model_path = _model_path(model_dir, config)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `ddkast train` first."
        )
    forecaster: ForecasterRecursive = joblib_load(model_path)  # type: ignore[assignment]

    exog_future = build_exog_matrix(forecast_start, forecast_end, weather_df, config)
    steps = len(exog_future)

    return forecaster.predict(steps=steps, exog=exog_future)  # type: ignore[return-value]
