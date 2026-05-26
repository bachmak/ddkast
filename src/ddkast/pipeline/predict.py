from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from joblib import load as joblib_load  # pyright: ignore
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.data.weather import fetch_weather
from ddkast.preprocessing.features import build_exog_matrix

_console = Console()


def _set_freq(series: pd.Series) -> pd.Series:  # type: ignore[type-arg]
    if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
        inferred = pd.infer_freq(series.index)
        if inferred is not None:
            series = series.copy()
            dti: pd.DatetimeIndex = series.index  # type: ignore[assignment]
            dti.freq = inferred  # type: ignore[assignment]
    return series


def run(config: Config, target_date: date | None = None) -> None:
    """Load the trained model and generate a 24-hour forecast.

    Default path (no target_date): forecasts the test-split window using the
    exog matrix persisted by `train`. Fully offline, no API calls.

    Live path (target_date provided): fetches weather forecast from Open-Meteo
    and uses the last 168 h of actual load as the autoregressive window.
    Requires the load data to extend up to target_date - 1 day.
    """
    processed = ParquetStore(config.processed_dir)
    clean_load: pd.Series[float] = processed.read(config.processed_load)[
        config.model_target
    ]  # type: ignore[assignment]

    model_path = Path(config.models_dir) / f"forecaster_{config.model_target}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `ddkast train` first."
        )
    forecaster = joblib_load(model_path)  # pyright: ignore[reportUnknownVariableType]

    if target_date is not None:
        # --- LIVE PATH ---------------------------------------------------
        target_start = pd.Timestamp(target_date, tz="UTC")
        target_end = target_start + pd.Timedelta(hours=config.horizon - 1)
        target_index = pd.date_range(target_start, target_end, freq="h")

        # Continuity check: load must end before the target day
        actual_last_hour: pd.Timestamp = clean_load.index[-1]  # type: ignore[assignment]
        if actual_last_hour >= target_start:
            raise ValueError(
                f"Load data extends into the target day ({actual_last_hour}). "
                "This would leak future data into last_window."
            )
        if target_start - actual_last_hour > pd.Timedelta(hours=config.lags):
            raise ValueError(
                f"Last load hour {actual_last_hour} is more than {config.lags}h "
                f"before target {target_date}. Download more recent data first."
            )

        last_window = _set_freq(clean_load.iloc[-config.lags :].copy())

        weather_cache = config.weather_cache_dir / "weather_forecast.parquet"
        weather_forecast = fetch_weather(
            start=target_start,
            end=target_end,
            latitude=config.weather_latitude,
            longitude=config.weather_longitude,
            cache_path=weather_cache,
            use_forecast=True,
        )
        exog_pred = build_exog_matrix(
            start=target_start,
            end=target_end,
            weather_df=weather_forecast,
            config=config,
        )

        _console.print(
            f"[bold]predict[/bold]  live forecast for {target_date} "
            f"({config.horizon}h, exog: {exog_pred.shape[1]} cols)"
        )

    else:
        # --- DEFAULT PATH (test-split evaluation) -------------------------
        cutoff: pd.Timestamp = clean_load.index[-1] - pd.Timedelta(  # type: ignore[assignment]
            days=config.test_days
        )
        train_end: pd.Timestamp = clean_load.loc[:cutoff].index[-1]  # type: ignore[assignment]

        last_window = _set_freq(clean_load.loc[:train_end].iloc[-config.lags :].copy())

        target_start = train_end + pd.Timedelta(hours=1)
        target_end = target_start + pd.Timedelta(hours=config.horizon - 1)
        target_index = pd.date_range(target_start, target_end, freq="h")

        exog_full = processed.read(config.processed_exog)
        exog_pred = exog_full.loc[target_index]

        _console.print(
            f"[bold]predict[/bold]  forecasting {config.horizon}h from {cutoff.date()}… "  # noqa: E501
            f"(exog: {exog_pred.shape[1]} cols)"
        )

    # --- SHARED: predict + validate + persist ----------------------------
    y_pred: pd.Series[float] = forecaster.predict(  # type: ignore[union-attr]
        steps=config.horizon,
        last_window=last_window,
        exog=exog_pred,
    )
    y_pred.name = config.model_target
    y_pred.index = target_index

    if len(y_pred) != config.horizon:
        raise ValueError(f"Expected {config.horizon} forecast rows, got {len(y_pred)}")
    if not (y_pred > 0).all():
        raise ValueError("Forecast contains non-positive values")

    predictions_df = y_pred.to_frame(name=config.model_target)
    processed.write(config.processed_predictions, predictions_df)

    _console.print(
        f"  [green]✓[/green] {len(y_pred)} predictions "
        f"({y_pred.index[0]} → {y_pred.index[-1]}) "
        f"→ {config.processed_dir / config.processed_predictions}.parquet"
    )

    # Write submission CSV for the challenge leaderboard (live path only)
    if target_date is not None:
        sub_dir = config.data_dir / "submissions" / config.team_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_path = sub_dir / f"{target_date.isoformat()}.csv"
        pd.DataFrame(
            {
                "timestamp_utc": [
                    ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in y_pred.index
                ],
                "forecast_mw": y_pred.to_numpy(),
            }
        ).to_csv(sub_path, index=False)
        _console.print(f"  [green]✓[/green] submission -> {sub_path}")
