from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import load as joblib_load  # pyright: ignore
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore

_console = Console()


def _set_freq(series: pd.Series) -> pd.Series:  # type: ignore[type-arg]
    if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
        inferred = pd.infer_freq(series.index)
        if inferred is not None:
            series = series.copy()
            dti: pd.DatetimeIndex = series.index  # type: ignore[assignment]
            dti.freq = inferred  # type: ignore[assignment]
    return series


def run(config: Config) -> None:
    """Load the trained model and forecast the test-split window using exog_full."""
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
        f"[bold]predict[/bold]  forecasting {config.horizon}h from {cutoff.date()}… "
        f"(exog: {exog_pred.shape[1]} cols)"
    )

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
