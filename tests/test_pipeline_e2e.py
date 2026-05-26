"""End-to-end smoke test for the train → predict → evaluate pipeline stages.
Uses synthetic data written directly to the processed store — no API call needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.pipeline import evaluate, predict, train

_WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]


@pytest.fixture
def e2e_config(config: Config) -> Config:
    # Tiny lags and short test window so the test stays fast
    return config.model_copy(update={"lags": 5, "test_days": 2, "horizon": 6})


@pytest.fixture
def synthetic_processed(e2e_config: Config) -> None:
    """Write synthetic clean load + ENTSO-E DAF + weather to the processed store."""
    processed = ParquetStore(e2e_config.processed_dir)

    # 7-day naive needs data 168h before the first forecast timestamp.
    # With test_days=2, forecast starts at hour N-48; naive needs N-216.
    # 250 hours guarantees the lookback window is covered.
    idx = pd.date_range("2024-01-01", periods=250, freq="1h", tz="UTC")
    rng = np.random.default_rng(99)
    load_values = rng.uniform(40_000, 60_000, len(idx))
    load_df = pd.DataFrame({e2e_config.model_target: load_values}, index=idx)
    processed.write(e2e_config.processed_load, load_df)

    daf_values = load_values + rng.normal(0, 500, 250)
    daf_df = pd.DataFrame({"forecast_mw": daf_values}, index=idx)
    processed.write(e2e_config.processed_entso_forecast, daf_df)

    weather_df = pd.DataFrame(
        rng.uniform(0, 20, (len(idx), len(_WEATHER_COLS))),
        index=idx,
        columns=_WEATHER_COLS,
    )
    processed.write(e2e_config.processed_weather, weather_df)


def test_train_creates_model_and_test_split(
    synthetic_processed: None, e2e_config: Config
) -> None:
    train.run(e2e_config)
    model_path = e2e_config.models_dir / f"forecaster_{e2e_config.model_target}.joblib"
    test_path = e2e_config.processed_dir / f"{e2e_config.processed_test}.parquet"
    assert model_path.exists()
    assert test_path.exists()


def test_predict_creates_predictions(
    synthetic_processed: None, e2e_config: Config
) -> None:
    train.run(e2e_config)
    predict.run(e2e_config)
    predictions_path = (
        e2e_config.processed_dir / f"{e2e_config.processed_predictions}.parquet"
    )
    assert predictions_path.exists()

    processed = ParquetStore(e2e_config.processed_dir)
    preds = processed.read(e2e_config.processed_predictions)
    assert len(preds) == e2e_config.horizon


def test_evaluate_runs_without_error(
    synthetic_processed: None,
    e2e_config: Config,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train.run(e2e_config)
    predict.run(e2e_config)
    evaluate.run(e2e_config)  # should not raise


# TODO: add metric quality gates to this test so CI enforces stable forecasting quality:
#   1. assert model MAE < naive MAE on synthetic data (model must beat the baseline)
#   2. assert model MAE is within X% of a stored snapshot (catch silent regressions)
# Since spotforecast2-safe is deterministic and synthetic data is seeded, results are
# reproducible — making this a reliable PR gate without an API key.
