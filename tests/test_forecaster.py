from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.models.forecaster import fit, forecast


@pytest.fixture
def training_df(config: Config) -> pd.DataFrame:
    # 200 hours is enough for lags=168 + a few training samples
    idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    values = rng.uniform(40_000, 60_000, len(idx))
    return pd.DataFrame({config.model_target: values}, index=idx)


@pytest.fixture
def small_config(config: Config) -> Config:
    # Use tiny lags so the test runs fast
    return config.model_copy(update={"lags": 5})


@pytest.fixture
def weather_df(
    training_df: pd.DataFrame,
    small_config: Config,
    make_weather: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    # Cover the training window plus the forecast horizon so both
    # fit() and forecast() find weather for every timestamp they need.
    start = training_df.index[0]
    end = training_df.index[-1] + pd.Timedelta(hours=small_config.horizon)
    return make_weather(start, end, seed=11)


def test_fit_creates_model_file(
    training_df: pd.DataFrame, weather_df: pd.DataFrame, small_config: Config
) -> None:
    fit(training_df, weather_df, small_config)
    model_path = (
        small_config.models_dir / f"forecaster_{small_config.model_target}.joblib"
    )
    assert model_path.exists()


def test_forecast_returns_correct_length(
    training_df: pd.DataFrame, weather_df: pd.DataFrame, small_config: Config
) -> None:
    fit(training_df, weather_df, small_config)
    result = forecast(training_df, weather_df, small_config)
    assert len(result) == small_config.horizon


def test_forecast_index_starts_after_training(
    training_df: pd.DataFrame, weather_df: pd.DataFrame, small_config: Config
) -> None:
    fit(training_df, weather_df, small_config)
    result = forecast(training_df, weather_df, small_config)
    expected_start = training_df.index[-1] + pd.Timedelta(hours=1)
    assert result.index[0] == expected_start


def test_forecast_raises_without_model(
    training_df: pd.DataFrame, weather_df: pd.DataFrame, small_config: Config
) -> None:
    with pytest.raises(FileNotFoundError, match="ddkast train"):
        forecast(training_df, weather_df, small_config)


def test_forecast_values_are_finite(
    training_df: pd.DataFrame, weather_df: pd.DataFrame, small_config: Config
) -> None:
    fit(training_df, weather_df, small_config)
    result = forecast(training_df, weather_df, small_config)
    assert np.isfinite(result.values).all()
