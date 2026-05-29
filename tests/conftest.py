from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.weather import WEATHER_COLS


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        entsoe_api_key="test_key",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def weather_cols() -> list[str]:
    return list(WEATHER_COLS)


@pytest.fixture
def make_weather() -> Callable[..., pd.DataFrame]:
    """Factory for a synthetic hourly weather frame (tz-aware UTC, 15 columns).

    The index spans [start, end] inclusive at hourly frequency so it joins
    cleanly with the calendar exog ExogBuilder.build produces for the same range.
    """

    def _make(start: pd.Timestamp, end: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="1h", tz="UTC")
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {col: rng.uniform(0.0, 100.0, len(idx)) for col in WEATHER_COLS},
            index=idx,
        )

    return _make


@pytest.fixture
def load_series() -> pd.Series:  # type: ignore[type-arg]
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    values = rng.uniform(30_000, 70_000, len(idx))
    return pd.Series(values, index=idx, name="load_mw")
