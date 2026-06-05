from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.data.weather import WEATHER_COLS
from tests.fixtures.generate import generate
from tests.fixtures.generate import make_weather as _make_weather


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        entsoe_api_key="test_key",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
    )


@pytest.fixture(scope="session")
def smoke_fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the offline smoke fixtures once per session (gitignored, on demand)."""
    out_dir = tmp_path_factory.mktemp("smoke_fixtures")
    generate(out_dir, Config(entsoe_api_key="test_key"))
    return out_dir


@pytest.fixture
def fixtures_config(config: Config, smoke_fixtures_dir: Path) -> Config:
    """A fixtures-mode Config pointed at the freshly generated smoke fixtures."""
    return config.model_copy(
        update={"data_source": "fixtures", "fixtures_dir": smoke_fixtures_dir}
    )


@pytest.fixture
def weather_cols() -> list[str]:
    return list(WEATHER_COLS)


@pytest.fixture
def make_weather() -> Callable[..., pd.DataFrame]:
    return _make_weather


@pytest.fixture
def load_series() -> pd.Series:  # type: ignore[type-arg]
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    values = rng.uniform(30_000, 70_000, len(idx))
    return pd.Series(values, index=idx, name="load_mw")


@pytest.fixture
def write_processed(
    make_weather: Callable[..., pd.DataFrame],
) -> Callable[[Config, int], None]:
    """Seed the processed store with deterministic, seasonal synthetic data.

    Writes clean load + ENTSO-E DAF + weather so the fold-based train→predict→evaluate
    stages can run offline. Load carries daily + weekly cycles plus seeded noise, kept
    strictly positive so MAPE/SMAPE denominators are safe. Weather extends ``horizon``
    hours past the load tail so the latest origin predict forecasts has future exog.
    """

    def _write(config: Config, periods: int, seed: int = 99) -> None:
        processed = ParquetStore(config.processed_dir)
        idx = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
        rng = np.random.default_rng(seed)
        t = np.arange(periods, dtype=float)
        load_values = (
            50_000.0
            + 8_000.0 * np.sin(2 * np.pi * t / 24)
            + 4_000.0 * np.sin(2 * np.pi * t / 168)
            + rng.normal(0.0, 1_000.0, periods)
        )
        processed.write(
            config.processed_load,
            pd.DataFrame({config.model_target: load_values}, index=idx),
        )
        daf_values = load_values + rng.normal(0.0, 500.0, periods)
        processed.write(
            config.processed_entso_forecast,
            pd.DataFrame({"forecast_mw": daf_values}, index=idx),
        )
        weather_end = idx[-1] + pd.Timedelta(hours=config.horizon)
        processed.write(
            config.processed_weather, make_weather(idx[0], weather_end, seed=123)
        )

    return _write
