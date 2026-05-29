from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
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
