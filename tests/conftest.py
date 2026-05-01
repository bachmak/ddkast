from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        entsoe_api_key="test_key",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def load_series() -> pd.Series:  # type: ignore[type-arg]
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    values = rng.uniform(30_000, 70_000, len(idx))
    return pd.Series(values, index=idx, name="load_mw")
