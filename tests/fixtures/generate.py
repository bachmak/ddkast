"""Generate the offline smoke-test fixtures on demand.

Deterministic, seeded synthetic data that matches the `download` stage's output
schema exactly, so the offline CI smoke test can run the whole CLI pipeline
without touching ENTSO-E or Open-Meteo. The fixtures are generated on demand and
gitignored — never committed; this module is the single source of truth, used by
the pytest session fixture and by scripts/smoke_test.sh. Run from the project
root to materialise them for manual inspection:

    uv run python tests/fixtures/generate.py [out_dir]

Writes three parquet files into out_dir (default tests/fixtures/smoke/):
  - load_actual.parquet    15-min Europe/Berlin, column `load_mw`
  - load_forecast.parquet  same index,           column `forecast_mw`
  - weather_raw.parquet    hourly UTC,            WEATHER_COLS

The window is deliberately wider than the smoke test's DOWNLOAD_* window so the
FixtureDataSource [start, end) slice has something to trim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.data.weather import WEATHER_COLS

_SEED = 20240101
_DEFAULT_DIR = Path(__file__).parent / "smoke"
# Wider than the smoke window (2024-01-01 → 2024-01-18) on both sides — the extra
# span gives the rolling-origin folds room for their 7-day naive lookback and the
# latest origin's forecast-weather tail past the load grid.
_START = "2023-12-31"
_END = "2024-01-23"

# Realistic Germany load (MW); kept smooth + positive so clean() drops nothing.
_BASE_MW = 52_000.0
_DAILY_AMP = 8_000.0
_WEEKLY_AMP = 5_000.0
_NOISE_STD = 400.0
_FORECAST_BIAS = 300.0
_FORECAST_NOISE_STD = 500.0


def _synthetic_load(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    hours = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60
    dow = index.dayofweek.to_numpy(dtype=float)  # 0=Mon … 6=Sun

    # Daily double-peak (morning + evening), weekend dip, light noise.
    daily = (
        -np.cos(2 * np.pi * hours / 24) * _DAILY_AMP * 0.6
        + np.sin(2 * np.pi * hours / 24) * _DAILY_AMP * 0.4
    )
    weekly = np.where(dow >= 5, -_WEEKLY_AMP, 0.0)
    noise = rng.normal(0.0, _NOISE_STD, len(index))
    return _BASE_MW + daily + weekly + noise


def make_weather(start: pd.Timestamp, end: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly weather frame covering [start, end] at UTC, all WEATHER_COLS."""
    idx = pd.date_range(start, end, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {col: rng.uniform(0.0, 100.0, len(idx)) for col in WEATHER_COLS},
        index=idx,
    )


def generate(out_dir: Path, config: Config) -> None:
    """Write the three smoke fixtures into out_dir (deterministic, seeded)."""
    store = ParquetStore(out_dir)
    rng = np.random.default_rng(_SEED)

    # 15-min Europe/Berlin load + forecast (download produces 15-min Berlin).
    load_index = pd.date_range(start=_START, end=_END, freq="15min", tz="Europe/Berlin")
    load_values = _synthetic_load(load_index, rng)
    store.write(
        config.raw_load_actual,
        pd.DataFrame({"load_mw": load_values}, index=load_index),
    )
    forecast_values = (
        load_values
        + _FORECAST_BIAS
        + rng.normal(0.0, _FORECAST_NOISE_STD, len(load_index))
    )
    store.write(
        config.raw_load_forecast,
        pd.DataFrame({"forecast_mw": forecast_values}, index=load_index),
    )

    # Hourly UTC weather.
    store.write(
        config.raw_weather,
        make_weather(
            pd.Timestamp(_START, tz="UTC"), pd.Timestamp(_END, tz="UTC"), seed=_SEED
        ),
    )


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DIR
    generate(out_dir, Config(entsoe_api_key="unused-by-generate"))
    print(f"wrote fixtures to {out_dir}/")


if __name__ == "__main__":
    main()
