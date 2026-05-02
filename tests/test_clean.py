from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.preprocessing.clean import clean


def _make_df(series: pd.Series) -> pd.DataFrame:  # type: ignore[type-arg]
    return pd.DataFrame({"load_mw": series})


@pytest.fixture
def hourly_series(config: Config) -> pd.Series:  # type: ignore[type-arg]
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    values = rng.uniform(40_000, 60_000, len(idx))
    return pd.Series(values, index=idx, name="load_mw")


def test_clean_passthrough(hourly_series: pd.Series, config: Config) -> None:
    df = _make_df(hourly_series)
    result = clean(df, config)
    assert result["load_mw"].isna().sum() == 0
    assert len(result) == len(hourly_series)


def test_clean_interpolates_short_gap(hourly_series: pd.Series, config: Config) -> None:
    series = hourly_series.copy()
    series.iloc[10:12] = np.nan  # 2-hour gap — within tolerance
    df = _make_df(series)
    result = clean(df, config)
    assert result["load_mw"].isna().sum() == 0


def test_clean_rejects_long_gap(hourly_series: pd.Series, config: Config) -> None:
    series = hourly_series.copy()
    series.iloc[10:15] = np.nan  # 5-hour gap — exceeds default tolerance of 3
    df = _make_df(series)
    with pytest.raises(ValueError, match="unresolvable"):
        clean(df, config)


def test_clean_removes_outliers(hourly_series: pd.Series, config: Config) -> None:
    series = hourly_series.copy()
    series.iloc[50] = 999_999.0  # extreme spike well outside IQR * 3
    df = _make_df(series)
    result = clean(df, config)
    assert result["load_mw"].iloc[50] < 999_999.0
    assert result["load_mw"].isna().sum() == 0


def test_clean_resamples_subhourly(config: Config) -> None:
    idx = pd.date_range("2024-01-01", periods=24 * 4, freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    values = rng.uniform(40_000, 60_000, len(idx))
    df = pd.DataFrame({"load_mw": pd.Series(values, index=idx)})
    result = clean(df, config)
    assert len(result) == 24  # 15-min → 1h: 96 rows → 24 rows


def test_clean_handles_tz_naive_index(hourly_series: pd.Series, config: Config) -> None:
    series = hourly_series.copy()
    series.index = series.index.tz_localize(None)
    df = _make_df(series)
    result = clean(df, config)
    assert result.index.tz is not None
    assert result["load_mw"].isna().sum() == 0
