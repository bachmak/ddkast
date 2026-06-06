"""Tests for the merge stage's processing of DAF and weather (issue #10).

These tests pin the contract that `merge.run` must put the ENTSO-E day-ahead
forecast (DAF) and the weather frame onto the same hourly-UTC grid as the
cleaned actual load. They are written to FAIL against the current
`merge._passthrough` implementation, which writes both series through
untransformed:

  * ENTSO-E delivers the DAF at 15-min resolution in the bidding-zone local
    time (Europe/Berlin in the fixtures). Passed through, it stays 15-min and
    tz-localized, so it does not line up with the hourly-UTC load grid that
    `evaluate.py` reindexes the DAF benchmark onto (`evaluate.py:46`).
  * Open-Meteo weather may arrive without a timezone; passed through it stays
    tz-naive, which breaks the left-join in `build_exog_matrix`.

Each test maps to an acceptance criterion in issue #10. They are the red phase:
implement the DAF/weather processing in `merge.py` to turn them green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.data.weather import WEATHER_COLS
from ddkast.pipeline import merge

# A week of 15-min samples in Europe/Berlin — how ENTSO-E delivers load + DAF.
_WEEK_BERLIN = pd.date_range(
    "2024-01-01", "2024-01-08", freq="15min", tz="Europe/Berlin"
)


def _smooth_load(index: pd.DatetimeIndex) -> np.ndarray:
    """A smooth, strictly-positive daily curve so clean() drops nothing."""
    hours = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60
    return 50_000.0 + 5_000.0 * np.sin(2 * np.pi * hours / 24)


def _write_raw_inputs(
    config: Config,
    *,
    load_index: pd.DatetimeIndex,
    daf_index: pd.DatetimeIndex | None = None,
    weather_index: pd.DatetimeIndex | None = None,
) -> None:
    """Materialise the three raw artifacts merge.run reads, mirroring download.

    Schemas match tests/fixtures/generate.py: 15-min Europe/Berlin load + DAF
    (columns `load_mw` / `forecast_mw`) and hourly weather over WEATHER_COLS.
    """
    if daf_index is None:
        daf_index = load_index
    if weather_index is None:
        weather_index = pd.date_range("2024-01-01", "2024-01-08", freq="1h", tz="UTC")

    raw = ParquetStore(config.raw_dir)
    raw.write(
        config.raw_load_actual,
        pd.DataFrame({"load_mw": _smooth_load(load_index)}, index=load_index),
    )
    raw.write(
        config.raw_load_forecast,
        pd.DataFrame({"forecast_mw": _smooth_load(daf_index) + 300.0}, index=daf_index),
    )
    raw.write(
        config.raw_weather,
        pd.DataFrame(
            {col: np.linspace(0.0, 100.0, len(weather_index)) for col in WEATHER_COLS},
            index=weather_index,
        ),
    )


def test_merge_resamples_daf_to_hourly(config: Config) -> None:
    """DAF must be resampled from its native 15-min grid to hourly (#10 AC-1)."""
    _write_raw_inputs(config, load_index=_WEEK_BERLIN)

    merge.run(config)

    daf = ParquetStore(config.processed_dir).read(config.processed_entso_forecast)
    diffs = daf.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all(), (
        "merge left the DAF at its native ENTSO-E resolution instead of "
        f"resampling to hourly; observed spacings: {sorted(diffs.unique())}"
    )


def test_merge_converts_daf_index_to_utc(config: Config) -> None:
    """DAF index must be UTC-aware after the merge stage (#10 AC-2)."""
    _write_raw_inputs(config, load_index=_WEEK_BERLIN)

    merge.run(config)

    daf = ParquetStore(config.processed_dir).read(config.processed_entso_forecast)
    assert isinstance(daf.index, pd.DatetimeIndex)
    assert daf.index.tz is not None, "merge dropped the DAF timezone entirely"
    assert str(daf.index.tz) == "UTC", (
        f"DAF index tz is {daf.index.tz}; merge does not convert ENTSO-E's "
        "local-time index to UTC, so it cannot align with the UTC load grid"
    )


def test_merge_puts_daf_on_processed_load_grid(config: Config) -> None:
    """DAF and load must share an identical hourly-UTC index (#10 AC-6).

    evaluate.py aligns the ENTSO-E benchmark via
    ``entso["forecast_mw"].reindex(predictions.index)`` (evaluate.py:46), whose
    index lives on the hourly-UTC load grid. A 15-min/Berlin DAF does not match
    that grid, silently corrupting the published-DAF benchmark.
    """
    _write_raw_inputs(config, load_index=_WEEK_BERLIN)

    merge.run(config)

    store = ParquetStore(config.processed_dir)
    load = store.read(config.processed_load)
    daf = store.read(config.processed_entso_forecast)
    assert daf.index.equals(load.index), (
        "processed DAF is not on the same hourly-UTC grid as processed load; "
        "the ENTSO-E benchmark in evaluate.py will not align with the forecast"
    )


def test_merge_trims_daf_onto_load_grid_without_nan(config: Config) -> None:
    """A narrower DAF is trimmed onto the load grid with no NaN (#10 AC-5/7).

    When DAF covers fewer timestamps than load, every processed-DAF timestamp
    must fall on load's hourly-UTC grid and carry a real value (the hard NaN
    check). Passthrough leaves the off-hour (:15/:30/:45) Berlin stamps, which
    are not on the hourly grid.
    """
    load_index = pd.date_range(
        "2024-01-01", "2024-01-08", freq="15min", tz="Europe/Berlin"
    )
    daf_index = pd.date_range(
        "2024-01-03", "2024-01-06", freq="15min", tz="Europe/Berlin"
    )
    _write_raw_inputs(config, load_index=load_index, daf_index=daf_index)

    merge.run(config)

    store = ParquetStore(config.processed_dir)
    load = store.read(config.processed_load)
    daf = store.read(config.processed_entso_forecast)
    assert daf.index.isin(load.index).all(), (
        "processed DAF has timestamps off the hourly-UTC load grid; merge does "
        "not trim/resample DAF to the load grid"
    )
    assert not bool(daf.isna().any().any()), "processed DAF contains NaN after merge"


def test_merge_localizes_weather_to_utc(config: Config) -> None:
    """Weather must be UTC-aware after the merge stage (#10 AC-3).

    Open-Meteo can return tz-naive timestamps; merge must localize them to UTC
    so the weather left-join in build_exog_matrix lines up. Passthrough leaves
    the index tz-naive.
    """
    naive_weather_index = pd.date_range("2024-01-01", "2024-01-08", freq="1h")
    _write_raw_inputs(
        config, load_index=_WEEK_BERLIN, weather_index=naive_weather_index
    )

    merge.run(config)

    weather = ParquetStore(config.processed_dir).read(config.processed_weather)
    assert isinstance(weather.index, pd.DatetimeIndex)
    assert weather.index.tz is not None, (
        "processed weather index is tz-naive; merge must localize Open-Meteo "
        "weather to UTC before it is joined into the exog matrix"
    )


def test_merge_drops_daf_gap_longer_than_interpolation_limit(config: Config) -> None:
    """A DAF gap > max_interpolation_hours warns and drops; never aborts (#10 AC-3).

    DAF quality is independent of load quality, so an unfillable DAF gap is a
    soft failure: interpolate up to ``max_interpolation_hours``, then drop the
    rows that survive as NaN. Here a 10 h hole (Europe/Berlin == UTC+1 in
    January) lands as UTC 03:00–12:00 on 2024-01-04; with a 3 h interpolation
    limit, the middle of the hole (08:00 UTC) cannot be filled from either edge
    and must be dropped rather than raised on.
    """
    full = _WEEK_BERLIN
    gap_start = pd.Timestamp("2024-01-04 04:00", tz="Europe/Berlin")
    gap_end = pd.Timestamp("2024-01-04 14:00", tz="Europe/Berlin")
    daf_index = full[(full < gap_start) | (full >= gap_end)]
    _write_raw_inputs(config, load_index=full, daf_index=daf_index)

    merge.run(config)  # must not raise — DAF gaps are a soft failure

    store = ParquetStore(config.processed_dir)
    load = store.read(config.processed_load)
    daf = store.read(config.processed_entso_forecast)
    assert daf.index.isin(load.index).all(), (
        "processed DAF is off the hourly-UTC load grid; the gap path must still "
        "resample/trim DAF onto load's grid"
    )
    deep_gap = pd.Timestamp("2024-01-04 08:00", tz="UTC")
    assert deep_gap not in daf.index, (
        "merge kept a DAF hour that sits >max_interpolation_hours into a gap; "
        "the unfillable rows must be dropped, not interpolated or retained"
    )
    assert not bool(daf.isna().any().any()), (
        "processed DAF still contains NaN; the soft-drop after interpolation did "
        "not run"
    )


def test_merge_interpolates_daf_gap_within_interpolation_limit(config: Config) -> None:
    """A DAF gap <= max_interpolation_hours is filled and kept, not dropped (#10 AC-3).

    The complement of test_merge_drops_daf_gap_longer_than_interpolation_limit:
    that test pins the "drop the unfillable rows" half of the policy; this one
    pins the "interpolate up to the limit" half. A 2 h hole (Europe/Berlin ==
    UTC+1 in January) lands as UTC 04:00–05:00 on 2024-01-04; with the default
    3 h limit both hours are within reach of their real neighbours and must be
    linearly filled and retained. A merge that skips interpolation and simply
    drops every resampling NaN would erase these hours and fail here.
    """
    assert config.max_interpolation_hours >= 2, "fixture assumes a >=2 h fill limit"
    full = _WEEK_BERLIN
    gap_start = pd.Timestamp("2024-01-04 05:00", tz="Europe/Berlin")
    gap_end = pd.Timestamp("2024-01-04 07:00", tz="Europe/Berlin")
    daf_index = full[(full < gap_start) | (full >= gap_end)]
    _write_raw_inputs(config, load_index=full, daf_index=daf_index)

    merge.run(config)

    daf = ParquetStore(config.processed_dir).read(config.processed_entso_forecast)
    gap_hours = pd.DatetimeIndex(["2024-01-04 04:00", "2024-01-04 05:00"], tz="UTC")
    assert gap_hours.isin(daf.index).all(), (
        "merge dropped DAF hours that sit within max_interpolation_hours of their "
        "neighbours; a <=limit gap must be interpolated and kept, not dropped"
    )
    filled = daf.loc[gap_hours].iloc[:, 0]
    assert not bool(filled.isna().any()), "interpolated DAF gap hours came back NaN"
    lo = daf.loc[pd.Timestamp("2024-01-04 03:00", tz="UTC")].iloc[0]
    hi = daf.loc[pd.Timestamp("2024-01-04 06:00", tz="UTC")].iloc[0]
    assert min(lo, hi) <= filled.min() and filled.max() <= max(lo, hi), (
        "interpolated DAF values fall outside their bracketing real neighbours; "
        "they do not look linearly interpolated"
    )


def test_merge_raises_on_nan_in_load_after_trimming(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NaN in load on the shared load/DAF grid is a hard error (#10 AC-7).

    Unlike DAF/weather gaps, a NaN in the cleaned load after trimming means the
    cleaning step produced incomplete output, which must fail loudly. The clean
    pipeline never emits such a NaN by construction, so we inject one through
    ``merge.clean`` to exercise the defensive ``ValueError`` branch directly.
    """
    _write_raw_inputs(config, load_index=_WEEK_BERLIN)
    real_clean = merge.clean

    def clean_then_puncture(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        cleaned = real_clean(df, cfg)
        cleaned.iloc[10, 0] = np.nan  # a hole on the hourly-UTC load grid
        return cleaned

    monkeypatch.setattr(merge, "clean", clean_then_puncture)

    with pytest.raises(ValueError, match="NaN"):
        merge.run(config)


def test_merge_resamples_weather_to_hourly(config: Config) -> None:
    """Sub-hourly weather must be resampled to the hourly grid (#10 AC-5).

    The fixtures already deliver hourly weather, so passthrough hides whether
    merge resamples at all. Feeding 15-min weather makes the contract testable:
    the processed weather must come out on a strictly hourly spacing.
    """
    sub_hourly = pd.date_range("2024-01-01", "2024-01-08", freq="15min", tz="UTC")
    _write_raw_inputs(config, load_index=_WEEK_BERLIN, weather_index=sub_hourly)

    merge.run(config)

    weather = ParquetStore(config.processed_dir).read(config.processed_weather)
    diffs = weather.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all(), (
        "merge left weather at its native sub-hourly resolution instead of "
        f"resampling to 1 h; observed spacings: {sorted(diffs.unique())}"
    )


def test_merge_drops_preload_archive_but_keeps_forecast_tail(config: Config) -> None:
    """Weather is lower-bound trimmed; the forecast tail past load is kept (#23).

    Weather is exog on the model path: the latest origin predict forecasts runs
    *beyond* the load tail, which means the weather frame must extend
    past ``load.max`` (its forecast hours, capped at ~now+horizon by
    fetch_weather) or build_exog_matrix gets an all-NaN future exog and raises.
    So merge only drops the over-long archive before the load start — it must
    NOT cap weather at the load grid. Here weather runs from well before load to
    a week past it: the pre-load archive is dropped, the post-load tail survives.
    Trimming stays soft, so an in-range NaN hour is dropped, not raised on.
    """
    weather_index = pd.date_range("2023-12-20", "2024-01-15", freq="1h", tz="UTC")
    _write_raw_inputs(config, load_index=_WEEK_BERLIN)
    weather = pd.DataFrame(
        {col: np.linspace(0.0, 100.0, len(weather_index)) for col in WEATHER_COLS},
        index=weather_index,
    )
    preload_hour = pd.Timestamp("2023-12-25 00:00", tz="UTC")
    nan_hour = pd.Timestamp("2024-01-03 12:00", tz="UTC")
    weather.loc[nan_hour] = np.nan
    ParquetStore(config.raw_dir).write(config.raw_weather, weather)

    merge.run(config)  # must not raise — weather gaps are a soft failure

    store = ParquetStore(config.processed_dir)
    load = store.read(config.processed_load)
    weather_out = store.read(config.processed_weather)
    assert weather_out.index.max() > load.index.max(), (
        "merge capped weather at the load grid; the forecast hours past load.max "
        "must survive so the latest origin predict forecasts has future exog (#23)"
    )
    assert weather_out.index.min() >= load.index.min(), (
        "merge kept weather older than the load start; the over-long pre-load "
        "archive must be lower-bound trimmed off the model frame"
    )
    assert preload_hour not in weather_out.index, (
        "a pre-load archive hour survived the lower-bound trim"
    )
    assert nan_hour not in weather_out.index, (
        "merge kept an in-range weather hour that was NaN; residual weather NaN "
        "must be dropped, not retained"
    )
    assert not bool(weather_out.isna().any().any()), (
        "processed weather still contains NaN after the soft trim/drop"
    )
