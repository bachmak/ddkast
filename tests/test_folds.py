from __future__ import annotations

import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.folds import build_folds


def _index(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")


def _config(
    idx: pd.DatetimeIndex,
    historical_folds: int,
    *,
    lags: int = 24,
    horizon: int = 24,
    stride_h: int = 24,
) -> Config:
    """``historical_folds`` historical origins plus the live origin at ``idx[-1]``.

    Mirrors the old index-anchored geometry: origins step ``stride_h`` apart, ending on
    the data tail.
    """
    end: pd.Timestamp = idx[-1]
    start = end - pd.Timedelta(hours=historical_folds * stride_h)
    return Config(  # type: ignore[call-arg]
        entsoe_api_key="x",
        lags=lags,
        horizon=horizon,
        n_forecasts=historical_folds + 1,
        forecasts_start=start.to_pydatetime(),
        forecasts_end=end.to_pydatetime(),
    )


def test_origins_step_by_stride_earliest_first() -> None:
    idx = _index(24 * 10)
    folds = build_folds(idx, _config(idx, 3))
    last_ts = idx[-1]
    assert [f.origin for f in folds] == [
        last_ts - pd.Timedelta(hours=72),
        last_ts - pd.Timedelta(hours=48),
        last_ts - pd.Timedelta(hours=24),
        last_ts,
    ]
    assert [f.origin for f in folds] == sorted(f.origin for f in folds)


def test_forecast_block_anchored_on_origin() -> None:
    idx = _index(24 * 10)
    folds = build_folds(idx, _config(idx, 3))
    for f in folds:
        assert f.forecast_start == f.origin + pd.Timedelta(hours=1)
        assert f.forecast_end == f.origin + pd.Timedelta(hours=24)
    # Every fold but the last ends on an actual; only the latest runs past the tail.
    for f in folds[:-1]:
        assert f.forecast_end <= idx[-1]


def test_latest_origin_is_forecasts_end() -> None:
    idx = _index(24 * 10)
    folds = build_folds(idx, _config(idx, 3))
    latest = folds[-1]
    assert len(folds) == 4  # historical_folds + 1
    assert latest.origin == idx[-1]
    assert latest.forecast_start == idx[-1] + pd.Timedelta(hours=1)
    assert latest.forecast_end == idx[-1] + pd.Timedelta(hours=24)


def test_single_fold_yields_only_its_origin() -> None:
    idx = _index(24 * 10)
    folds = build_folds(
        idx, _config(idx, 0)
    )  # n_forecasts=1, forecasts_start == forecasts_end
    assert len(folds) == 1
    assert folds[0].origin == idx[-1]


def test_forecast_start_uses_resolution_step() -> None:
    # A 15-min grid: the forecast starts one resolution step past the origin, not +1h.
    idx = pd.date_range("2024-01-01", periods=4 * 24 * 8, freq="15min", tz="UTC")
    cfg = _config(idx, 0).model_copy(update={"resolution": "15min"})
    fold = build_folds(idx, cfg)[0]
    assert fold.forecast_start == fold.origin + pd.Timedelta(minutes=15)
    assert fold.forecast_end == fold.origin + pd.Timedelta(hours=24)


def test_fold_ids_are_unique_and_deterministic() -> None:
    idx = _index(24 * 10)
    cfg = _config(idx, 3)
    a = build_folds(idx, cfg)
    b = build_folds(idx, cfg)
    ids = [f.fold_id for f in a]
    assert ids == [f.fold_id for f in b]  # deterministic
    assert len(set(ids)) == len(ids)  # unique


def test_raises_on_insufficient_history() -> None:
    idx = _index(24 * 3)  # 72h of data
    # earliest origin needs lags=48h of history before it, but 3 folds eat 72h.
    with pytest.raises(ValueError, match="Insufficient history"):
        build_folds(idx, _config(idx, 3, lags=48))


def test_raises_on_empty_index() -> None:
    empty = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
    with pytest.raises(ValueError, match="empty index"):
        build_folds(empty, _config(_index(24 * 10), 1))


def test_raises_on_unsorted_index() -> None:
    idx = _index(24 * 10)
    with pytest.raises(ValueError, match="sorted ascending"):
        build_folds(idx[::-1], _config(idx, 1))


def test_raises_on_zero_folds() -> None:
    idx = _index(24 * 10)
    with pytest.raises(ValueError, match="n_forecasts must be >= 1"):
        build_folds(idx, _config(idx, 1).model_copy(update={"n_forecasts": 0}))


def test_raises_when_single_fold_window_is_not_a_point() -> None:
    idx = _index(24 * 10)
    cfg = _config(idx, 0).model_copy(
        update={"forecasts_end": (idx[-1] + pd.Timedelta(hours=24)).to_pydatetime()}
    )
    with pytest.raises(ValueError, match="forecasts_start == forecasts_end"):
        build_folds(idx, cfg)


def test_raises_when_window_not_evenly_divisible() -> None:
    idx = _index(24 * 10)
    # A 1h span cannot be split into 7 equal whole-nanosecond steps for n_forecasts=8.
    cfg = Config(  # type: ignore[call-arg]
        entsoe_api_key="x",
        lags=24,
        horizon=24,
        n_forecasts=8,
        forecasts_start=(idx[-1] - pd.Timedelta(hours=1)).to_pydatetime(),
        forecasts_end=idx[-1].to_pydatetime(),
    )
    with pytest.raises(ValueError, match="not divisible into 7 equal steps"):
        build_folds(idx, cfg)


def test_raises_when_stride_off_resolution_grid() -> None:
    idx = _index(24 * 10)
    # 3h span / 2 steps = 90min stride, not a whole multiple of the 1h grid.
    cfg = Config(  # type: ignore[call-arg]
        entsoe_api_key="x",
        lags=24,
        horizon=24,
        n_forecasts=3,
        forecasts_start=(idx[-1] - pd.Timedelta(hours=3)).to_pydatetime(),
        forecasts_end=idx[-1].to_pydatetime(),
    )
    with pytest.raises(ValueError, match="whole multiple"):
        build_folds(idx, cfg)


def test_raises_on_nonpositive_resolution() -> None:
    idx = _index(24 * 10)
    with pytest.raises(ValueError, match="resolution must be a positive duration"):
        build_folds(idx, _config(idx, 1).model_copy(update={"resolution": "0h"}))
