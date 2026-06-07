"""Unit tests for the evaluate stage's pure helpers and CR-3 fail-safe branches."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.folds import Fold
from ddkast.pipeline import evaluate
from ddkast.pipeline.evaluate import (
    _align_fold,
    _fold_is_scorable,
    _metric_rows,
    flatten_summary,
    summarize,
)
from tests.synthetic import fold_window

_FULL = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
_TARGET = pd.Series(np.arange(1, 301, dtype=float), index=_FULL)  # strictly positive


_STEP = pd.Timedelta(hours=1)


def _fold_at(window: pd.DatetimeIndex) -> Fold:
    # Half-open block: forecast_end is one step past the window's last realized stamp.
    return Fold(
        fold_id="f",
        forecast_start=window[0],
        forecast_end=window[-1] + _STEP,
    )


def test_fold_is_scorable_true_when_block_realized() -> None:
    window = _FULL[200:204]
    assert _fold_is_scorable(_fold_at(window), _TARGET.index[-1] + _STEP) is True


def test_fold_is_scorable_false_when_block_future() -> None:
    window = pd.date_range(
        "2024-06-01", periods=4, freq="1h", tz="UTC"
    )  # entirely past the actuals → not yet scorable
    assert _fold_is_scorable(_fold_at(window), _TARGET.index[-1] + _STEP) is False


def test_fold_is_scorable_raises_on_straddle() -> None:
    window = _FULL[200:204]
    last_actual = window[2]  # block starts before, ends after the last actual
    with pytest.raises(ValueError, match="straddles the actual end"):
        _fold_is_scorable(_fold_at(window), last_actual + _STEP)


def test_align_fold_happy_path() -> None:
    window = _FULL[200:204]
    preds = pd.Series([10.0, 11.0, 12.0, 13.0], index=window)
    aligned = _align_fold(_fold_at(window), preds, _TARGET, _TARGET)
    assert aligned.daf_available
    # naive looks up exactly 168h earlier.
    assert aligned.naive.iloc[0] == _TARGET.loc[window[0] - pd.Timedelta(hours=168)]


def test_align_fold_raises_on_missing_naive_lookback() -> None:
    window = _FULL[10:14]  # actuals present, but 168h before is outside the data range
    preds = pd.Series([1.0, 2.0, 3.0, 4.0], index=window)
    with pytest.raises(ValueError, match="naive"):
        _align_fold(_fold_at(window), preds, _TARGET, _TARGET)


def test_metric_rows_skips_daf_when_unavailable() -> None:
    window = _FULL[200:204]
    preds = pd.Series([10.0, 11.0, 12.0, 13.0], index=window)
    daf = _TARGET.copy()
    daf.loc[window[1]] = np.nan  # one missing DAF point → DAF not scored this fold
    aligned = _align_fold(_fold_at(window), preds, _TARGET, daf)
    assert not aligned.daf_available
    rows = _metric_rows(_fold_at(window), aligned)
    assert {r["forecaster"] for r in rows} == {"model", "naive"}


def test_summarize_and_flatten() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "forecaster": "model",
                "MAE": 100.0,
                "RMSE": 1.0,
                "MAPE": 1.0,
                "SMAPE": 1.0,
            },
            {
                "forecaster": "model",
                "MAE": 200.0,
                "RMSE": 1.0,
                "MAPE": 1.0,
                "SMAPE": 1.0,
            },
            {
                "forecaster": "naive",
                "MAE": 300.0,
                "RMSE": 1.0,
                "MAPE": 1.0,
                "SMAPE": 1.0,
            },
            {
                "forecaster": "naive",
                "MAE": 300.0,
                "RMSE": 1.0,
                "MAPE": 1.0,
                "SMAPE": 1.0,
            },
            {"forecaster": "daf", "MAE": 50.0, "RMSE": 1.0, "MAPE": 1.0, "SMAPE": 1.0},
            {"forecaster": "daf", "MAE": 150.0, "RMSE": 1.0, "MAPE": 1.0, "SMAPE": 1.0},
        ]
    )
    summary = summarize(metrics_df)
    flat = flatten_summary(summary)
    assert flat["model_MAE_mean"] == 150.0
    assert flat["naive_MAE_mean"] == 300.0
    # skill_vs_naive = 1 - 150/300 = 0.5
    assert flat["skill_vs_naive"] == pytest.approx(0.5)
    # skill_vs_daf = 1 - 150/100 = -0.5
    assert flat["skill_vs_daf"] == pytest.approx(-0.5)


def test_run_raises_when_no_scorable_folds(
    config: Config, write_processed: Callable[[Config, int], None]
) -> None:
    cfg = config.model_copy(
        update={"lags": 24, "horizon": 24, **fold_window(24 * 10, 0)}
    )
    write_processed(cfg, 24 * 10)
    with pytest.raises(ValueError, match="No scorable folds"):
        evaluate.run(cfg)
