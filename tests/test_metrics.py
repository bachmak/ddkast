from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddkast.evaluation.metrics import mae, mape, rmse, smape


def _series(values: list[float]) -> pd.Series:  # type: ignore[type-arg]
    return pd.Series(values)


def test_mae_perfect_forecast() -> None:
    s = _series([1.0, 2.0, 3.0])
    assert mae(s, s) == 0.0


def test_mae_known_value() -> None:
    actual = _series([10.0, 20.0])
    predicted = _series([12.0, 18.0])
    assert mae(actual, predicted) == 2.0


def test_rmse_perfect_forecast() -> None:
    s = _series([1.0, 2.0, 3.0])
    assert rmse(s, s) == 0.0


def test_mape_perfect_forecast() -> None:
    s = _series([10.0, 20.0, 30.0])
    assert mape(s, s) == 0.0


def test_smape_perfect_forecast() -> None:
    s = _series([10.0, 20.0, 30.0])
    assert smape(s, s) == 0.0


# --- CR-3 fail-safe: invalid inputs raise rather than impute ---


@pytest.mark.parametrize("metric", [mae, rmse, mape, smape])
def test_raises_on_nan(metric) -> None:  # type: ignore[no-untyped-def]
    actual = _series([10.0, 20.0, 30.0])
    predicted = _series([10.0, np.nan, 30.0])
    with pytest.raises(ValueError, match="NaN or non-finite"):
        metric(actual, predicted)


@pytest.mark.parametrize("metric", [mae, rmse, mape, smape])
def test_raises_on_inf(metric) -> None:  # type: ignore[no-untyped-def]
    actual = _series([10.0, 20.0, 30.0])
    predicted = _series([10.0, np.inf, 30.0])
    with pytest.raises(ValueError, match="NaN or non-finite"):
        metric(actual, predicted)


@pytest.mark.parametrize("metric", [mae, rmse, mape, smape])
def test_raises_on_length_mismatch(metric) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="mismatched lengths"):
        metric(_series([1.0, 2.0]), _series([1.0]))


@pytest.mark.parametrize("metric", [mae, rmse, mape, smape])
def test_raises_on_empty(metric) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="empty"):
        metric(_series([]), _series([]))


def test_mape_raises_on_zero_actual() -> None:
    with pytest.raises(ValueError, match="undefined where actual is zero"):
        mape(_series([0.0, 1.0]), _series([1.0, 1.0]))


def test_smape_raises_when_both_zero() -> None:
    with pytest.raises(ValueError, match="both zero"):
        smape(_series([0.0, 1.0]), _series([0.0, 1.0]))
