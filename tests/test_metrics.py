from __future__ import annotations

import numpy as np
import pandas as pd

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
