from __future__ import annotations

import pandas as pd

from ddkast.models.baseline import predict


def test_predict_returns_correct_length(load_series: pd.Series) -> None:  # type: ignore[type-arg]
    result = predict(load_series, horizon=24)
    assert len(result) == 24


def test_predict_uses_week_ago_values(load_series: pd.Series) -> None:  # type: ignore[type-arg]
    result = predict(load_series, horizon=24)
    expected = load_series.iloc[-24 - 24 * 7 : -24 * 7]
    assert list(result.values) == list(expected.values)
