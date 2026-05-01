from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: pd.Series, predicted: pd.Series) -> float:  # type: ignore[type-arg]
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:  # type: ignore[type-arg]
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: pd.Series, predicted: pd.Series) -> float:  # type: ignore[type-arg]
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def smape(actual: pd.Series, predicted: pd.Series) -> float:  # type: ignore[type-arg]
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    return float(np.mean(np.abs(actual - predicted) / denom) * 100)


def report(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:  # type: ignore[type-arg]
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "SMAPE": smape(actual, predicted),
    }
