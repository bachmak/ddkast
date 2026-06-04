"""Forecast error metrics — MAE (primary), RMSE, MAPE, SMAPE.

CR-3 (fail-safe): every metric validates its inputs up front and raises ``ValueError``
on empty, length-mismatched, or non-finite (NaN/±inf) data rather than silently
imputing or returning a meaningless number. Load is strictly positive, so the
MAPE/SMAPE denominators are safe in production; the validation guards them explicitly
anyway so a corrupt upstream stage fails loud instead of dividing by zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate(actual: pd.Series, predicted: pd.Series) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Return finite, equal-length numpy arrays or raise ``ValueError`` (CR-3)."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.size == 0 or p.size == 0:
        raise ValueError("Metric inputs are empty.")
    if a.shape != p.shape:
        raise ValueError(
            f"Metric inputs have mismatched lengths: {a.shape} vs {p.shape}."
        )
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Metric inputs contain NaN or non-finite values.")
    return a, p


def mae(actual: pd.Series, predicted: pd.Series) -> float:  # type: ignore[type-arg]
    a, p = _validate(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:  # type: ignore[type-arg]
    a, p = _validate(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: pd.Series, predicted: pd.Series) -> float:  # type: ignore[type-arg]
    a, p = _validate(actual, predicted)
    if (a == 0).any():
        raise ValueError("MAPE is undefined where actual is zero.")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def smape(actual: pd.Series, predicted: pd.Series) -> float:  # type: ignore[type-arg]
    a, p = _validate(actual, predicted)
    denom = (np.abs(a) + np.abs(p)) / 2
    if (denom == 0).any():
        raise ValueError("SMAPE is undefined where actual and predicted are both zero.")
    return float(np.mean(np.abs(a - p) / denom) * 100)


def report(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:  # type: ignore[type-arg]
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "SMAPE": smape(actual, predicted),
    }
