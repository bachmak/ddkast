"""Rolling-origin folds: the single source of truth for forecast origins."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ddkast.config import Config


@dataclass(frozen=True)
class Fold:
    """One rolling origin (last timestamp the model may see) and its forecast block."""

    fold_id: str
    origin: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp


def build_folds(index: pd.DatetimeIndex, config: Config) -> list[Fold]:
    """Build the deterministic rolling-origin fold set from training history ``index``.

    Raises ValueError (CR-3) if the index is empty/unsorted, the forecast-window knobs
    are inconsistent, the window can't be evenly spaced on the resolution grid, or the
    earliest origin lacks ``lags`` hours of history.
    """
    step = _resolution_step(config)
    _validate_index(index)
    _validate_knobs(config)

    origins = _origins(config, step)
    _check_sufficient_history(index, origins[0], config)

    return [_make_fold(origin, config, step) for origin in origins]


def _resolution_step(config: Config) -> pd.Timedelta:
    """The sampling step between consecutive timestamps, from ``config.resolution``."""
    step = pd.Timedelta(config.resolution)
    if step <= pd.Timedelta(0):
        raise ValueError(
            f"resolution must be a positive duration, got {config.resolution!r}"
        )
    return step


def _validate_index(index: pd.DatetimeIndex) -> None:
    """Raise if the load index is empty or not sorted ascending (CR-3)."""
    if len(index) == 0:
        raise ValueError("Cannot build folds from an empty index.")
    if not index.is_monotonic_increasing:
        raise ValueError("Fold index must be sorted ascending.")


def _validate_knobs(config: Config) -> None:
    """Raise if the forecast-window knobs are out of range (CR-3)."""
    if config.n_forecasts < 1:
        raise ValueError(f"n_forecasts must be >= 1, got {config.n_forecasts}")

    start, end = _as_utc(config.forecasts_start), _as_utc(config.forecasts_end)
    if config.n_forecasts == 1:
        if start != end:
            raise ValueError(
                f"n_forecasts=1 needs forecasts_start == forecasts_end, "
                f"got {start} != {end}"
            )
    elif start >= end:
        raise ValueError(
            f"forecasts_start must be < forecasts_end when n_forecasts > 1, "
            f"got {start} >= {end}"
        )


def _origins(config: Config, step: pd.Timedelta) -> list[pd.Timestamp]:
    """The ``n_forecasts`` evenly spaced origins, ``forecasts_start`` to
    ``forecasts_end``."""
    start = _as_utc(config.forecasts_start)
    if config.n_forecasts == 1:
        return [start]
    stride = _stride(start, _as_utc(config.forecasts_end), config.n_forecasts, step)
    return [start + k * stride for k in range(config.n_forecasts)]


def _stride(
    start: pd.Timestamp, end: pd.Timestamp, n_forecasts: int, step: pd.Timedelta
) -> pd.Timedelta:
    """Derived stride between origins, integer-ns on the resolution grid (CR-2/CR-3)."""
    intervals = n_forecasts - 1
    span_ns = (end - start).value
    if span_ns % intervals != 0:
        raise ValueError(
            f"Cannot place {n_forecasts} evenly spaced origins over {start} → {end}: "
            f"the span is not divisible into {intervals} equal steps."
        )
    stride = pd.Timedelta(span_ns // intervals, unit="ns")
    if stride.value % step.value != 0:
        raise ValueError(
            f"Derived origin stride {stride} is not a whole multiple of the {step} "
            "resolution; origins would fall off the data grid."
        )
    return stride


def _check_sufficient_history(
    index: pd.DatetimeIndex, earliest_origin: pd.Timestamp, config: Config
) -> None:
    """Raise if the earliest fold lacks ``lags`` hours of training history (CR-3)."""
    first_ts: pd.Timestamp = index[0]
    if earliest_origin - first_ts < pd.Timedelta(hours=config.lags):
        raise ValueError(
            f"Insufficient history: earliest origin {earliest_origin} leaves "
            f"< {config.lags}h of training data after {first_ts}. "
            "Move forecasts_start later or extend the data window."
        )


def _make_fold(origin: pd.Timestamp, config: Config, step: pd.Timedelta) -> Fold:
    """Build the fold anchored at ``origin``: its id and its ``horizon``-hour block."""
    return Fold(
        fold_id=_fold_id(origin),
        origin=origin,
        forecast_start=origin + step,
        forecast_end=origin + pd.Timedelta(hours=config.horizon),
    )


def _fold_id(origin: pd.Timestamp) -> str:
    """Deterministic, filesystem-safe id from the origin timestamp (UTC)."""
    return origin.strftime("%Y%m%dT%H%M%SZ")


def _as_utc(value: object) -> pd.Timestamp:
    """Coerce a config datetime to a UTC ``Timestamp`` (naive is read as UTC)."""
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
