from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from ddkast.data.incremental import FetchPlan, append_tail, plan_fetch


def _frame(start: str, periods: int, freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"load_mw": range(periods)}, index=idx)


# --- plan_fetch ---------------------------------------------------------------


def test_plan_full_when_nothing_stored() -> None:
    plan = plan_fetch(None, datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert plan == FetchPlan("full", datetime(2024, 1, 1), datetime(2024, 1, 3))


def test_plan_full_when_single_row() -> None:
    # One row can't reveal the sampling interval, so we can't safely append.
    existing = _frame("2024-01-01", 1)
    plan = plan_fetch(existing, datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert plan.mode == "full"


def test_plan_full_on_non_datetime_index() -> None:
    existing = pd.DataFrame({"load_mw": [1, 2, 3]})  # RangeIndex
    plan = plan_fetch(existing, datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert plan.mode == "full"


def test_plan_tail_when_end_extends() -> None:
    # Stored: 2024-01-01 00:00 .. 2024-01-02 00:00 (hourly). Request to 01-03.
    existing = _frame("2024-01-01", 25)
    plan = plan_fetch(existing, datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert plan.mode == "tail"
    # First unfetched stamp is exactly one step past the stored maximum.
    assert plan.start == datetime(2024, 1, 2, 1)
    assert plan.end == datetime(2024, 1, 3)


def test_plan_uptodate_when_end_already_covered() -> None:
    existing = _frame("2024-01-01", 25)  # ends 2024-01-02 00:00
    plan = plan_fetch(existing, datetime(2024, 1, 1), datetime(2024, 1, 2, 1))
    assert plan.mode == "uptodate"


def test_plan_full_when_start_misaligned() -> None:
    # Stored starts five hours after the requested start → unfetched head.
    existing = _frame("2024-01-01 05:00", 25)
    plan = plan_fetch(existing, datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert plan.mode == "full"


# --- append_tail --------------------------------------------------------------


def test_append_tail_seamless() -> None:
    existing = _frame("2024-01-01", 25)  # ends 2024-01-02 00:00
    tail = _frame("2024-01-02 01:00", 5)  # 01:00 .. 05:00
    result = append_tail(existing, tail)
    assert len(result) == 30
    assert result.index.is_monotonic_increasing
    assert result.index.max() == pd.Timestamp("2024-01-02 05:00", tz="UTC")


def test_append_tail_raises_on_seam_gap() -> None:
    existing = _frame("2024-01-01", 25)  # ends 2024-01-02 00:00
    tail = _frame("2024-01-02 03:00", 5)  # skips 01:00 and 02:00
    with pytest.raises(ValueError, match="Gap at incremental seam"):
        append_tail(existing, tail)


def test_append_tail_noop_when_nothing_new() -> None:
    existing = _frame("2024-01-01", 25)
    # Tail entirely within the stored range → no strictly-newer rows.
    overlap = _frame("2024-01-01 12:00", 3)
    result = append_tail(existing, overlap)
    assert result is existing


def test_append_tail_drops_overlap_then_appends() -> None:
    existing = _frame("2024-01-01", 25)  # ends 2024-01-02 00:00
    # Tail overlaps the boundary (00:00) then continues seamlessly.
    tail = _frame("2024-01-02 00:00", 4)  # 00:00 .. 03:00
    result = append_tail(existing, tail)
    assert len(result) == 28  # 25 + 3 strictly-new rows
    assert not result.index.duplicated().any()
