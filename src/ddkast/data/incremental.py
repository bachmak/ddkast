"""Incremental-fetch planning for the download stage.

Pure, side-effect-free helpers that decide *what* range to fetch given what is
already on disk, and stitch a freshly-fetched tail onto the stored data. The
download stage owns all I/O; this module only reasons about timestamps.

Strategy (see issue #15):
- No usable stored data, or a stored range that does not line up with the
  requested start → fetch the full range and overwrite ("full").
- Stored data whose start matches the request and whose end is behind the
  requested end → fetch only the strictly-newer tail ``(stored_max, end]``
  and append it ("tail").
- Stored data that already reaches the requested end → fetch nothing
  ("uptodate").

The tail append is *strict*: it never re-fetches or de-duplicates an overlap to
chase upstream revisions. As a fail-safe it raises if appending the tail would
leave a gap at the seam (the first new timestamp is not exactly one step after
the stored maximum). Gaps *within* a contiguous fetch are left for the cleaning
stage, matching full-download behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

FetchMode = Literal["full", "tail", "uptodate"]


@dataclass(frozen=True)
class FetchPlan:
    """How to satisfy a requested ``[start, end)`` range given stored data.

    ``start`` / ``end`` are naive UTC datetimes ready to pass to a fetch
    function. For ``uptodate`` they echo the request and should not be used.
    """

    mode: FetchMode
    start: datetime
    end: datetime


def _utc_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Drop tz information, normalising to UTC first so instants are preserved."""
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def _step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Native sampling interval: the smallest gap between consecutive stamps."""
    diffs = index[1:] - index[:-1]
    return diffs.min()


def plan_fetch(
    existing: pd.DataFrame | None, start: datetime, end: datetime
) -> FetchPlan:
    """Decide how to fetch ``[start, end)`` given already-stored ``existing``."""
    if existing is None or len(existing) < 2:
        return FetchPlan("full", start, end)

    index = existing.index
    if not isinstance(index, pd.DatetimeIndex):
        return FetchPlan("full", start, end)

    utc = _utc_naive(index)
    step = _step(utc)
    stored_min = utc.min()
    stored_max = utc.max()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Stored range must line up with the requested start; otherwise we would
    # either leave an unfetched head or carry stale extra history → full refetch.
    if abs(stored_min - start_ts) > step:
        return FetchPlan("full", start, end)

    expected_next = stored_max + step
    if expected_next >= end_ts:
        return FetchPlan("uptodate", start, end)

    return FetchPlan("tail", expected_next.to_pydatetime(), end)


def append_tail(existing: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    """Append the strictly-newer rows of ``fetched`` to ``existing``.

    Returns ``existing`` unchanged when the fetch yielded nothing new (e.g. the
    upstream frontier has not advanced). Raises ``ValueError`` if the first new
    timestamp is not exactly one step after the stored maximum.
    """
    existing_index = existing.index
    fetched_index = fetched.index
    if not isinstance(existing_index, pd.DatetimeIndex) or not isinstance(
        fetched_index, pd.DatetimeIndex
    ):
        raise TypeError("append_tail requires DatetimeIndex-ed frames")

    stored = _utc_naive(existing_index)
    step = _step(stored)
    stored_max = stored.max()

    fetched_naive = _utc_naive(fetched_index)
    is_new = fetched_naive > stored_max
    tail = fetched[is_new]
    if tail.empty:
        return existing

    tail_min = fetched_naive[is_new].min()
    expected = stored_max + step
    if tail_min != expected:
        raise ValueError(
            f"Gap at incremental seam: stored data ends at {stored_max} (UTC); "
            f"next fetched timestamp is {tail_min} (UTC), expected {expected}. "
            f"Re-run download with --full to refetch the whole range."
        )

    return pd.concat([existing, tail]).sort_index()
