"""Orchestration tests for the download stage's incremental logic.

The ENTSO-E / Open-Meteo fetch functions (called by the ApiDataSource) are
replaced with fakes that record their (start, end) arguments and return a
contiguous hourly frame, so no network access is needed. These tests assert the
fix for issue #15: a second run with an extended end fetches only the new tail
rather than the whole range.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from ddkast.config import Config
from ddkast.data import source
from ddkast.data.store import ParquetStore
from ddkast.pipeline import download


class FakeFetch:
    """Stand-in for a fetch function: records calls, returns [start, end) hourly."""

    def __init__(self) -> None:
        self.calls: list[tuple[datetime, datetime]] = []

    def __call__(self, *args: object, **kwargs: object) -> pd.DataFrame:
        start = kwargs.get("start", args[-2] if len(args) >= 2 else None)
        end = kwargs.get("end", args[-1] if len(args) >= 2 else None)
        assert isinstance(start, datetime) and isinstance(end, datetime)
        self.calls.append((start, end))
        idx = pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC")
        return pd.DataFrame({"value": range(len(idx))}, index=idx)


@pytest.fixture
def fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, FakeFetch]:
    registry = {
        "fetch_load": FakeFetch(),
        "fetch_load_forecast": FakeFetch(),
        "fetch_weather": FakeFetch(),
    }
    for name, fake in registry.items():
        monkeypatch.setattr(source, name, fake)
    return registry


def _cfg(config: Config, start: date, end: date) -> Config:
    return config.model_copy(update={"download_start": start, "download_end": end})


def test_first_run_fetches_full_range(
    config: Config, fakes: dict[str, FakeFetch]
) -> None:
    download.run(_cfg(config, date(2024, 1, 1), date(2024, 1, 2)))

    for fake in fakes.values():
        assert fake.calls == [(datetime(2024, 1, 1), datetime(2024, 1, 3))]

    stored = ParquetStore(config.raw_dir).read(config.raw_load_actual)
    assert len(stored) == 48  # 2 full days, hourly


def test_second_run_fetches_only_the_tail(
    config: Config, fakes: dict[str, FakeFetch]
) -> None:
    download.run(_cfg(config, date(2024, 1, 1), date(2024, 1, 2)))
    download.run(_cfg(config, date(2024, 1, 1), date(2024, 1, 3)))

    # The second call starts at the first unstored timestamp, not the range start.
    load_fake = fakes["fetch_load"]
    assert load_fake.calls[1] == (datetime(2024, 1, 3), datetime(2024, 1, 4))

    stored = ParquetStore(config.raw_dir).read(config.raw_load_actual)
    assert len(stored) == 72  # 48 appended with 24 new
    assert stored.index.is_monotonic_increasing
    assert not stored.index.duplicated().any()


def test_third_run_is_uptodate_and_skips_fetch(
    config: Config, fakes: dict[str, FakeFetch]
) -> None:
    download.run(_cfg(config, date(2024, 1, 1), date(2024, 1, 3)))
    download.run(_cfg(config, date(2024, 1, 1), date(2024, 1, 3)))

    # End is already covered → the second run makes no fetch call at all.
    for fake in fakes.values():
        assert len(fake.calls) == 1


def test_full_flag_refetches_everything(
    config: Config, fakes: dict[str, FakeFetch]
) -> None:
    download.run(_cfg(config, date(2024, 1, 1), date(2024, 1, 2)))
    download.run(_cfg(config, date(2024, 1, 1), date(2024, 1, 3)), full=True)

    # --full ignores stored data: the second call covers the whole range.
    load_fake = fakes["fetch_load"]
    assert load_fake.calls[1] == (datetime(2024, 1, 1), datetime(2024, 1, 4))
