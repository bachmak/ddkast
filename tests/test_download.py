"""Tests for the download stage and its config-keyed cache."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

import ddkast.pipeline.download as download
from ddkast.config import Config


class _SpySource:
    """Stand-in DataSource that counts fetches and returns tiny deterministic frames."""

    def __init__(self) -> None:
        self.calls = 0

    def _frame(self, col: str) -> pd.DataFrame:
        self.calls += 1
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        return pd.DataFrame({col: [1.0, 2.0, 3.0]}, index=idx)

    def load_actual(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self._frame("load_mw")

    def load_forecast(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self._frame("forecast_mw")

    def weather(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self._frame("temperature")


def _spy(monkeypatch) -> _SpySource:  # type: ignore[no-untyped-def]
    spy = _SpySource()
    monkeypatch.setattr(download, "make_data_source", lambda config: spy)
    return spy


def test_writes_outputs_and_fingerprint(config: Config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spy = _spy(monkeypatch)
    download.run(config)

    assert spy.calls == 3
    for name in (config.raw_load_actual, config.raw_load_forecast, config.raw_weather):
        assert (config.raw_dir / f"{name}.parquet").exists()
    fingerprint = config.raw_dir / download._FINGERPRINT_NAME
    assert json.loads(fingerprint.read_text()) == download._cache_key(config)


def test_cache_hit_skips_fetch(config: Config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spy = _spy(monkeypatch)
    download.run(config)
    download.run(config)

    assert spy.calls == 3  # second run served from cache


def test_force_refetches(config: Config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spy = _spy(monkeypatch)
    download.run(config)
    download.run(config, force=True)

    assert spy.calls == 6


def test_changed_config_busts_cache(config: Config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spy = _spy(monkeypatch)
    download.run(config)
    download.run(config.model_copy(update={"country_code": "FR"}))

    assert spy.calls == 6


def test_missing_output_busts_cache(config: Config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spy = _spy(monkeypatch)
    download.run(config)
    (config.raw_dir / f"{config.raw_load_actual}.parquet").unlink()
    download.run(config)

    assert spy.calls == 6


def test_missing_fingerprint_busts_cache(config: Config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spy = _spy(monkeypatch)
    download.run(config)
    (config.raw_dir / download._FINGERPRINT_NAME).unlink()
    download.run(config)

    assert spy.calls == 6
