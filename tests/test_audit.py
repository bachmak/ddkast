from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddkast import audit
from ddkast.config import Config


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(entsoe_api_key="secret", data_dir=tmp_path / "data")


def _read_lines(cfg: Config) -> list[dict[str, object]]:
    path = cfg.audit_dir / audit.AUDIT_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_record_creates_jsonl_with_schema(cfg: Config) -> None:
    rec = audit.record_run(
        cfg,
        stage="evaluate",
        status="ok",
        metrics={"model_MAE_mean": 123.0},
        extra={"n_folds": 3, "seed": 42},
    )
    lines = _read_lines(cfg)
    assert len(lines) == 1
    stored = lines[0]
    assert stored == rec
    for key in (
        "timestamp",
        "stage",
        "status",
        "git_sha",
        "config_hash",
        "packages",
        "data_window",
        "metrics",
        "extra",
    ):
        assert key in stored
    assert stored["stage"] == "evaluate"
    assert stored["metrics"] == {"model_MAE_mean": 123.0}
    assert stored["extra"]["n_folds"] == 3
    assert stored["data_window"] == {
        "start": cfg.download_start.isoformat(),
        "end": cfg.download_end.isoformat(),
    }


def test_record_appends(cfg: Config) -> None:
    audit.record_run(cfg, stage="train", status="ok", metrics={})
    audit.record_run(cfg, stage="evaluate", status="ok", metrics={})
    lines = _read_lines(cfg)
    assert [line["stage"] for line in lines] == ["train", "evaluate"]


def test_config_hash_is_stable_and_sensitive(cfg: Config) -> None:
    audit.record_run(cfg, stage="train", status="ok", metrics={})
    other = cfg.model_copy(update={"lags": cfg.lags + 1})
    audit.record_run(other, stage="train", status="ok", metrics={})
    h1, h2 = (line["config_hash"] for line in _read_lines(cfg))
    assert len(h1) == 64  # sha256 hex
    assert h1 != h2  # different config → different hash


def test_git_absent_falls_back_to_unknown(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("git not found")

    monkeypatch.setattr(audit.subprocess, "run", _boom)
    rec = audit.record_run(cfg, stage="train", status="ok", metrics={})
    assert rec["git_sha"] == "unknown"


def test_packages_include_tracked(cfg: Config) -> None:
    rec = audit.record_run(cfg, stage="train", status="ok", metrics={})
    packages = rec["packages"]
    assert isinstance(packages, dict)
    assert set(packages) == {"spotforecast2-safe", "lightgbm", "ddkast"}


def test_missing_package_falls_back_to_unknown(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _missing(_name: str) -> str:
        raise audit.metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(audit.metadata, "version", _missing)
    rec = audit.record_run(cfg, stage="train", status="ok", metrics={})
    assert rec["packages"] == {
        "spotforecast2-safe": "unknown",
        "lightgbm": "unknown",
        "ddkast": "unknown",
    }
