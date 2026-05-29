from __future__ import annotations

import os
from pathlib import Path

import pytest

from ddkast.config import Config, load


def test_init_kwargs_used_when_no_env(tmp_path: Path) -> None:
    cfg = Config(entsoe_api_key="key", lags=42)
    assert cfg.lags == 42


def test_env_var_overrides_init_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAGS", "999")
    cfg = Config(entsoe_api_key="key", lags=42)
    assert cfg.lags == 999


def test_env_var_overrides_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("LAGS=77\n")
    monkeypatch.setenv("LAGS", "999")
    monkeypatch.chdir(tmp_path)
    cfg = Config(entsoe_api_key="key")
    assert cfg.lags == 999


def test_dotenv_overrides_init_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("LAGS=77\n")
    monkeypatch.chdir(tmp_path)
    cfg = Config(entsoe_api_key="key", lags=42)
    assert cfg.lags == 77


def test_load_uses_toml_when_no_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text('entsoe_api_key = "toml_key"\nlags = 55\n')
    monkeypatch.chdir(tmp_path)
    cfg = load(toml)
    assert cfg.lags == 55
    assert cfg.entsoe_api_key == "toml_key"


def test_env_var_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text('entsoe_api_key = "toml_key"\nlags = 55\n')
    monkeypatch.setenv("LAGS", "999")
    monkeypatch.chdir(tmp_path)
    cfg = load(toml)
    assert cfg.lags == 999


def test_load_missing_toml_uses_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENTSOE_API_KEY", "env_key")
    monkeypatch.chdir(tmp_path)
    cfg = load(tmp_path / "nonexistent.toml")
    assert cfg.entsoe_api_key == "env_key"
    assert cfg.lags == 168
