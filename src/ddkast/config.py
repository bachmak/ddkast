from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    entsoe_api_key: str

    country_code: str = "DE_LU"
    horizon: int = 24
    resolution: str = "1H"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def load(config_path: Path = Path("config.toml")) -> Config:
    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    return Config(**data)
