from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    entsoe_api_key: str

    # --- shared ---
    country_code: str = "DE_LU"
    horizon: int = 24
    resolution: str = "1h"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    # --- download ---
    download_start: date = date(2022, 1, 1)
    download_end: date = date(2026, 4, 30)

    # --- merge / cleaning ---
    outlier_iqr_multiplier: float = 3.0
    max_interpolation_hours: int = 3

    # --- features ---
    # ISO 2-letter country code for holiday calendar (DE_LU zone → Germany holidays)
    holiday_country_code: str = "DE"
    rbf_periods_hour: int = 10
    rbf_periods_dow: int = 7
    rbf_periods_month: int = 6

    # --- train ---
    lags: int = 168
    test_days: int = 30

    # --- visualise ---
    backend: Literal["plotly", "matplotlib"] = "plotly"
    plots: list[str] = ["forecast", "daf", "residuals"]
    plots_dir: Path = Path("plots")
    figure_format: Literal["pdf", "png"] = "pdf"

    # --- report (email) ---
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: SecretStr = SecretStr("")
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: str = ""

    # --- inter-stage filenames ---
    raw_load_actual: str = "load_actual"
    raw_load_forecast: str = "load_forecast"
    processed_load: str = "load_clean"
    processed_entso_forecast: str = "forecast_entso"
    processed_test: str = "load_test"
    processed_predictions: str = "load_predicted"
    evaluation_series: str = "evaluation_series"
    model_target: str = "load_mw"

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
