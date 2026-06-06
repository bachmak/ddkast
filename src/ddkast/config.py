from __future__ import annotations

import tomllib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # OS env > .env > init (config.toml) > secrets.
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    entsoe_api_key: str

    # --- shared ---
    country_code: str = "DE_LU"
    horizon: int = 24
    resolution: str = "1h"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    # --- data source ---
    data_source: Literal["api", "fixtures"] = "api"
    fixtures_dir: Path = Path("tests/fixtures/smoke")

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

    # --- forecast origins (rolling) ---
    n_forecasts: int = 365
    forecasts_start: datetime = datetime(2025, 5, 2, 0, tzinfo=UTC)
    forecasts_end: datetime = datetime(2026, 5, 1, 0, tzinfo=UTC)

    # --- visualise ---
    backend: Literal["plotly", "matplotlib"] = "plotly"
    plots: list[str] = ["forecast", "daf", "residuals"]
    plots_dir: Path = Path("plots")
    figure_format: Literal["pdf", "png"] = "pdf"

    # --- weather ---
    weather_latitude: float = 50.110924
    weather_longitude: float = 8.682127

    # --- inter-stage filenames ---
    raw_load_actual: str = "load_actual"
    raw_load_forecast: str = "load_forecast"
    raw_weather: str = "weather_raw"
    processed_load: str = "load_clean"
    processed_weather: str = "weather_processed"
    processed_entso_forecast: str = "forecast_entso"
    predictions_subdir: str = "predictions"
    evaluation_metrics: str = "evaluation_metrics"
    evaluation_summary: str = "evaluation_summary"
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
    local_path = config_path.with_stem(config_path.stem + ".local")
    if local_path.exists():
        with open(local_path, "rb") as f:
            data.update(tomllib.load(f))
    return Config(**data)
