from __future__ import annotations

from datetime import datetime

import pandas as pd
from entsoe.entsoe import EntsoePandasClient


def _client(api_key: str) -> EntsoePandasClient:
    return EntsoePandasClient(api_key=api_key)


def fetch_load(
    api_key: str,
    country_code: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch actual total load from ENTSO-E for the given country and time range."""
    raw = _client(api_key).query_load(
        country_code,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
    )
    return raw.rename(columns={"Actual Load": "load_mw"})


def fetch_load_forecast(
    api_key: str,
    country_code: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch ENTSO-E day-ahead load forecast for the given country and time range."""
    raw = _client(api_key).query_load_forecast(
        country_code,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
    )
    return raw.rename(columns={"Forecasted Load": "forecast_mw"})
