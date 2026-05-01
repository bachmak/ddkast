from __future__ import annotations

from datetime import datetime

import pandas as pd
from entsoe.entsoe import EntsoePandasClient


def fetch_load(
    api_key: str,
    country_code: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch actual total load from ENTSO-E for the given country and time range."""
    client = EntsoePandasClient(api_key=api_key)
    raw = client.query_load(
        country_code,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
    )
    return raw.rename(columns={"Actual Load": "load_mw"})
