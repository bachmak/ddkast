from __future__ import annotations

from datetime import datetime
from math import ceil

import pandas as pd
from spotforecast2_safe.weather import WeatherClient

# Full weather schema Open-Meteo serves and the model can consume as exog.
WEATHER_COLS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]


def fetch_weather(
    start: datetime,
    end: datetime,
    latitude: float,
    longitude: float,
    horizon: int,
) -> pd.DataFrame:
    """Fetch a contiguous hourly weather series for the model exog matrix.

    Covers [start, min(end, now)] from the reanalysis archive and, whenever the
    requested window reaches the present, the next ``horizon`` hours from the
    forecast endpoint so the predict stage has future exog.

    Composes two *public* spotforecast2-safe endpoints rather than
    WeatherService.get_dataframe: the latter's hybrid fetch caps the archive at
    now-5d and never passes ``past_days``, leaving a multi-day gap between the
    archive tail and the forecast window. Here the archive is queried right up
    to ``now`` (Open-Meteo serves it near-real-time) and the forecast frame wins
    the one-day overlap, so a not-yet-final archive tail can't shadow a valid
    forecast value.

    The result is validated to be gap-free and NaN-free (CR-3 fail-safe): a real
    archive lag or hole raises loudly here rather than surfacing downstream as
    NaN in the exog matrix.
    """
    client = WeatherClient(latitude=latitude, longitude=longitude)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")

    # The archive only exists in the past; cap its end at now and let the
    # forecast endpoint cover anything from today onward.
    frames: list[pd.DataFrame] = []
    if end_ts > now:
        # forecast_days counts whole days from today 00:00; the +1 buffers the
        # current partial day so now+horizon is always inside the window.
        forecast_days = ceil(horizon / 24) + 1
        frames.append(client.fetch_forecast(forecast_days, timezone="UTC"))
    frames.append(client.fetch_archive(start_ts, min(end_ts, now), timezone="UTC"))

    combined = pd.concat(frames)
    # frames[0] is the forecast (when present), so keep="first" lets it win the
    # archive/forecast overlap on today's hours.
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    idx = combined.index
    assert isinstance(idx, pd.DatetimeIndex)
    if idx.tz is None:
        combined.index = idx.tz_localize("UTC")

    missing = set(WEATHER_COLS) - set(combined.columns)
    if missing:
        raise ValueError(
            f"Open-Meteo response is missing expected columns: {sorted(missing)}"
        )
    result = combined[WEATHER_COLS]
    _assert_contiguous_and_complete(result)
    return result


def _assert_contiguous_and_complete(df: pd.DataFrame) -> None:
    """Reject any missing hour or NaN in the weather frame (CR-3 fail-safe)."""
    idx = df.index
    assert isinstance(idx, pd.DatetimeIndex)
    full = pd.date_range(idx.min(), idx.max(), freq="h", tz="UTC")
    gaps = full.difference(idx)
    if len(gaps) > 0:
        preview = ", ".join(str(ts) for ts in gaps[:5])
        more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
        raise ValueError(
            f"Weather series has {len(gaps)} missing hour(s) between "
            f"{idx.min()} and {idx.max()}. First gaps: [{preview}]{more}"
        )
    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        raise ValueError(f"Weather series has {nan_count} NaN value(s) after fetch")
