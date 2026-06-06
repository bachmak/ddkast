"""Merge stage: clean raw load and align DAF + weather onto the load grid.

``download`` writes three raw series straight from their providers:

* **load** — ENTSO-E actual load, 15-min Europe/Berlin grid
* **DAF** — ENTSO-E day-ahead forecast, 15-min Europe/Berlin grid
* **weather** — Open-Meteo, hourly grid (may be tz-naive)

``merge`` is where they become a single analysis-ready, hourly-UTC dataset.
Load is cleaned and resampled to the hourly grid; DAF and weather are
tz-normalised to UTC. DAF is reindexed onto the load grid (evaluate lines the
benchmark up against the actuals). Weather is exog on the model path, so it is
only lower-bound trimmed to the load start — its forecast hours past the load
tail are kept so predict has future exog (#23).

Failure policy (issue #10): load is the spine, so a NaN left on the cleaned
load grid is a hard error. DAF and weather quality is independent of load, so
their gaps are *soft* — interpolate up to ``max_interpolation_hours`` and drop
whatever survives as NaN rather than aborting the run.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.preprocessing.clean import clean

_console = Console()


def run(config: Config) -> None:
    """Clean load, then align the DAF and weather onto its hourly-UTC grid."""
    raw = ParquetStore(config.raw_dir)
    processed = ParquetStore(config.processed_dir)

    load = _clean_load_spine(raw, config)
    daf = _align_daf(raw, load.index, config)
    weather = _align_weather(raw, load.index, config)

    _persist(processed, config.processed_load, load, "clean load", config)
    _persist(processed, config.processed_entso_forecast, daf, "DAF", config)
    _persist(processed, config.processed_weather, weather, "weather", config)


def _clean_load_spine(raw: ParquetStore, config: Config) -> pd.DataFrame:
    """Clean actual load into the hourly-UTC spine every other series aligns to.

    Load is the backbone of the dataset, so a NaN surviving cleaning is a hard
    error (see :func:`_require_gap_free`) — unlike the soft DAF/weather gaps.
    """
    _console.print("[bold]merge[/bold] cleaning actual load…")
    load = clean(raw.read(config.raw_load_actual), config)
    _require_gap_free(load)
    return load


def _align_daf(raw: ParquetStore, load_index: pd.Index, config: Config) -> pd.DataFrame:
    """Resample DAF to hourly UTC and align it onto the load grid (soft gaps).

    Reindexing onto ``load_index`` is what lets evaluate.py line the published
    benchmark up against the forecast. Interior gaps are linearly interpolated up
    to ``max_interpolation_hours``; anything still NaN (unfillable holes, or load
    timestamps the DAF never covered) is dropped rather than raised on.
    """
    _console.print("  aligning ENTSO-E day-ahead forecast onto the load grid…")
    daf = _to_hourly_utc(raw.read(config.raw_load_forecast), config)
    daf = daf.reindex(load_index)
    daf = daf.interpolate(
        method="linear",
        limit=config.max_interpolation_hours,
        limit_area="inside",
    )
    daf = daf.dropna()
    _warn_dropped(len(load_index) - len(daf), "DAF")
    return daf


def _align_weather(
    raw: ParquetStore, load_index: pd.Index, config: Config
) -> pd.DataFrame:
    """Resample weather to hourly UTC, drop the pre-load archive, keep the tail.

    Unlike DAF — which is reindexed onto the load grid so evaluate can line the
    benchmark up against the actuals — weather is exog on the *model* path. It is
    needed both over the training range (where it overlaps load) and over the
    future forecast window, which by definition sits beyond the load tail for
    the latest origin predict forecasts. So weather is only lower-bound trimmed:
    the over-long archive before the load start is dropped, but the forecast
    hours past ``load.max`` are preserved so predict has future exog (#23). Their
    upper bound is already capped at ~now+horizon upstream by ``fetch_weather``.

    Trimming stays soft (Open-Meteo's publication lag rarely lines up with load
    exactly): drop any residual NaN inside the kept range rather than aborting.
    """
    _console.print("  aligning weather onto the load grid…")
    weather = _to_hourly_utc(raw.read(config.raw_weather), config)
    in_range = weather.index >= load_index.min()
    trimmed = cast(pd.DataFrame, weather.loc[in_range])
    result = trimmed.dropna()
    _warn_dropped(len(trimmed) - len(result), "weather")
    return result


def _to_hourly_utc(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Coerce *df* to a sorted hourly-UTC frame, deduplicating timestamps.

    ``pd.to_datetime(..., utc=True)`` both localises tz-naive indices (Open-Meteo
    can return these) and converts tz-aware ones (ENTSO-E's Europe/Berlin) to UTC,
    so a single path normalises every provider before resampling.
    """
    out = df.copy()
    out.index = pd.to_datetime(df.index, utc=True)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.resample(config.resolution).mean()


def _require_gap_free(load: pd.DataFrame) -> None:
    """Hard-fail if cleaned load still carries NaN: the spine must be gap-free."""
    if load.isna().to_numpy().any():
        raise ValueError(
            "cleaned load still contains NaN on the hourly-UTC grid; load is the "
            "spine of the dataset and must be gap-free before DAF and weather are "
            "aligned to it"
        )


def _warn_dropped(dropped: int, label: str) -> None:
    """Report rows lost off the load grid or beyond the interpolation limit."""
    if dropped > 0:
        _console.print(
            f"  [yellow]⚠[/yellow] dropped {dropped:,} {label} hour(s) off the load "
            "grid or beyond the interpolation limit"
        )


def _persist(
    store: ParquetStore, name: str, frame: pd.DataFrame, label: str, config: Config
) -> None:
    """Write *frame* under *name* and report its row count and destination."""
    store.write(name, frame)
    _console.print(
        f"  [green]✓[/green] {len(frame):,} {label} rows → "
        f"{config.processed_dir / name}.parquet"
    )
