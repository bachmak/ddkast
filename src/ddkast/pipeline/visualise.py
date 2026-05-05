from __future__ import annotations

from datetime import datetime

import pandas as pd
from rich.console import Console

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.visualisation.protocol import PlotBackend, VisualisationData

_console = Console()


def _clip(
    s: pd.Series[float],
    ts_from: pd.Timestamp | None,
    ts_to: pd.Timestamp | None,
) -> pd.Series[float]:
    if ts_from is not None:
        s = s.loc[s.index >= ts_from]  # type: ignore[operator]
    if ts_to is not None:
        s = s.loc[s.index <= ts_to]  # type: ignore[operator]
    return s


def run(
    config: Config,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> None:
    """Load evaluation series and render visualisation using the configured backend."""
    processed = ParquetStore(config.processed_dir)
    series = processed.read(config.evaluation_series)

    ts_from = pd.Timestamp(date_from) if date_from is not None else None
    ts_to = pd.Timestamp(date_to) if date_to is not None else None

    actual: pd.Series[float] = _clip(series["actual"], ts_from, ts_to)
    forecast: pd.Series[float] = _clip(series["forecast"], ts_from, ts_to)
    entso_daf: pd.Series[float] = _clip(series["entso_daf"], ts_from, ts_to)
    residuals_forecast: pd.Series[float] = _clip(
        series["residuals_forecast"], ts_from, ts_to
    )
    residuals_daf: pd.Series[float] | None = (
        _clip(series["residuals_daf"], ts_from, ts_to)
        if "residuals_daf" in series.columns
        else None
    )

    data = VisualisationData(
        actual=actual,
        forecast=forecast,
        entso_daf=entso_daf,
        residuals_forecast=residuals_forecast,
        residuals_daf=residuals_daf,
    )

    backend = _load_backend(config)
    uris = backend.render(data, config)

    _console.print(
        f"[bold]visualise[/bold]  [green]{config.backend}[/green] "
        f"→ {config.plots_dir}"
    )
    for uri in uris:
        _console.print(f"  [link={uri}]{uri}[/link]")


def _load_backend(config: Config) -> PlotBackend:
    if config.backend == "plotly":
        from ddkast.visualisation.plotly_backend import PlotlyBackend

        return PlotlyBackend()
    if config.backend == "matplotlib":
        from ddkast.visualisation.matplotlib_backend import MatplotlibBackend

        return MatplotlibBackend()
    raise ValueError(
        f"Unknown backend {config.backend!r}. Choose 'plotly' or 'matplotlib'."
    )
