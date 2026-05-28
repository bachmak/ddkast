from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

import ddkast.pipeline.download as _download
import ddkast.pipeline.evaluate as _evaluate
import ddkast.pipeline.merge as _merge
import ddkast.pipeline.predict as _predict
import ddkast.pipeline.report as _report
import ddkast.pipeline.train as _train
import ddkast.pipeline.visualise as _visualise
from ddkast.config import load

app = typer.Typer(help="ddkast — energy consumption forecasting")

_ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.toml")]


@app.command()
def download(config: _ConfigOpt = Path("config.toml")) -> None:
    """Fetch raw load data from ENTSO-E."""
    _download.run(load(config))


@app.command()
def merge(config: _ConfigOpt = Path("config.toml")) -> None:
    """Merge and clean raw data into a single processed dataset."""
    _merge.run(load(config))


@app.command()
def train(config: _ConfigOpt = Path("config.toml")) -> None:
    """Train the forecasting model."""
    _train.run(load(config))


@app.command()
def predict(config: _ConfigOpt = Path("config.toml")) -> None:
    """Generate a forecast for the next horizon hours."""
    _predict.run(load(config))


@app.command()
def evaluate(config: _ConfigOpt = Path("config.toml")) -> None:
    """Evaluate forecast accuracy against the baseline and ground truth."""
    _evaluate.run(load(config))


@app.command()
def report(config: _ConfigOpt = Path("config.toml")) -> None:
    """Email tomorrow's hourly forecast (UTC) to the configured recipient."""
    _report.run(load(config))


@app.command()
def visualise(
    config: _ConfigOpt = Path("config.toml"),
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Rendering backend: plotly (interactive) or matplotlib (static).",
        ),
    ] = None,
    date_from: Annotated[
        datetime | None,
        typer.Option("--from", help="Start of the visualisation window (ISO 8601)."),
    ] = None,
    date_to: Annotated[
        datetime | None,
        typer.Option("--to", help="End of the visualisation window (ISO 8601)."),
    ] = None,
    plots: Annotated[
        list[str] | None,
        typer.Option(
            "--plots",
            help="Plots to include (pass multiple times): forecast, daf, residuals.",
        ),
    ] = None,
) -> None:
    """Visualise forecast results using the configured or specified backend."""
    cfg = load(config)
    if backend is not None:
        cfg = cfg.model_copy(update={"backend": backend})
    if plots is not None:
        cfg = cfg.model_copy(update={"plots": plots})
    _visualise.run(cfg, date_from=date_from, date_to=date_to)
