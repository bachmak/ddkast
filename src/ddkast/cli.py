from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

import ddkast.pipeline.download as _download
import ddkast.pipeline.evaluate as _evaluate
import ddkast.pipeline.format_submission as _format_submission
import ddkast.pipeline.merge as _merge
import ddkast.pipeline.predict as _predict
import ddkast.pipeline.replay_competition as _replay_competition
import ddkast.pipeline.train as _train
import ddkast.pipeline.visualise as _visualise
from ddkast.config import load

app = typer.Typer(help="ddkast — energy consumption forecasting")

_ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.toml")]


@app.command()
def download(
    config: _ConfigOpt = Path("config.toml"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Refetch even if a cached download matches the config.",
        ),
    ] = False,
) -> None:
    """Fetch raw load data from ENTSO-E."""
    _download.run(load(config), force=force)


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


@app.command(name="format-submission")
def format_submission(
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Directory to write the {YYYY-MM-DD}.csv submission file into.",
        ),
    ],
    config: _ConfigOpt = Path("config.toml"),
) -> None:
    """Write tomorrow's hourly forecast to the leaderboard submission schema."""
    _format_submission.run(load(config), out_dir)


@app.command(name="replay-competition")
def replay_competition(
    leaderboard_dir: Annotated[
        Path,
        typer.Option(
            "--leaderboard-dir",
            help="Checkout of the challenge-leaderboard repo (submissions + data).",
        ),
    ],
    team_id: Annotated[
        str, typer.Option("--team-id", help="Our team id in the leaderboard repo.")
    ],
    restart_date: Annotated[
        str,
        typer.Option(
            "--restart-date",
            help="First live-phase target day (the leaderboard's RESTART_DATE).",
        ),
    ] = "2026-06-10",
    scores_json: Annotated[
        Path | None,
        typer.Option(
            "--scores-json",
            help="Published data/scores.json to verify the live aggregate against.",
        ),
    ] = None,
    summary_out: Annotated[
        Path | None,
        typer.Option(
            "--summary-out",
            help="File to append the markdown report to (e.g. $GITHUB_STEP_SUMMARY).",
        ),
    ] = None,
) -> None:
    """Replay the leaderboard scoring over our submissions and verify the metrics."""
    ok = _replay_competition.run(
        leaderboard_dir, team_id, restart_date, scores_json, summary_out
    )
    if not ok:
        raise typer.Exit(1)


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
