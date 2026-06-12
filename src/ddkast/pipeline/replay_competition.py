"""Replay the leaderboard's scoring pipeline and verify it bit-for-bit.

The competition repo (``bartzbeielstein/challenge-leaderboard``) commits everything
its scoring used: the per-team submission CSVs, the ground truth in
``data/actual_load.parquet`` (kept in sync with scoring by its revision mechanism),
and the per-day scores in ``data/scores.parquet``. This stage re-runs the scoring
locally — same 24-hour UTC target window, same LOCF rule for missing submissions,
same metric formulas and rounding as the leaderboard's ``score_day.py`` — and
compares every replayed day against the official row. It then rebuilds the live
leaderboard aggregate (mean over target days >= the restart date) and, when the
published ``scores.json`` is supplied, checks our team's leaderboard entry too.

The metric replica deliberately mirrors the leaderboard formulas instead of
``ddkast.evaluation.metrics`` (which raises on zero actuals and has no Bias/UPR):
any deviation, including the zero-masking in MAPE, would defeat the bit-for-bit
reproduction this stage exists to prove.

Everything here is deployment context (team id, repo paths, restart date), so it
arrives as function arguments, never via ``Config``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

_console = Console()

METRICS = ("mae", "rmse", "mape", "bias", "upr")
# Tolerance for the published aggregate only: scores.json stores pandas means whose
# float summation order differs from ours; per-day values are compared exactly.
_PUBLISHED_TOLERANCE = 1e-6
_DATE_CSV_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.csv$")


@dataclass(frozen=True)
class DayScore:
    """One scored day: which CSV was graded and the five official metrics."""

    target_date: str
    source_date: str
    carried_forward: bool
    metrics: dict[str, float]


@dataclass(frozen=True)
class DayComparison:
    """A replayed day lined up against its official ``scores.parquet`` row."""

    replayed: DayScore
    official: DayScore

    @property
    def matches(self) -> bool:
        return (
            self.replayed.source_date == self.official.source_date
            and self.replayed.carried_forward == self.official.carried_forward
            and all(
                _values_equal(self.replayed.metrics[m], self.official.metrics[m])
                for m in METRICS
            )
        )


@dataclass(frozen=True)
class Aggregate:
    """Mean of the per-day metrics over one leaderboard phase."""

    n_days: int
    means: dict[str, float]


@dataclass(frozen=True)
class ReplayReport:
    team_id: str
    restart_date: str
    days: list[DayComparison]
    live: Aggregate | None
    test: Aggregate | None

    @property
    def matches(self) -> bool:
        return all(day.matches for day in self.days)


@dataclass(frozen=True)
class PublishedComparison:
    """Our live aggregate vs. the team's entry in the published ``scores.json``."""

    replayed: dict[str, float]
    published: dict[str, float]

    @property
    def matches(self) -> bool:
        if not self.published:
            return not self.replayed  # absent from both live boards is consistent
        return set(self.replayed) == set(self.published) and all(
            abs(self.replayed[k] - self.published[k]) <= _PUBLISHED_TOLERANCE
            for k in self.replayed
        )


def _values_equal(a: float, b: float) -> bool:
    """Exact equality, with NaN == NaN (the leaderboard's all-zero-load MAPE)."""
    return a == b or (math.isnan(a) and math.isnan(b))


def score_submission(forecast: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Replica of the leaderboard's ``score_day.score_submission`` — bit-for-bit.

    ``err = forecast - actual``; MAPE masks zero-load hours (NaN if every hour is
    zero); Bias is signed; UPR is the share of under-predicted hours. All values
    rounded to 4 decimals, exactly like the official scorer.
    """
    err = forecast - actual
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "mape": round(mape, 4),
        "bias": round(float(np.mean(err)), 4),
        "upr": round(float(np.mean(err < 0) * 100), 4),
    }


def _day_actual(actuals: pd.DataFrame, target_date: str) -> np.ndarray:
    """The 24 hourly ground-truth values for one UTC day — or raise (CR-3)."""
    day = actuals[actuals["timestamp_utc"].str.startswith(target_date)]
    day = day.sort_values("timestamp_utc")
    values = day["load_mw"].to_numpy(dtype=float)
    if len(values) != 24 or np.isnan(values).any():
        raise ValueError(
            f"Committed ground truth for {target_date} is incomplete "
            f"({len(values)} rows, {int(np.isnan(values).sum())} NaN); the "
            "leaderboard repo state is inconsistent with its scores."
        )
    return values


def _resolve_forecast(team_dir: Path, target_date: str) -> tuple[Path, bool]:
    """The CSV the leaderboard graded for this day: fresh, else LOCF — or raise."""
    exact = team_dir / f"{target_date}.csv"
    if exact.exists():
        return exact, False
    prior = sorted(
        p
        for p in team_dir.glob("*.csv")
        if _DATE_CSV_RE.match(p.name) and p.stem < target_date
    )
    if not prior:
        raise ValueError(
            f"No submission at or before {target_date} under {team_dir}, yet the "
            "day has an official score."
        )
    return prior[-1], True


def _aggregate(days: list[DayScore]) -> Aggregate | None:
    if not days:
        return None
    means = {
        m: float(np.mean([day.metrics[m] for day in days], dtype=float))
        for m in METRICS
    }
    return Aggregate(n_days=len(days), means=means)


def replay(leaderboard_dir: Path, team_id: str, restart_date: str) -> ReplayReport:
    """Re-score every officially scored day of ``team_id`` and compare."""
    scores = pd.read_parquet(leaderboard_dir / "data" / "scores.parquet")
    official_rows = scores[scores["team_id"] == team_id]
    if official_rows.empty:
        raise ValueError(
            f"Team '{team_id}' has no rows in data/scores.parquet — nothing to replay."
        )
    actuals = pd.read_parquet(leaderboard_dir / "data" / "actual_load.parquet")
    team_dir = leaderboard_dir / "submissions" / team_id

    comparisons: list[DayComparison] = []
    rows = sorted(official_rows.to_dict("records"), key=lambda r: str(r["target_date"]))
    for row in rows:
        target_date = str(row["target_date"])
        csv_path, carried = _resolve_forecast(team_dir, target_date)
        forecast = pd.read_csv(csv_path)["forecast_mw"].to_numpy(dtype=float)
        if len(forecast) != 24:
            raise ValueError(
                f"{csv_path.name} has {len(forecast)} rows, expected 24 — the "
                "official scorer would have skipped it, yet the day has a score."
            )
        replayed = DayScore(
            target_date=target_date,
            source_date=csv_path.stem,
            carried_forward=carried,
            metrics=score_submission(forecast, _day_actual(actuals, target_date)),
        )
        official = DayScore(
            target_date=target_date,
            source_date=str(row["source_date"]),
            carried_forward=bool(row.get("carried_forward", False)),
            metrics={m: float(row[m]) for m in METRICS},
        )
        comparisons.append(DayComparison(replayed=replayed, official=official))

    replayed_days = [c.replayed for c in comparisons]
    return ReplayReport(
        team_id=team_id,
        restart_date=restart_date,
        days=comparisons,
        live=_aggregate([d for d in replayed_days if d.target_date >= restart_date]),
        test=_aggregate([d for d in replayed_days if d.target_date < restart_date]),
    )


def compare_published(report: ReplayReport, scores_json: Path) -> PublishedComparison:
    """Line our live aggregate up against the published leaderboard entry."""
    entries = json.loads(scores_json.read_text())
    entry = next((e for e in entries if e.get("team_id") == report.team_id), None)
    published = (
        {f"mean_{m}": float(entry[f"mean_{m}"]) for m in METRICS}
        | {"n_submissions": float(entry["n_submissions"])}
        if entry is not None
        else {}
    )
    replayed = (
        {f"mean_{m}": report.live.means[m] for m in METRICS}
        | {"n_submissions": float(report.live.n_days)}
        if report.live is not None
        else {}
    )
    return PublishedComparison(replayed=replayed, published=published)


def _format_value(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.4f}"


def _day_table(report: ReplayReport) -> list[str]:
    lines = [
        "| Target day | Phase | Source | MAE | RMSE | MAPE % | Bias | UPR % "
        "| Official |",
        "|---|---|---|--:|--:|--:|--:|--:|:-:|",
    ]
    for day in report.days:
        r = day.replayed
        phase = "live" if r.target_date >= report.restart_date else "test"
        source = f"LOCF ← {r.source_date}" if r.carried_forward else "fresh"
        cells = " | ".join(_format_value(r.metrics[m]) for m in METRICS)
        verdict = "✅" if day.matches else "❌"
        lines.append(f"| {r.target_date} | {phase} | {source} | {cells} | {verdict} |")
    return lines


def _mismatch_details(report: ReplayReport) -> list[str]:
    lines: list[str] = []
    for day in report.days:
        if day.matches:
            continue
        r, o = day.replayed, day.official
        diffs = [
            f"{m}: replay {_format_value(r.metrics[m])} ≠ official "
            f"{_format_value(o.metrics[m])}"
            for m in METRICS
            if not _values_equal(r.metrics[m], o.metrics[m])
        ]
        if (r.source_date, r.carried_forward) != (o.source_date, o.carried_forward):
            diffs.append(
                f"graded CSV: replay {r.source_date} (LOCF={r.carried_forward}) ≠ "
                f"official {o.source_date} (LOCF={o.carried_forward})"
            )
        lines.append(f"- **{r.target_date}**: " + "; ".join(diffs))
    return lines


def _aggregate_section(
    report: ReplayReport, published: PublishedComparison | None
) -> list[str]:
    lines = [f"## Live leaderboard aggregate (target days ≥ {report.restart_date})"]
    if report.live is None:
        lines.append(f"No live-phase day scored yet for `{report.team_id}`.")
        return lines
    if published is None:
        cells = ", ".join(
            f"mean {m.upper()} {_format_value(report.live.means[m])}" for m in METRICS
        )
        lines.append(f"{cells} over {report.live.n_days} day(s).")
        return lines
    lines += [
        "| Metric | Replay | Published leaderboard | Match |",
        "|---|--:|--:|:-:|",
    ]
    for key in (*(f"mean_{m}" for m in METRICS), "n_submissions"):
        ours = published.replayed.get(key)
        theirs = published.published.get(key)
        ok = (
            ours is not None
            and theirs is not None
            and abs(ours - theirs) <= _PUBLISHED_TOLERANCE
        )
        lines.append(
            f"| {key} | {'—' if ours is None else f'{ours:.5f}'} "
            f"| {'—' if theirs is None else f'{theirs:.5f}'} "
            f"| {'✅' if ok else '❌'} |"
        )
    return lines


def render_markdown(report: ReplayReport, published: PublishedComparison | None) -> str:
    """The replay report as GitHub-flavoured markdown (job summary / stdout)."""
    ok = report.matches and (published is None or published.matches)
    lines = [
        f"# Competition replay — {report.team_id}",
        "",
        "Replica of the leaderboard's `score_day.py`: 24 UTC hours per target day,"
        " ground truth from the committed `data/actual_load.parquet`, LOCF for"
        " missing submissions, metrics rounded to 4 decimals.",
        "",
        "## Per-day reproduction vs `data/scores.parquet`",
        "",
        *_day_table(report),
    ]
    details = _mismatch_details(report)
    if details:
        lines += ["", "### Mismatches", "", *details]
    lines += ["", *_aggregate_section(report, published)]
    if report.test is not None:
        cells = ", ".join(
            f"mean {m.upper()} {_format_value(report.test.means[m])}" for m in METRICS
        )
        lines += [
            "",
            f"## Test phase (target days < {report.restart_date}, frozen)",
            f"{cells} over {report.test.n_days} day(s).",
        ]
    verdict = (
        "✅ The replay reproduces the official competition metrics."
        if ok
        else "❌ The replay deviates from the official competition metrics — "
        "the published page may lag the repo by one deploy; otherwise the "
        "scoring replica no longer matches the leaderboard."
    )
    lines += ["", f"**Verdict: {verdict}**", ""]
    return "\n".join(lines)


def run(
    leaderboard_dir: Path,
    team_id: str,
    restart_date: str,
    scores_json: Path | None = None,
    summary_out: Path | None = None,
) -> bool:
    """Replay, render, optionally persist the summary; ``True`` iff all matches."""
    report = replay(leaderboard_dir, team_id, restart_date)
    published = (
        compare_published(report, scores_json) if scores_json is not None else None
    )
    markdown = render_markdown(report, published)
    _console.print(markdown, soft_wrap=True)  # keep table rows unwrapped in CI logs
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        with summary_out.open("a") as fh:
            fh.write(markdown)
    return report.matches and (published is None or published.matches)
