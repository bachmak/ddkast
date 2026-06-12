"""Tests for the ``ddkast replay-competition`` stage.

Each test builds a miniature challenge-leaderboard checkout under ``tmp_path``:
committed ground truth (``data/actual_load.parquet``), submission CSVs, and the
official per-day scores (``data/scores.parquet``) with hand-computed metric values.
The replay must reproduce those values bit-for-bit; the tampering tests prove that
any deviation — metric, graded CSV, or published aggregate — is detected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddkast.pipeline import replay_competition as rc

TEAM = "ddkast"
RESTART = "2026-06-10"
# Flat actual load of 1000 MW; +10 MW over-forecast on the fresh test-phase day,
# -10 MW under-forecast on the fresh live day (carried into the LOCF day).
DAYS = ("2026-06-08", "2026-06-10", "2026-06-11")
OFFICIAL_ROWS = [
    # 1010 vs 1000 → MAE 10, RMSE 10, MAPE 1 %, Bias +10, UPR 0 %.
    {
        "team_id": TEAM,
        "target_date": "2026-06-08",
        "source_date": "2026-06-08",
        "carried_forward": False,
        "mae": 10.0,
        "rmse": 10.0,
        "mape": 1.0,
        "bias": 10.0,
        "upr": 0.0,
    },
    # 990 vs 1000 → Bias flips to −10, UPR to 100 %.
    {
        "team_id": TEAM,
        "target_date": "2026-06-10",
        "source_date": "2026-06-10",
        "carried_forward": False,
        "mae": 10.0,
        "rmse": 10.0,
        "mape": 1.0,
        "bias": -10.0,
        "upr": 100.0,
    },
    # No CSV for the 11th → LOCF from the 10th, same forecast values.
    {
        "team_id": TEAM,
        "target_date": "2026-06-11",
        "source_date": "2026-06-10",
        "carried_forward": True,
        "mae": 10.0,
        "rmse": 10.0,
        "mape": 1.0,
        "bias": -10.0,
        "upr": 100.0,
    },
]


def _day_stamps(day: str) -> pd.Index:
    idx = pd.date_range(f"{day}T00:00:00", periods=24, freq="h", tz="UTC")
    return idx.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_submission(team_dir: Path, day: str, mw: float) -> None:
    team_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp_utc": _day_stamps(day), "forecast_mw": [mw] * 24}).to_csv(
        team_dir / f"{day}.csv", index=False
    )


@pytest.fixture
def leaderboard(tmp_path: Path) -> Path:
    root = tmp_path / "challenge-leaderboard"
    (root / "data").mkdir(parents=True)
    actuals = pd.concat(
        pd.DataFrame({"timestamp_utc": _day_stamps(day), "load_mw": [1000.0] * 24})
        for day in DAYS
    )
    actuals.to_parquet(root / "data" / "actual_load.parquet", index=False)
    pd.DataFrame(OFFICIAL_ROWS).to_parquet(
        root / "data" / "scores.parquet", index=False
    )
    team_dir = root / "submissions" / TEAM
    _write_submission(team_dir, "2026-06-08", 1010.0)
    _write_submission(team_dir, "2026-06-10", 990.0)
    return root


def _published_entry(**overrides: float) -> dict[str, object]:
    entry: dict[str, object] = {
        "team_id": TEAM,
        "mean_mae": 10.0,
        "mean_rmse": 10.0,
        "mean_mape": 1.0,
        "mean_bias": -10.0,
        "mean_upr": 100.0,
        "n_submissions": 2,
    }
    entry.update(overrides)
    return entry


def _write_scores_json(path: Path, entries: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(entries))
    return path


def test_score_submission_replicates_official_formula() -> None:
    metrics = rc.score_submission(np.array([110.0, 180.0]), np.array([100.0, 200.0]))
    assert metrics == {
        "mae": 15.0,
        "rmse": round(math.sqrt(250.0), 4),
        "mape": 10.0,
        "bias": -5.0,
        "upr": 50.0,
    }


def test_score_submission_masks_zero_actual_hours_in_mape() -> None:
    metrics = rc.score_submission(np.array([10.0, 110.0]), np.array([0.0, 100.0]))
    assert metrics["mape"] == 10.0
    assert metrics["mae"] == 10.0


def test_score_submission_all_zero_actuals_yield_nan_mape() -> None:
    metrics = rc.score_submission(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert math.isnan(metrics["mape"])


def test_values_equal_treats_nan_as_equal() -> None:
    assert rc._values_equal(float("nan"), float("nan"))
    assert rc._values_equal(1.0, 1.0)
    assert not rc._values_equal(1.0, 2.0)


def test_replay_reproduces_official(leaderboard: Path) -> None:
    report = rc.replay(leaderboard, TEAM, RESTART)
    assert report.matches
    assert [d.replayed.target_date for d in report.days] == list(DAYS)
    locf = report.days[2].replayed
    assert locf.carried_forward and locf.source_date == "2026-06-10"
    assert report.live is not None and report.live.n_days == 2
    assert report.live.means == {
        "mae": 10.0,
        "rmse": 10.0,
        "mape": 1.0,
        "bias": -10.0,
        "upr": 100.0,
    }
    assert report.test is not None and report.test.n_days == 1
    assert report.test.means["bias"] == 10.0


def test_replay_detects_tampered_official_metric(leaderboard: Path) -> None:
    scores_path = leaderboard / "data" / "scores.parquet"
    scores = pd.read_parquet(scores_path)
    scores.loc[scores["target_date"] == "2026-06-10", "mae"] = 11.0
    scores.to_parquet(scores_path, index=False)

    report = rc.replay(leaderboard, TEAM, RESTART)
    assert not report.matches
    bad = [d for d in report.days if not d.matches]
    assert [d.replayed.target_date for d in bad] == ["2026-06-10"]
    markdown = rc.render_markdown(report, None)
    assert "Mismatches" in markdown
    assert "replay 10.0000 ≠ official 11.0000" in markdown


def test_replay_detects_wrong_graded_source(leaderboard: Path) -> None:
    scores_path = leaderboard / "data" / "scores.parquet"
    scores = pd.read_parquet(scores_path)
    scores.loc[scores["target_date"] == "2026-06-11", "carried_forward"] = False
    scores.loc[scores["target_date"] == "2026-06-11", "source_date"] = "2026-06-11"
    scores.to_parquet(scores_path, index=False)

    report = rc.replay(leaderboard, TEAM, RESTART)
    assert not report.matches
    assert "graded CSV" in rc.render_markdown(report, None)


def test_replay_raises_on_incomplete_actuals(leaderboard: Path) -> None:
    actuals_path = leaderboard / "data" / "actual_load.parquet"
    actuals = pd.read_parquet(actuals_path)
    actuals = actuals[actuals["timestamp_utc"] != "2026-06-10T07:00:00Z"]
    actuals.to_parquet(actuals_path, index=False)
    with pytest.raises(ValueError, match="incomplete"):
        rc.replay(leaderboard, TEAM, RESTART)


def test_replay_raises_on_nan_actuals(leaderboard: Path) -> None:
    actuals_path = leaderboard / "data" / "actual_load.parquet"
    actuals = pd.read_parquet(actuals_path)
    actuals.loc[actuals["timestamp_utc"] == "2026-06-10T07:00:00Z", "load_mw"] = float(
        "nan"
    )
    actuals.to_parquet(actuals_path, index=False)
    with pytest.raises(ValueError, match="incomplete"):
        rc.replay(leaderboard, TEAM, RESTART)


def test_replay_raises_on_unknown_team(leaderboard: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        rc.replay(leaderboard, "nonexistent", RESTART)


def test_replay_raises_when_no_submission_covers_a_scored_day(
    leaderboard: Path,
) -> None:
    for csv in (leaderboard / "submissions" / TEAM).glob("*.csv"):
        csv.unlink()
    with pytest.raises(ValueError, match="No submission"):
        rc.replay(leaderboard, TEAM, RESTART)


def test_replay_raises_on_wrong_submission_length(leaderboard: Path) -> None:
    csv = leaderboard / "submissions" / TEAM / "2026-06-10.csv"
    pd.read_csv(csv).head(23).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="23 rows"):
        rc.replay(leaderboard, TEAM, RESTART)


def test_published_comparison_matches(leaderboard: Path, tmp_path: Path) -> None:
    report = rc.replay(leaderboard, TEAM, RESTART)
    scores_json = _write_scores_json(tmp_path / "scores.json", [_published_entry()])
    published = rc.compare_published(report, scores_json)
    assert published.matches


def test_published_comparison_detects_deviation(
    leaderboard: Path, tmp_path: Path
) -> None:
    report = rc.replay(leaderboard, TEAM, RESTART)
    scores_json = _write_scores_json(
        tmp_path / "scores.json", [_published_entry(mean_mae=12.0)]
    )
    published = rc.compare_published(report, scores_json)
    assert not published.matches
    markdown = rc.render_markdown(report, published)
    assert "❌" in markdown


def test_published_comparison_team_missing_from_live_board(
    leaderboard: Path, tmp_path: Path
) -> None:
    report = rc.replay(leaderboard, TEAM, RESTART)
    scores_json = _write_scores_json(tmp_path / "scores.json", [])
    assert not rc.compare_published(report, scores_json).matches


def test_published_comparison_consistent_when_not_live(
    leaderboard: Path, tmp_path: Path
) -> None:
    # Restart date past every scored day → we are not on the live board, and
    # neither is the published page listing us: consistent.
    report = rc.replay(leaderboard, TEAM, "2027-01-01")
    assert report.live is None
    scores_json = _write_scores_json(tmp_path / "scores.json", [])
    assert rc.compare_published(report, scores_json).matches
    markdown = rc.render_markdown(report, None)
    assert "No live-phase day scored yet" in markdown


def test_render_markdown_without_published_shows_aggregate(
    leaderboard: Path,
) -> None:
    report = rc.replay(leaderboard, TEAM, RESTART)
    markdown = rc.render_markdown(report, None)
    assert "✅ The replay reproduces" in markdown
    assert "LOCF ← 2026-06-10" in markdown
    assert "mean MAE 10.0000" in markdown


def test_run_writes_summary_and_reports_success(
    leaderboard: Path, tmp_path: Path
) -> None:
    summary = tmp_path / "summary.md"
    scores_json = _write_scores_json(tmp_path / "scores.json", [_published_entry()])
    assert rc.run(leaderboard, TEAM, RESTART, scores_json, summary)
    assert "Competition replay — ddkast" in summary.read_text()


def test_run_returns_false_on_mismatch(leaderboard: Path, tmp_path: Path) -> None:
    scores_json = _write_scores_json(
        tmp_path / "scores.json", [_published_entry(n_submissions=3)]
    )
    assert not rc.run(leaderboard, TEAM, RESTART, scores_json)
