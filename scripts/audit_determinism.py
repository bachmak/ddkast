"""Determinism audit: run the whole pipeline twice from clean state and compare.

Evidence generator for CR-2 (same input → bit-identical output) and, at the same
time, a liveness check that all six stages still run end to end on real data.

What it does
------------
1. Downloads the raw data **once** into a shared snapshot (proves ``download``
   works, and pins the inputs so a mid-audit ENTSO-E revision cannot make the two
   runs disagree for reasons outside our code).
2. Copies that snapshot into two pristine run directories and executes
   ``download → merge → train → predict → evaluate → visualise`` in each, with
   ``n_forecasts=2`` so the audit costs two forecast days instead of a year.
3. Hashes every artifact the two runs produced, compares the parquet data on
   content as well, and writes a timestamped report next to them.

The claim being audited is that the **forecasts** reproduce: every parquet file —
the per-fold predictions above all, plus the cleaned inputs and the evaluation —
must match exactly. Persisted models and rendered plots are hashed and listed for
the record but never fail the audit, since both embed wall-clock stamps that move
between runs regardless of what the model learned.

The runs are fully sandboxed: ``data_dir``, ``models_dir`` and ``plots_dir`` are
redirected into ``audit/<timestamp>/run_{a,b}/``, so the repository's real
``data/`` and ``models/`` are never read from or written to. "Clearing the cache"
between the runs is therefore structural — run B starts in a directory that did
not exist a moment earlier and cannot cache-hit off run A.

Only the forecast window is overridden (``n_forecasts``, ``forecasts_start``,
``forecasts_end``). The download range and every model/cleaning knob keep their
production values, because the point is to audit the production configuration.

Usage::

    uv run python scripts/audit_determinism.py
    uv run python scripts/audit_determinism.py --start 2026-07-14 --end 2026-07-15

Exit code is 0 on PASS (or PASS-with-warnings) and 1 on FAIL, so it can gate CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from ddkast.config import Config, load
from ddkast.data.store import ParquetStore
from ddkast.pipeline import download, evaluate, merge, predict, train, visualise

_console = Console()

# Two origins is the smallest fold set that still exercises the rolling-origin
# machinery (a stride has to be derived, more than one model is fitted).
_N_FORECASTS = 2
_DAY = pd.Timedelta(hours=24)

# Rendered figures are presentation, not model output: plotly's write_html embeds
# a fresh div uuid per call and PDFs carry a creation date, so a byte difference
# here says nothing about determinism. Pinning SOURCE_DATE_EPOCH removes the PDF
# timestamp; HTML is compared but never allowed to fail the audit.
_PLOT_SUFFIXES = (".html", ".pdf", ".png", ".svg")
_SOURCE_DATE_EPOCH = "1704067200"  # 2024-01-01T00:00:00Z

# Persisted models are hashed and listed for the record, but a byte difference in
# one is never a failure: spotforecast2-safe stamps fit_date/creation_date onto
# every fitted forecaster, so two runs cannot produce identical .joblib files. What
# has to reproduce is the forecast the model emits, and that is compared exactly.

_PROVENANCE_PACKAGES = (
    "ddkast",
    "spotforecast2-safe",
    "pandas",
    "numpy",
    "pyarrow",
    "lightgbm",
    "entsoe-py",
    "holidays",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "plotly",
)


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


@dataclass
class Comparison:
    """The verdict on one artifact, as it appears in both runs."""

    path: str
    kind: str  # data | model | plot | other
    status: str  # identical | byte-differs | content-differs | missing
    hash_a: str | None
    hash_b: str | None
    detail: str = ""

    @property
    def is_failure(self) -> bool:
        """Only a content difference (or a missing artifact) is a determinism defect.

        ``byte-differs`` survives as a warning: on parquet it means writer metadata
        moved while the table did not, and on models and plots it means an embedded
        wall-clock stamp moved. Neither is forecast output drifting.
        """
        return self.status in ("missing", "content-differs")


def _kind_of(path: Path) -> str:
    if path.suffix == ".parquet":
        return "data"
    if path.suffix == ".joblib":
        return "model"
    if path.suffix in _PLOT_SUFFIXES:
        return "plot"
    return "other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_parquet(path_a: Path, path_b: Path) -> tuple[bool, str]:
    """Whether two parquet files carry the same table, and how they differ.

    Byte equality is the stricter claim but a noisy one — writer metadata can
    shift without the data moving. This is the claim that actually matters: same
    values, same dtypes, same index, same column order.
    """
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    if list(df_a.columns) != list(df_b.columns):
        return False, f"columns {list(df_a.columns)} vs {list(df_b.columns)}"
    if df_a.shape != df_b.shape:
        return False, f"shape {df_a.shape} vs {df_b.shape}"
    if not df_a.index.equals(df_b.index):
        return False, "index differs"
    if list(df_a.dtypes) != list(df_b.dtypes):
        return False, "dtypes differ"
    if df_a.equals(df_b):
        return True, ""

    numeric = df_a.select_dtypes("number").columns
    deltas = {
        str(col): float((df_a[col] - df_b[col]).abs().max())
        for col in numeric
        if not df_a[col].equals(df_b[col])
    }
    if deltas:
        worst = max(deltas.items(), key=lambda kv: kv[1])
        return False, f"max |Δ| {worst[1]:.6g} in {worst[0]} ({len(deltas)} columns)"
    return False, "non-numeric values differ"


def _relative_files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def compare_runs(dir_a: Path, dir_b: Path) -> list[Comparison]:
    """Compare every artifact under two run directories, in a fixed order (CR-2)."""
    results: list[Comparison] = []
    for rel in sorted(_relative_files(dir_a) | _relative_files(dir_b)):
        path_a, path_b = dir_a / rel, dir_b / rel
        kind = _kind_of(path_a)
        rel_str = rel.as_posix()

        if not path_a.exists() or not path_b.exists():
            present = "run_a" if path_a.exists() else "run_b"
            results.append(
                Comparison(rel_str, kind, "missing", None, None, f"only in {present}")
            )
            continue

        hash_a, hash_b = _sha256(path_a), _sha256(path_b)
        if hash_a == hash_b:
            results.append(Comparison(rel_str, kind, "identical", hash_a, hash_b))
            continue

        if kind == "data":
            same, detail = _compare_parquet(path_a, path_b)
            status = "byte-differs" if same else "content-differs"
            note = detail or "same table, different bytes"
            results.append(Comparison(rel_str, kind, status, hash_a, hash_b, note))
            continue

        results.append(Comparison(rel_str, kind, "byte-differs", hash_a, hash_b))
    return results


# --------------------------------------------------------------------------- #
# forecast window
# --------------------------------------------------------------------------- #


def _as_utc(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _default_window(
    index: pd.DatetimeIndex, config: Config
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The two most recent midnight origins whose blocks the actuals fully cover.

    Documented in the README: with no ``--start``/``--end``, the audit walks back
    from the end of the cleaned load series and takes the last two complete days,
    so ``evaluate`` always has ground truth to score both folds against.
    """
    actual_end = index.max() + pd.Timedelta(config.resolution)
    last_origin = (actual_end - pd.Timedelta(hours=config.horizon)).floor("D")
    return last_origin - _DAY, last_origin


def resolve_window(
    index: pd.DatetimeIndex,
    config: Config,
    start: datetime | None,
    end: datetime | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pick the two forecast origins and refuse anything the audit cannot score.

    Fail-safe (CR-3): an unscorable window would produce an audit that compares
    two empty evaluations and reports PASS, which is worse than an error.
    """
    if (start is None) != (end is None):
        raise ValueError("Pass both --start and --end, or neither.")

    if start is None or end is None:
        first, second = _default_window(index, config)
    else:
        first, second = _as_utc(start), _as_utc(end)

    if second - first != _DAY:
        raise ValueError(
            f"--start and --end must be exactly 24h apart (two consecutive days), "
            f"got {first} → {second}."
        )

    actual_end = index.max() + pd.Timedelta(config.resolution)
    block_end = second + pd.Timedelta(hours=config.horizon)
    if block_end > actual_end:
        raise ValueError(
            f"Forecast block ends {block_end}, past the actuals ({actual_end}); "
            "evaluate would skip both folds as unrealized. Choose earlier days."
        )
    if first - index.min() < pd.Timedelta(hours=config.lags):
        raise ValueError(
            f"Origin {first} leaves < {config.lags}h of history after {index.min()}."
        )
    return first, second


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #


def _audit_config(
    base: Config,
    root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Config:
    """The production config with only the sandbox paths and window overridden."""
    return base.model_copy(
        update={
            "data_dir": root / "data",
            "models_dir": root / "models",
            "plots_dir": root / "plots",
            "n_forecasts": _N_FORECASTS,
            "forecasts_start": start.to_pydatetime(),
            "forecasts_end": end.to_pydatetime(),
        }
    )


def _prepare_snapshot(base: Config, root: Path) -> Config:
    """Fetch the raw data once and clean it, so the window can be derived from it."""
    config = base.model_copy(update={"data_dir": root / "data"})
    _console.rule("[bold]snapshot[/bold] — one real download, shared by both runs")
    download.run(config)
    merge.run(config)
    return config


def _execute_run(
    base: Config,
    root: Path,
    raw_src: Path,
    window: tuple[pd.Timestamp, pd.Timestamp],
    label: str,
) -> None:
    """Run all six stages in a pristine directory seeded with the raw snapshot."""
    config = _audit_config(base, root, *window)
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(raw_src.iterdir()):
        shutil.copy2(item, config.raw_dir / item.name)

    _console.rule(f"[bold]{label}[/bold] — {root}")
    download.run(config)  # cache hit against the copied fingerprint
    merge.run(config)
    train.run(config)
    predict.run(config)
    evaluate.run(config)
    visualise.run(config)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _git_state() -> dict[str, str]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unavailable"

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "yes" if status and status != "unavailable" else "no",
    }


def _provenance() -> dict[str, object]:
    """Everything needed to reconstruct this audit later (Art. 11, 12)."""
    packages: dict[str, str] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "git": _git_state(),
        "packages": packages,
    }


_STATUS_STYLE = {
    "identical": "green",
    "byte-differs": "yellow",
    "content-differs": "red",
    "missing": "red",
}


def _render(results: list[Comparison]) -> None:
    table = Table(title="run_a vs run_b", show_lines=False)
    table.add_column("artifact", overflow="fold")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for r in results:
        style = _STATUS_STYLE[r.status]
        table.add_row(r.path, r.kind, f"[{style}]{r.status}[/{style}]", r.detail)
    _console.print(table)


def _write_report(
    out_dir: Path,
    results: list[Comparison],
    window: tuple[pd.Timestamp, pd.Timestamp],
    failed: list[Comparison],
) -> None:
    report = {
        "verdict": "FAIL" if failed else "PASS",
        "forecast_window": {
            "n_forecasts": _N_FORECASTS,
            "origins": [str(window[0]), str(window[1])],
        },
        "provenance": _provenance(),
        "artifacts": [asdict(r) for r in results],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Determinism audit",
        "",
        f"**Verdict:** {report['verdict']}",
        f"**Origins:** {window[0]} · {window[1]} (n_forecasts={_N_FORECASTS})",
        f"**Commit:** {_git_state()['commit']} (dirty: {_git_state()['dirty']})",
        "",
        "| artifact | kind | status | detail |",
        "| --- | --- | --- | --- |",
        *(f"| `{r.path}` | {r.kind} | {r.status} | {r.detail} |" for r in results),
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--start",
        type=datetime.fromisoformat,
        default=None,
        help="First forecast origin, ISO 8601 (default: second-to-last complete day).",
    )
    parser.add_argument(
        "--end",
        type=datetime.fromisoformat,
        default=None,
        help="Second forecast origin, ISO 8601; must be --start + 24h.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("audit"),
        help="Directory the timestamped audit run is written into (default: audit/).",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.toml"), help="Path to config.toml."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ.setdefault("SOURCE_DATE_EPOCH", _SOURCE_DATE_EPOCH)

    base = load(args.config)
    root = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True)

    snapshot = _prepare_snapshot(base, root / "snapshot")
    index = ParquetStore(snapshot.processed_dir).read(snapshot.processed_load).index
    window = resolve_window(index, base, args.start, args.end)  # type: ignore[arg-type]

    _console.print()
    _console.print(
        f"[bold]running audit on {window[0]:%Y-%m-%d %H:%M UTC} "
        f"→ {window[1]:%Y-%m-%d %H:%M UTC}[/bold] "
        f"({_N_FORECASTS} forecast origins, {base.horizon}h each)"
    )

    for label in ("run_a", "run_b"):
        _execute_run(base, root / label, snapshot.raw_dir, window, label)

    _console.rule("[bold]comparison[/bold]")
    results = compare_runs(root / "run_a", root / "run_b")
    _render(results)

    failed = [r for r in results if r.is_failure]
    warned = [r for r in results if r.status != "identical" and not r.is_failure]
    _write_report(root, results, window, failed)

    if failed:
        _console.print(
            f"[bold red]✗ FAIL[/bold red] — {len(failed)}/{len(results)} artifacts "
            f"differ. Report: {root / 'report.md'}"
        )
        return 1
    n_data = sum(1 for r in results if r.kind == "data")
    _console.print(
        f"[bold green]✓ PASS[/bold green] — forecasts reproduce: {n_data} data "
        f"artifacts match across both runs"
        + (
            f", {len(warned)} model/plot file(s) differ byte-wise "
            "(embedded timestamps, not audited)"
            if warned
            else ""
        )
        + f". Report: {root / 'report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
