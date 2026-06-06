from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from rich.console import Console
from rich.table import Table

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.evaluation.metrics import report
from ddkast.folds import Fold, build_folds

_console = Console()

_WEEK_HOURS = 168
_METRICS = ("MAE", "RMSE", "MAPE", "SMAPE")
# Fixed forecaster order → deterministic metrics/summary parquet (CR-2).
_FORECASTERS = ("model", "naive", "daf")


@dataclass
class _Aligned:
    """A fold's forecast block lined up against its actuals + benchmarks."""

    actual: pd.Series[float]
    model: pd.Series[float]
    naive: pd.Series[float]
    daf: pd.Series[float]
    daf_available: bool


def _fold_is_scorable(fold: Fold, last_actual: pd.Timestamp) -> bool:
    """Whether a fold's forecast block is covered by the actuals — realized vs future.

    This is the single place the system tells realized from not-yet-realized time,
    and it decides it purely from the latest ground-truth timestamp (CR-3): a block
    fully past ``last_actual`` is not yet scorable (``False``); a fully covered block
    is scorable (``True``); a block straddling ``last_actual`` is a data defect and
    raises rather than scoring a partly-realized block against imputed values.
    """
    if fold.forecast_start > last_actual:
        return False
    if fold.forecast_end <= last_actual:
        return True
    raise ValueError(
        f"Fold {fold.fold_id}: forecast block straddles the last actual {last_actual} "
        f"({fold.forecast_start} → {fold.forecast_end}); expected it fully realized "
        "or fully in the future."
    )


def _align_fold(
    fold: Fold,
    predictions: pd.Series[float],
    target: pd.Series[float],
    daf: pd.Series[float],
) -> _Aligned:
    """Line a scorable fold's forecast up against its actuals, 7-day naive, and DAF.

    The fold is already known realized (see :func:`_fold_is_scorable`), so its actuals
    are present. Fail-safe (CR-3): a missing 7-day-naive lookback (download window too
    short) is a data error and raises rather than scoring against imputed values.
    """
    idx = predictions.index
    assert isinstance(idx, pd.DatetimeIndex)

    actual = target.reindex(idx)
    naive_values = target.reindex(idx - pd.Timedelta(hours=_WEEK_HOURS)).to_numpy()
    naive = pd.Series(naive_values, index=idx, dtype=float)
    if naive.isna().any():
        raise ValueError(
            f"Fold {fold.fold_id}: 7-day-naive lookback falls outside the data range. "
            "Extend download_start or move forecasts_start later."
        )

    fold_daf = daf.reindex(idx)
    return _Aligned(
        actual=actual,
        model=predictions,
        naive=naive,
        daf=fold_daf,
        daf_available=not bool(fold_daf.isna().any()),
    )


def _metric_rows(fold: Fold, aligned: _Aligned) -> list[dict[str, object]]:
    """Long-form per-fold metric rows for model + naive + (if available) DAF."""
    series_by_name: dict[str, pd.Series[float]] = {
        "model": aligned.model,
        "naive": aligned.naive,
    }
    if aligned.daf_available:
        series_by_name["daf"] = aligned.daf

    rows: list[dict[str, object]] = []
    for name in _FORECASTERS:
        if name not in series_by_name:
            continue
        metrics = report(aligned.actual, series_by_name[name])
        rows.append(
            {
                "fold_id": fold.fold_id,
                "origin": fold.forecast_start,
                "forecaster": name,
                **metrics,
            }
        )
    return rows


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-form per-fold metrics into per-forecaster×metric mean/std/median.

    Appends skill rows (``forecaster="model"``): ``skill_vs_naive`` = 1 -
    MAE_model/MAE_naive and, when DAF was scored, ``skill_vs_daf``. Fixed order (CR-2).
    """
    present = [f for f in _FORECASTERS if f in set(metrics_df["forecaster"])]
    mean_mae: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for forecaster in present:
        sub = metrics_df.loc[metrics_df["forecaster"] == forecaster]
        for metric in _METRICS:
            col = sub[metric]
            rows.append(
                {
                    "forecaster": forecaster,
                    "metric": metric,
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "median": float(col.median()),
                }
            )
        mean_mae[forecaster] = float(sub["MAE"].mean())

    for baseline, label in (("naive", "skill_vs_naive"), ("daf", "skill_vs_daf")):
        if baseline in mean_mae and mean_mae[baseline] != 0:
            rows.append(
                {
                    "forecaster": "model",
                    "metric": label,
                    "mean": 1 - mean_mae["model"] / mean_mae[baseline],
                    "std": float("nan"),
                    "median": float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=["forecaster", "metric", "mean", "std", "median"])


def flatten_summary(summary_df: pd.DataFrame) -> dict[str, float]:
    """Flatten the summary frame to a finite-only ``{key: value}`` dict.

    Stat rows become ``"{forecaster}_{metric}_{stat}"``; skill rows keep their bare
    label (``"skill_vs_naive"``). Non-finite values (e.g. single-fold std, absent skill)
    are dropped so the dict is a stable basis for golden comparison and quality gates.
    """
    flat: dict[str, float] = {}
    for rec in summary_df.to_dict(orient="records"):
        metric = str(rec["metric"])
        if metric.startswith("skill_"):
            value = float(rec["mean"])
            if math.isfinite(value):
                flat[metric] = value
            continue
        for stat in ("mean", "std", "median"):
            value = float(rec[stat])
            if math.isfinite(value):
                flat[f"{rec['forecaster']}_{metric}_{stat}"] = value
    return flat


def _evaluation_series(aligned_folds: list[_Aligned]) -> pd.DataFrame:
    """Concatenate the scored folds into one predicted-vs-actual series for visualise.

    Folds are non-overlapping daily blocks, so the concatenation is a continuous view of
    the backtest span. ``residuals_daf`` is included only when DAF covers all folds.
    """
    actual = pd.concat([a.actual for a in aligned_folds])
    forecast = pd.concat([a.model for a in aligned_folds])
    entso_daf = pd.concat([a.daf for a in aligned_folds])
    series: dict[str, pd.Series[float]] = {
        "actual": actual,
        "forecast": forecast,
        "entso_daf": entso_daf,
        "residuals_forecast": actual - forecast,
    }
    if all(a.daf_available for a in aligned_folds):
        series["residuals_daf"] = actual - entso_daf
    return pd.DataFrame(series)


def _render_table(summary_df: pd.DataFrame, n_folds: int) -> Table:
    present = [
        f for f in _FORECASTERS if f in set(summary_df["forecaster"]) and f != "model"
    ]
    table = Table(title=f"Rolling-origin backtest ({n_folds} scored folds)")
    table.add_column("Metric", style="bold")
    label = {"naive": "7-day Naive", "daf": "ENTSO-E DAF"}
    for forecaster in present:
        table.add_column(label[forecaster], justify="right")
    table.add_column("Model", justify="right", style="green")

    def cell(forecaster: str, metric: str, unit: str) -> str:
        row = summary_df[
            (summary_df["forecaster"] == forecaster) & (summary_df["metric"] == metric)
        ].iloc[0]
        return f"{row['mean']:,.1f} ± {row['std']:,.1f}{unit}"

    for metric in _METRICS:
        unit = " %" if metric in ("MAPE", "SMAPE") else " MW"
        cells = [cell(f, metric, unit) for f in present]
        cells.append(cell("model", metric, unit))
        table.add_row(metric, *cells)
    return table


def run(config: Config) -> None:
    """Score every fold whose block the actuals already cover; persist the metrics.

    Folds whose forecast block lies entirely in the future (no actuals yet) are skipped
    — scorability is decided here from the ground-truth load, not flagged upstream.
    """
    processed = ParquetStore(config.processed_dir)
    load_df = processed.read(config.processed_load)
    entso = processed.read(config.processed_entso_forecast)
    target: pd.Series[float] = load_df[config.model_target]
    daf: pd.Series[float] = entso["forecast_mw"]

    folds = build_folds(load_df.index, config)  # type: ignore[arg-type]
    last_actual: pd.Timestamp = target.index.max()  # type: ignore[assignment]
    rows: list[dict[str, object]] = []
    aligned_folds: list[_Aligned] = []
    for fold in folds:
        if not _fold_is_scorable(fold, last_actual):
            continue  # block not yet realized — nothing to score against
        predictions: pd.Series[float] = processed.read(
            f"{config.predictions_subdir}/{fold.fold_id}"
        )[config.model_target]
        aligned = _align_fold(fold, predictions, target, daf)
        aligned_folds.append(aligned)
        rows.extend(_metric_rows(fold, aligned))

    if not aligned_folds:
        raise ValueError(
            "No scorable folds: every fold's forecast block lies past the available "
            "actuals. Extend the data window or move forecasts_end within the actuals."
        )

    metrics_df = pd.DataFrame(
        rows, columns=["fold_id", "origin", "forecaster", *_METRICS]
    )
    summary_df = summarize(metrics_df)

    processed.write(config.evaluation_metrics, metrics_df)
    processed.write(config.evaluation_summary, summary_df)
    processed.write(config.evaluation_series, _evaluation_series(aligned_folds))

    _console.print(_render_table(summary_df, len(aligned_folds)))
