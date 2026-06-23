# Model Card — Load Forecaster (`load_mw`, DE_LU)

This card documents a **reproducible training procedure**, not a static artifact. No single
`.joblib` ships in this repository — `models/` is gitignored, and every rolling-origin fold
produces its own model under `models/folds/<fold_id>/`, refit on demand by `train`. What is
fixed and auditable is the recipe below, applied identically at every origin.

For the pipeline/toolkit this model runs inside (CLI, architecture, audit commands), see
[`MODEL_CARD.md`](MODEL_CARD.md) at the repo root.

## Model Details

- **Target**: `load_mw` — hourly electricity load, `DE_LU` bidding zone (Germany/Luxembourg).
- **Estimator**: `LGBMRegressor` wrapped in `spotforecast2_safe.forecaster.recursive.ForecasterRecursive` ([forecaster.py:56-66](src/ddkast/models/forecaster.py)).
- **Hyperparameters fixed for determinism (CR-2)**: `lags=168` (one week of hourly history), `random_state=42`, `deterministic=True`, `force_col_wise=True`. These mirror `spotforecast2-safe`'s own determinism guarantees — same input is expected to produce a bit-identical model.
- **Artifact**: not persisted to git. Each fold's model is a local build product under `models/folds/<fold_id>/`, written by `train` and read by `predict`; nothing under `models/` should be treated as "the" shipped model.

## Intended Use and Scope

The model produces a 24-hour-ahead hourly forecast that feeds the daily leaderboard submission. One rolling origin = one fold = one fitted model; `format-submission` selects exactly the fold whose forecast block covers tomorrow's submission window and raises if zero or more than one fold matches ([format_submission.py:45-61](src/ddkast/pipeline/format_submission.py)). It is not intended for any horizon, zone, or use beyond this.

## Training Data

- **Sources**: ENTSO-E actual load and day-ahead forecast (DAF), plus Open-Meteo weather, fetched for a single point at lat/lon `50.110924, 8.682127` — a representative station for the `DE_LU` zone, **not** zone-wide spatial coverage.
- **Window**: 2022-01-01 to 2026-04-30.
- **Cleaning**: outliers removed via manual IQR bounds with multiplier `3.0`; gaps interpolated linearly up to `3` hours, with longer gaps rejected rather than silently filled.

## Training Procedure

This is a **rolling-origin walk-forward backtest**, not a single temporal train/test split. `train` fits one model per fold, each on an expanding window of all data up to (and including) that fold's origin ([train.py:49-51](src/ddkast/pipeline/train.py)). The fold set spans `n_forecasts=365` daily origins from 2025-05-02 to 2026-05-01 ([config.py:66-68](src/ddkast/config.py)), giving 365 independently-fitted models for evaluation, plus a single pinned fold per day in the production CI job.

No hyperparameter tuning and no SHAP attribution are implemented anywhere in `src/` (confirmed by grep — no hits for either). These remain genuinely deferred, not partially built.

## Evaluation

`evaluate` scores each fold's model against two benchmarks — 7-day seasonal naive and the ENTSO-E published day-ahead forecast — on four metrics (MAE, RMSE, MAPE, SMAPE), then aggregates the per-fold results to mean/std/median per forecaster×metric, plus `skill_vs_naive` and `skill_vs_daf` (1 − MAE_model / MAE_baseline) ([evaluate.py:106-141](src/ddkast/pipeline/evaluate.py)).

No evaluation run has been executed for this card — there is no cached summary, and producing real numbers means training and scoring all 365 folds. Treat any numbers quoted elsewhere as illustrative, not this model's actual performance; run `uv run ddkast evaluate` against a synced dataset to get current figures.

## Fail-safe Behavior (CR-3)

Both the metrics layer and the fold builder fail loud rather than degrade silently:

- `metrics.py` raises `ValueError` on empty, length-mismatched, or non-finite (NaN/±inf) input, before computing MAE/RMSE/MAPE/SMAPE — guarding the MAPE/SMAPE denominators explicitly even though load is strictly positive in practice ([metrics.py:16-27](src/ddkast/evaluation/metrics.py)).
- `folds.py`'s `build_folds` raises on an empty/unsorted index, inconsistent forecast-window configuration, or insufficient history at the earliest origin — the same fail-loud philosophy `spotforecast2-safe` enforces at the estimator level.

## Limitations / Out-of-Distribution (ODD) Boundaries

Inherited from `spotforecast2-safe`'s own determinism/fail-safe guarantees, plus project-specific gaps:

- **Single weather station** stands in for the entire `DE_LU` zone — no spatial aggregation or multi-station coverage.
- **No uncertainty quantification** — forecasts are point estimates only, no prediction intervals.
- **No SHAP attribution** — feature importance/explainability is not yet implemented.
- **No hyperparameter tuning** — `LGBMRegressor` defaults (beyond the fixed determinism flags) are unvalidated against alternatives.

## Compliance Notes

This model is part of a high-risk AI system under EU AI Act Annex III No. 2 (critical infrastructure). The library-compliance question — whether the forecasting library itself needs to be "verified" or EU AI Act compliant — has been resolved at the project level: `spotforecast2-safe` is adopted as the only production-path library, documented as EU AI Act compliant and deterministic in `CLAUDE.md`, with CR-1 through CR-4 enforced by tests. No open compliance questions remain at the model-recipe level beyond the limitations above.

## Disclaimer

This model card describes the system as implemented at the time of writing. It is provided for transparency and audit purposes under EU AI Act Art. 13 and does not constitute a safety or performance guarantee. The system is a competition/research artifact and has not been validated for production grid operations.
