# Model Card — ddkast Load Forecaster

This card describes what the ddkast forecasting system is, the features it uses, the
output it produces, and the conditions under which its results are valid. It follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy. 

## 1. Model Details

| Field | Value |
| --- | --- |
| Name | ddkast |
| Target | `load_mw` — hourly electricity load, `DE_LU` bidding zone (Germany/Luxembourg) |
| Type | 7-command Typer CLI (`download`, `merge`, `train`, `predict`, `evaluate`, `format-submission`, `visualise`) orchestrating a deterministic time-series forecasting pipeline |
| Estimator | `LGBMRegressor` wrapped in `spotforecast2_safe.forecaster.recursive.ForecasterRecursive` ([forecaster.py:56-66](src/ddkast/models/forecaster.py)) |
| Horizon | 24 hours ahead, hourly resolution |
| Domain | Electricity load forecasting for Germany (ENTSO-E) — a high-risk AI system under EU AI Act Annex III No. 2 (critical infrastructure) |
| Repository | <https://github.com/bachmak/ddkast> |

**Production-path dependency.** [`spotforecast2-safe`](https://pypi.org/project/spotforecast2-safe/)
is the only forecasting library permitted in `src/`. It is EU AI Act compliant and
deterministic. The unconstrained `spotforecast2` package may be used for
development/experimentation only, never on the code path that produces a submitted or
evaluated forecast.

**Hyperparameters fixed for determinism (CR-2).** `lags=168` (one week of hourly
history), `random_state=42`, `deterministic=True`, `force_col_wise=True`. These mirror
`spotforecast2-safe`'s own determinism guarantees — the same input is expected to produce
a bit-identical model. No hyperparameter tuning is implemented; these are fixed choices,
not tuned optima.

## 2. Intended Use and Scope

ddkast is a **live leaderboard competition entry**, not a course demo and not an
operational grid tool. An end-to-end run of the pipeline performs the following: 
it downloads the freshest available data, trains, predicts, and submits **tomorrow's
24-hour forecast** as a CSV pull request against the upstream leaderboard repository.
`format-submission` selects exactly the one fold whose forecast block covers the
submission window and raises if zero or more than one fold matches
([format_submission.py:45-61](src/ddkast/pipeline/format_submission.py)) — a single
day-ahead block is the only output ever submitted.

**This is explicitly not validated for real grid operations.** It has not been assessed
for the accuracy, latency, or failure-mode guarantees that operational grid forecasting
would require, and it must not be used to inform actual dispatch, balancing, or
infrastructure decisions. It is a competition/research artifact provided for transparency
and audit under EU AI Act Art. 13, and does not constitute a safety or performance
guarantee.

## 3. Architecture

The pipeline is a linear chain of CLI stages, each reading and writing through the
`DataStore` protocol — no direct Parquet access in pipeline or model code:

```
download → merge → train → predict → evaluate
                                    → format-submission
                      (any stage) → visualise
```

Two structures shape how the later stages operate:

- **`folds.py`** is the single source of truth for forecast origins. `build_folds()`
  deterministically derives a list of `Fold` objects — rolling-origin, half-open
  forecast blocks — from the training history and `Config`. `evaluate` scores the full
  rolling-origin fold set; the daily-forecast CI job overrides `Config` to produce
  exactly one fold per run, pinned to the latest actual. Fold IDs and ordering are fixed,
  so the fold set is reproducible run to run (CR-2).
- **Per-fold model directory layout.** `train` persists one model per fold under
  `models/folds/<fold_id>/`, rather than shipping a single artifact; `predict` reads from
  the same path. This is why `models/` never contains one canonical forecaster — every
  fold gets its own, refit on its own expanding window.

## 4. Technical Specification

### Task and model family

The system performs recursive multi-step forecasting of the univariate `load_mw` series
from its own past values (168 lags), calendar features, and exogenous weather regressors.
The forecaster is a scikit-learn-compatible wrapper around a LightGBM regressor: the
wrapper handles feature construction and the recursive prediction loop, the regressor
handles learning. The target never appears in its own feature row, which prevents
look-ahead leakage by construction.

### Features

The exogenous feature matrix is built once per window and joined to the lag features
([features.py](src/ddkast/preprocessing/features.py)):

- **Calendar (cyclical), via `spotforecast2-safe`'s `ExogBuilder`** — Repeating Basis
  Function encodings of hour-of-day (10 basis functions), day-of-week (7), and
  month-of-year (6), so that, e.g., hour 23 sits next to hour 0.
- **Holidays** — German (`DE`) public-holiday flags for the `DE_LU` zone.
- **Weather (Open-Meteo, no API key)** — a single representative station for the zone
  (lat/lon `50.110924, 8.682127`, near Frankfurt), **not** zone-wide spatial coverage.
  The full served schema is consumed as exog: temperature, relative humidity,
  precipitation, rain, snowfall, weather code, sea-level and surface pressure, four
  cloud-cover levels, and wind speed / direction / gusts at 10 m.

Any NaN remaining after the calendar+weather join raises a `ValueError` before a model is
fit — features are never silently imputed (CR-3).

### Training procedure

Training is a **rolling-origin walk-forward backtest**, not a single temporal
train/test split. `train` fits one model per fold, each on an expanding window of all
data strictly before that fold's origin ([train.py:49-51](src/ddkast/pipeline/train.py)).
The evaluation fold set spans **365 daily origins** (one year of rolling origins), giving
365 independently fitted models for evaluation; the production CI job overrides this to a
single pinned origin per day. The exact origin dates shift daily with the automated run
and are not part of the model's behaviour, so they are intentionally not fixed here.

The library trains nothing on its own — learning happens entirely in the downstream
LightGBM regressor. No hyperparameter tuning and no SHAP attribution are implemented
anywhere in `src/`. 

## 5. Interfaces, Runtime, and Output

**Input.** The target is the numeric univariate `load_mw` series on a regular,
monotonic, hourly UTC index; exogenous features are a numeric frame aligned to that index
and complete. `merge` produces this analysis-ready dataset: load is cleaned and resampled
to the hourly UTC grid and is the reference every other series aligns to, while the ENTSO-E
day-ahead forecast and weather are tz-normalised to UTC and aligned onto it.

**Output.** A 24-hour-ahead hourly forecast series in megawatts, one block per fold. For
submission, that block is sliced to tomorrow 00:00–23:00 UTC, validated (exactly 24
strictly-positive finite values, correct start timestamp), rounded to 2 decimals, and
written as a leaderboard CSV. The system produces **point forecasts only** — no
prediction intervals or calibrated probabilities.

**Runtime & determinism.** Runs on Python 3.13+ on CPU; no GPU code. Bit-for-bit
reproducibility relies on fixed seeds, fixed fold/iteration order, and LightGBM's
`deterministic` / `force_col_wise` flags. A determinism audit script re-runs the pipeline
twice and diffs every artifact; in the latest run all **data artifacts are byte-identical
across runs**, while persisted `.joblib` models and the interactive HTML plot differ at
the byte level (serialization/timestamp noise) without affecting the forecasts.

## 6. Data and Operational Design Domain

### Data

- **Sources.** ENTSO-E actual load and day-ahead forecast (DAF), fetched in a single
  `download` pass and stored separately, plus Open-Meteo weather. ENTSO-E requires
  `ENTSOE_API_KEY` (env var); Open-Meteo needs no key.
- **Window.** From 2022-01-01 through the latest available day — roughly **4+ years of
  hourly history** at run time. The end date advances with each automated run, so it is
  expressed as a duration rather than pinned here.
- **Cleaning** ([clean.py](src/ddkast/preprocessing/clean.py)). Outliers removed via
  manual IQR bounds (multiplier `3.0`); gaps interpolated linearly up to `3` hours, with
  longer gaps left as NaN and rejected rather than silently filled. Both thresholds are
  configurable.

### ODD boundaries and limitations

The Operational Design Domain is the set of conditions under which results are valid;
outside them the system is designed to raise rather than return an unreliable forecast.

| Condition | Valid range | Outside the range |
| --- | --- | --- |
| Target series | numeric, univariate, regular monotonic hourly UTC index | `ValueError` |
| Exogenous features | numeric, complete, aligned to the target index | `ValueError` on any NaN |
| Minimum history | at least `lags` (168 h) before the earliest origin | `build_folds` raises |
| Load gaps | ≤ `max_interpolation_hours` (3 h) | rejected, not imputed |
| Metric inputs | finite, equal-length | `ValueError` (empty / mismatched / non-finite) |

Known, project-specific limitations:

- **Single weather station** stands in for the entire `DE_LU` zone — no spatial
  aggregation or multi-station coverage.
- **No uncertainty quantification** — point forecasts only.
- **No SHAP attribution** — feature importance is available only through LightGBM's own
  split/gain importances; no separate explainability backend ships.
- **No hyperparameter tuning** — LightGBM settings beyond the fixed determinism flags are
  unvalidated against alternatives.
- Forecast accuracy is bounded by the regressor and its training data, so concept drift,
  seasonal shifts, or regime changes degrade forecasts even when feature engineering
  stays correct.

## 7. Evaluation

Because no training runs inside `spotforecast2-safe`, accuracy is a property of the
deployment, not the library. `evaluate` scores each fold's model on a **rolling-origin
backtest** (365 daily origins, 24 h each) against two benchmarks, on four metrics, then
aggregates to mean/std/median per forecaster and to two skill scores
([evaluate.py:106-141](src/ddkast/pipeline/evaluate.py)). Numeric values depend on the
data vintage and are therefore not fixed in this card; reproduce them with
`uv run ddkast evaluate` on a synced dataset. 

**Benchmarks**:

- **7-day seasonal naive** ("same hour, one week ago") is the cheapest defensible
  forecast. Beating it shows the model has learned structure beyond weekly seasonality; a
  model that *cannot* beat it is not earning its complexity.
- **ENTSO-E published day-ahead forecast (DAF)** is the operational reference the market
  already publishes. Beating it is the demanding bar — it shows the model adds value over
  the relevant professional forecast for the same zone and horizon.

**Metrics**:

- **MAE** (primary) — average error in MW, in the units operators reason about; robust to
  outliers and the headline the model is optimised to reduce.
- **RMSE** — penalises large misses quadratically, so a gap between RMSE and MAE flags
  occasional big errors (e.g. holidays, extreme weather) rather than uniform drift.
- **MAPE / SMAPE** — percentage errors that make performance comparable across load
  levels and seasons; SMAPE is the symmetric variant that treats over- and
  under-prediction even-handedly.
- **Skill scores** — `skill_vs_naive` and `skill_vs_daf` (`1 − MAE_model / MAE_baseline`)
  express the same comparison as a single fraction: positive means the model beats that
  benchmark, and the magnitude is the fraction of the benchmark's error it removes.

## 8. Fail-safe Behaviour and Compliance (CR-1 to CR-4)

The system is built to support a high-risk AI system under the EU AI Act; the library
question is resolved at project level (`spotforecast2-safe` as the sole production-path
library, documented deterministic and compliant in `CLAUDE.md`). The four code rules are
enforced by the test suite, not a separate audit tool:

| Rule | Requirement | Where enforced |
| --- | --- | --- |
| CR-1 | No dead code — every function/branch reached by a test or docstring example | full `pytest` suite |
| CR-2 | Determinism — same input → bit-identical output; fixed seeds and iteration order | fixed `random_state`, deterministic LightGBM flags, ordered folds; determinism audit script |
| CR-3 | Fail-safe — invalid input (NaN, wrong dtype, too-long gaps) raises, never silently imputed | `clean.py`, `features.py`, `metrics.py`, `folds.py`, `format_submission.py` |
| CR-4 | Minimal CVE attack surface — short deny-list of forbidden dependencies checked against the lockfile | dependency test |

Every metric validates its inputs before computing and raises `ValueError` on empty,
length-mismatched, or non-finite data — guarding the MAPE/SMAPE denominators explicitly
even though load is strictly positive in practice
([metrics.py:16-27](src/ddkast/evaluation/metrics.py)). `build_folds` raises on an empty
or unsorted index, inconsistent forecast-window configuration, or insufficient history at
the earliest origin. This is the same fail-loud philosophy `spotforecast2-safe` enforces
at the estimator level.

## 9. How to Audit

### Static checks and tests

```bash
uv run pytest                        # test suite — every function/branch exercised (CR-1)
uv run ruff check                    # linting
uv run pyright                       # static type checking (strict mode)
uv run pre-commit run --all-files    # ruff + ruff-format + pyright, as enforced pre-commit
uv run ddkast evaluate               # reproduce the evaluation numbers on a synced dataset
```

### Determinism audit (CR-2)

`scripts/audit_determinism.py` is the CR-2 evidence generator: it runs the full pipeline
**twice from clean state** on a shared, once-downloaded data snapshot and diffs every
artifact the two runs produced.

```bash
uv run python scripts/audit_determinism.py                       # last two complete days
uv run python scripts/audit_determinism.py --start 2026-07-14 --end 2026-07-15  # explicit
```

Each artifact is compared by SHA-256 hash, with parquet files additionally checked on
content (values, dtypes, index/column order). The audited claim is that the **forecasts**
reproduce — persisted `.joblib` models and plots are logged but never fail the audit,
since `spotforecast2-safe` stamps `fit_date` and plotly embeds a fresh div id, so a
genuine change surfaces in the predictions instead. Results land in
`audit/<timestamp>/report.{json,md}` with full provenance (git commit, package versions,
window) — the Art. 11/12 record — exiting 0 on pass, 1 on fail so it can gate CI.

## 10. Authors and Contact

Developed by the ddkast team (D. Kochetov, D. Rodriguez and K. Yassen) for the *Data-Driven
Optimization* course challenge (SoSe 2026, AIT). Questions, issues, and audit requests go
through the repository issue tracker at <https://github.com/bachmak/ddkast/issues>. As the
named responsible party for this high-risk-domain artifact, the team is accountable for
the system as described here; full system-level safety validation before any operational
use would remain the deployer's responsibility.
