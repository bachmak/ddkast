# ddkast

Energy consumption forecasting for critical infrastructure.

Built for the *Data-Driven Optimization* course challenge (SoSe 2026, AIT).
The immediate target is electricity load forecasting for Germany using ENTSO-E data.
The long-term target is water consumption prediction for pump optimisation — the same pipeline applies because the underlying problem is identical: predict a time series 24 hours ahead so that an operator can schedule resources.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Models](#models)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Roadmap](#roadmap)

---

## Quick Start

### Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — the only tool you need to install manually. Everything else (Python, dependencies, virtual environment) is managed by uv.
- **VS Code** with the recommended extensions (you will be prompted on first open).

### One-time setup

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd ddkast

# 2. Install all dependencies and create the virtual environment
uv sync --group dev

# 3. Install pre-commit hooks (runs ruff + pyright before every commit)
uv run pre-commit install

# 4. Create your .env file
cp .env.example .env
```

If you have an **ENTSO-E API key**, open `.env` and replace the placeholder.
If you don't have one yet, set a dummy value for now:

```
ENTSOE_API_KEY=dummy
```

> Get a real key at [transparency.entsoe.eu](https://transparency.entsoe.eu) → My Account → Security Token.

### Verify the setup

```bash
uv run pytest   # all 41 tests should pass — no API key required
```

### Run the pipeline

**Without an API key** — uses synthetic data, full pipeline runs immediately:

```bash
uv run python scripts/seed_synthetic.py   # seeds load, DAF, and weather to data/raw/
uv run ddkast merge
uv run ddkast train
uv run ddkast predict
uv run ddkast evaluate
uv run ddkast visualise
```

**With a real API key** — downloads actual German load and weather data:

```bash
uv run ddkast download          # Jan 2024 – Apr 2026 (ENTSO-E + Open-Meteo)
uv run ddkast merge
uv run ddkast train
uv run ddkast predict
uv run ddkast evaluate
uv run ddkast visualise
```

To forecast a specific future date (live path, requires up-to-date load data):

```bash
uv run ddkast download --end 2026-05-24   # refresh load + weather to yesterday
uv run ddkast merge
uv run ddkast train
uv run ddkast predict --target-date 2026-05-25
```

Both paths produce the same evaluation table at the end:

```
          Evaluation  (2026-03-31 → 2026-03-31)
┏━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric ┃ 7-day Naive ┃ ENTSO-E DAF ┃    Model ┃
┡━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ MAE    │    739.9 MW │    512.3 MW │ 538.2 MW │
│ RMSE   │    926.2 MW │    627.3 MW │ 668.3 MW │
│ MAPE   │       1.4 % │       1.0 % │    1.1 % │
│ SMAPE  │       1.4 % │       1.0 % │    1.0 % │
└────────┴─────────────┴─────────────┴──────────┘
```

### VS Code

1. Open the project folder — VS Code will prompt you to install recommended extensions.
2. Select the Python interpreter: **Python: Select Interpreter** → choose `.venv`.
3. Open the **Run and Debug** panel (`Ctrl+Shift+D`).
4. Pick a configuration from the dropdown (**seed synthetic data**, **download**, **merge**, **train**, **predict**, **evaluate**, **visualise**, or **pipeline (all stages)**) and press `F5`.

Full breakpoint and step-through debugging works in all configurations, including inside library code.

### Code quality

Pre-commit hooks run automatically on `git commit`. To run them manually:

```bash
uv run ruff check src tests --fix   # lint + auto-fix
uv run ruff format src tests        # format
uv run pyright                      # strict type check
uv run pytest                       # tests
```

Every push to `main` and every pull request triggers the GitHub Actions CI workflow, which runs the same four checks in parallel. CI must pass before a branch is merged.

---

## Problem Statement

Germany's electricity grid must balance supply and demand in real time.
Accurate 24-hour-ahead load forecasts allow operators to schedule generation, reduce waste, and prevent outages.

Concretely, we want to answer:

> Given the last several weeks of hourly electricity consumption data for Germany, what will consumption look like for every hour of the next 24 hours?

**Baseline**: the simplest possible answer — assume tomorrow looks exactly like the same 24 hours one week ago (7-day seasonal naive forecast).
Our model must beat this baseline on a held-out test set.

**Constraint**: the software must meet the standards expected of critical infrastructure — reproducible results, explicit error handling, auditable logic, and minimal opaque dependencies.

---

## Solution Overview

The solution is a CLI pipeline with six stages that can be run independently:

```
ddkast download → ddkast merge → ddkast train → ddkast predict → ddkast evaluate → ddkast visualise
```

Each stage reads its input from disk and writes its output to disk.
This means any stage can be re-run without re-running earlier stages (e.g., retrain without re-downloading, evaluate a different model without retraining).

The core forecasting library is `spotforecast2-safe`, chosen specifically because it is designed for safety-critical production environments: deterministic transformations, explicit failure on incomplete data, EU AI Act compliant, and whitebox-auditable.

---

## Architecture

### Why a pipeline of independent stages?

Each stage has a single clear responsibility and a stable interface (read from `DataStore`, write to `DataStore`).
This means:

- Stages can be developed and tested independently.
- Responsibilities can be split across team members without merge conflicts.
- Future stages (e.g., `publish`, `retrain`) can be added without modifying existing code.
- A future web frontend or scheduler can call the same `pipeline.X.run(config)` functions directly — no changes needed.

### The `DataStore` abstraction

All data is read and written through a `DataStore` protocol:

```python
class DataStore(Protocol):
    def read(self, name: str) -> pd.DataFrame: ...
    def write(self, name: str, df: pd.DataFrame) -> None: ...
```

The concrete implementation today is `ParquetStore` (Parquet files on local disk).
If the project later needs to store data in S3, a database, or CSV files, only the `ParquetStore` class needs to change — not a single line of pipeline or model code.

### The `Config` object

All configuration lives in a single validated `Config` object loaded at startup from `config.toml` (non-secret settings) and `.env` (secrets such as the API key).
Every pipeline function receives `Config` as an explicit argument — it is never a global variable.

This makes functions easy to test (pass a `Config` with test values), easy to reason about (dependencies are visible in the signature), and safe to run in parallel with different configurations.

### The `pipeline/` layer

`cli.py` is deliberately thin: it only parses CLI arguments and calls `pipeline.X.run(config)`.
Business logic lives in `pipeline/`, which in turn calls `data/`, `preprocessing/`, `models/`, and `evaluation/`.

```
cli.py  (thin: parse args, call pipeline)
  └── pipeline/X.py  (orchestration: call modules in the right order)
        ├── data/          (data access)
        ├── preprocessing/ (cleaning, feature engineering)
        ├── models/        (forecasting, baseline)
        └── evaluation/    (metrics)
```

---

## Technical Stack

| Layer | Library | Why |
|---|---|---|
| CLI | `typer` | Type-annotated commands, auto-generated `--help`, minimal boilerplate |
| Terminal output | `rich` | Progress bars, formatted tables, coloured errors |
| Configuration | `pydantic-settings` | Validates types at startup, loads `.env` and `config.toml`, fails loudly on misconfiguration |
| Data access | `entsoe-py` | Mature wrapper for the ENTSO-E Transparency Platform API; returns pandas DataFrames |
| Weather data | `spotforecast2-safe` `WeatherService` | Open-Meteo archive (historical) and forecast (prospective) endpoints; UTC-aware, fail-safe on gaps |
| Data storage | `pyarrow` (Parquet) | Preserves column types including datetimes, compressed, industry standard for tabular time series |
| Core forecasting | `spotforecast2-safe` | Safety-critical design: deterministic, fail-safe on missing data, EU AI Act compliant |
| Regressor | `lightgbm` | Best accuracy/speed trade-off for tabular time-series; used via `spotforecast2-safe` |
| Feature engineering | `holidays`, `numpy` | German public holidays; RBF cyclical encodings and weather join via `spotforecast2-safe` |
| Package manager | `uv` | Fast, generates a lockfile, manages the virtual environment, `uv sync` is the only setup step |
| Testing | `pytest` | Standard |
| Linting + formatting | `ruff` | Replaces flake8 + isort + black in one fast tool |
| Type checking | `pyright` (strict) | Runs in VS Code via Pylance (inline errors) and in CI; strict mode requires full annotations |
| CI | GitHub Actions | Runs on pushes to `main` and all PRs; three parallel jobs: lint + format, type check, test (≥ 80% coverage required) |
| Pre-commit | `pre-commit` + ruff + pyright | Catches issues before they reach the remote |

### Why `spotforecast2-safe` and not `spotforecast2`?

`spotforecast2` is the full-featured version with AutoML, plotting, and weather integration.
`spotforecast2-safe` deliberately removes these features to produce a minimal, auditable codebase.

| Property | spotforecast2 | spotforecast2-safe |
|---|---|---|
| Missing data | Silently imputes | Explicitly rejects (fail-safe) |
| Reproducibility | Standard | Guaranteed deterministic |
| External dependencies | Many | Minimal (primarily scikit-learn) |
| EU AI Act | No | Compliant |
| Attack surface | Larger | Reduced |

Since the long-term application is water pump scheduling — an operational system where a wrong prediction has real consequences — `spotforecast2-safe`'s design constraints are features, not limitations.

`spotforecast2` is used in development only, for its richer hyperparameter optimisation tools (SpotOptim, Bayesian search). It is not on the production code path.

---

## Configuration

Configuration has three layers, in increasing priority:

1. **Defaults** — hard-coded in `Config` (e.g., `horizon = 24`)
2. **`config.toml`** — non-secret, version-controlled settings
3. **`.env`** — secrets (API key), gitignored

`cli.py` accepts `--config path/to/config.toml` to override the config file location.
All other settings can be overridden via environment variables (pydantic-settings convention: `COUNTRY_CODE=AT ddkast train`).

### `config.toml` reference

```toml
# --- challenge ---
team_id = "your_team_id"  # alphanumeric + underscore; used in submission CSV path

# --- shared ---
country_code = "DE_LU"  # ENTSO-E bidding zone
horizon      = 24       # hours ahead to forecast
resolution   = "1h"     # temporal resolution
data_dir     = "data"
models_dir   = "models"

# --- download ---
download_start = 2022-01-01
download_end   = 2026-04-30

# --- merge / cleaning ---
outlier_iqr_multiplier  = 3.0   # IQR multiplier for outlier detection
max_interpolation_hours = 3     # max consecutive missing hours to interpolate

# --- features ---
holiday_country_code = "DE"     # ISO 2-letter code for holiday calendar
rbf_periods_hour     = 10       # RBF basis functions for daily cycle
rbf_periods_dow      = 7        # RBF basis functions for weekly cycle
rbf_periods_month    = 6        # RBF basis functions for annual cycle

# --- train ---
lags      = 168   # autoregressive lags (1 full week of hourly data)
test_days = 30    # days held out for evaluation

# --- weather ---
weather_latitude  = 50.110924   # Open-Meteo fetch location (default: Frankfurt am Main)
weather_longitude = 8.682127
weather_cache_dir = "data/cache"  # local cache for Open-Meteo responses

# --- visualise ---
backend       = "plotly"      # "plotly" (interactive HTML) or "matplotlib" (static PDF)
plots         = ["forecast", "daf", "residuals"]  # which data series to include
plots_dir     = "plots"       # output directory for plot files
figure_format = "pdf"         # matplotlib output format: "pdf" or "png"

# --- inter-stage filenames (keys used by ParquetStore) ---
raw_load_actual          = "load_actual"
raw_load_forecast        = "load_forecast"
raw_weather              = "weather_raw"       # written by download, read by merge
processed_load           = "load_clean"
processed_entso_forecast = "forecast_entso"
processed_weather        = "weather_processed" # written by merge, read by train and predict
processed_exog           = "exog_full"         # 40-col exog matrix, written by train
processed_test           = "load_test"
processed_predictions    = "load_predicted"
evaluation_series        = "evaluation_series"  # written by evaluate, read by visualise
model_target             = "load_mw"
```

### `.env` reference

```
ENTSOE_API_KEY=your_security_token_here
```

Get your token at `https://transparency.entsoe.eu` → My Account → Security Token.

> `team_id` can also be set via the `TEAM_ID` environment variable instead of `config.toml`.

---

## Pipeline Stages

### `ddkast download`

Fetches three datasets in a single pass:

1. **Actual total load** (`ActualTotalLoad`) — the ground truth time series, from ENTSO-E
2. **Day-ahead load forecast** (`DayAheadTotalLoadForecast`) — the professional benchmark, from ENTSO-E
3. **Weather archive** (Open-Meteo) — 15 meteorological variables at hourly resolution for the configured location (default: Frankfurt am Main). Uses the reanalysis archive endpoint; responses are cached locally under `weather_cache_dir` to avoid repeat API calls.

All three are written to `data/raw/` via `ParquetStore`.
The date range (`download_start` → `download_end`) is configurable; the default covers Jan 2022 – Apr 2026.

### `ddkast merge`

Reads all three raw artifacts and produces three processed outputs:

**Actual load** — cleaned via `preprocessing/clean.py`:
1. Normalise to UTC (handles DST transitions in ENTSO-E local-time data)
2. Resample to configured resolution (averages sub-hourly data, collapses duplicate timestamps)
3. IQR-based outlier detection (`multiplier` configurable) — outliers replaced with NaN
4. Linear interpolation for gaps up to `max_interpolation_hours`
5. **Fail-safe**: raises `ValueError` if any NaN remains after interpolation

**ENTSO-E DAF** — resampled from 15-min to hourly resolution, then interpolated for short gaps. If NaN remain after interpolation (gap too long), a warning is logged and the affected rows are dropped — the DAF is best-effort and its gaps do not invalidate the load data.

**Weather** — resampled to 1h and converted to UTC.

**Trim logic:** load and DAF are trimmed to their mutual overlap. Weather is trimmed separately within the load range. This is necessary because the Open-Meteo reanalysis archive has a publication lag of a few days — the tail of the load series will often have no weather coverage yet, and that is normal.

### `ddkast train`

1. Reads the processed load and processed weather
2. Splits into train / test at `cutoff = last_timestamp − test_days`
3. Builds a 40-column exog matrix (25 calendar + 15 weather) over the full load range and writes it to `data/processed/exog_full.parquet` for reuse by `predict`
4. **Clips `train_df` to the exog coverage window** — weather may end a few days before the load series (Open-Meteo publication lag), so this step prevents a KeyError when aligning train labels with exog rows
5. Fits a `ForecasterRecursive` with `LGBMRegressor` using `lags=168` contiguous autoregressive lags
6. Persists the model to `models/forecaster_load_mw.joblib`
7. Writes the test split to `data/processed/` for use by `evaluate`

### `ddkast predict`

The stage operates in two modes depending on whether `--target-date` is supplied.

**Default path** (no `--target-date`):
1. Reads the processed load and the pre-built `exog_full` matrix from DataStore (written by `train`)
2. Uses the last `lags` hours of load before the train/test cutoff as the autoregressive window
3. Slices `exog_full` to the target `horizon` timestamps and generates a recursive forecast
4. Writes predictions to `data/processed/`

**Live path** (`--target-date YYYY-MM-DD`):
1. Checks that load data extends to `target_date − 1 day 23:00` (continuity guard — raises a helpful error with the required `download` command if not)
2. Fetches a weather forecast from Open-Meteo for the target 24-hour window (covers up to 7 days ahead)
3. Builds a 40-column exog matrix for the target window from the live weather forecast
4. Generates the recursive forecast using the last `lags` hours of load as the autoregressive window
5. Writes predictions to `data/processed/`
6. **Writes a submission CSV** to `data/submissions/<team_id>/<YYYY-MM-DD>.csv` in the challenge format (`timestamp_utc`, `forecast_mw`, 24 rows)

### `ddkast evaluate`

Loads the model forecast, the actual values, and the ENTSO-E DAF; aligns all three to the forecast timestamps; prints a comparison table; and writes the aligned series (actual, forecast, ENTSO-E DAF, residuals) to `data/processed/evaluation_series.parquet` for use by `visualise`.

```
                  Evaluation  (2026-03-31 → 2026-03-31)
 ┌─────────┬─────────────┬──────────────┬──────────────┐
 │ Metric  │ 7-day Naive │  ENTSO-E DAF │     Model    │
 ├─────────┼─────────────┼──────────────┼──────────────┤
 │ MAE     │  1 842.0 MW │   1 100.0 MW │    943.0 MW  │
 │ RMSE    │  2 310.0 MW │   1 420.0 MW │  1 201.0 MW  │
 │ MAPE    │      4.2 %  │       2.5 %  │      2.1 %   │
 │ SMAPE   │      4.1 %  │       2.4 %  │      2.0 %   │
 └─────────┴─────────────┴──────────────┴──────────────┘
```

### `ddkast visualise`

Reads the `evaluation_series` written by `evaluate` and renders the results using the configured backend.

**Plotly backend** (`--backend plotly`, default): a single interactive HTML figure with two linked subplots sharing an x-axis and a range slider. All traces are toggleable via the legend.

- Top panel: actual load, model forecast, ENTSO-E DAF
- Bottom panel: model residuals and DAF residuals, with a zero reference line

**Matplotlib backend** (`--backend matplotlib`): a static two-panel PDF suitable for reports.

- Top panel: actual load vs. model forecast
- Bottom panel: model residuals with a zero reference line

Both backends print a clickable terminal link to the output file. Additional CLI options:

```bash
ddkast visualise --backend matplotlib        # override backend
ddkast visualise --from 2026-03-01           # zoom to a date window
ddkast visualise --to   2026-04-01
ddkast visualise --plots forecast --plots residuals  # subset of series
```

Output files are written to `plots/` by default (`plots_dir` in `config.toml`).

---

## Models

### Baseline: 7-day seasonal naive (`models/baseline.py`)

```
prediction[t] = actual[t - 168h]
```

This is the simplest meaningful benchmark for hourly energy data.
Electricity consumption has a strong weekly pattern (workdays vs weekends, morning vs evening peaks), so the same hour last week is a surprisingly competitive predictor.
Any model that cannot beat it consistently is not useful.

### Forecasting model: recursive multi-step (`models/forecaster.py`)

The model uses `spotforecast2-safe`'s `ForecasterRecursive` directly (the low-level class, not the high-level manager wrapper) with a `LGBMRegressor` estimator.

**Autoregressive lags:** 168 contiguous lags (hours 1–168 back), capturing the full weekly cycle. LightGBM's built-in feature selection identifies which lags are actually predictive. The lag count is configurable; SHAP-driven refinement and long-range named lags (`[336, 720, 8760]`) are planned for Milestone 2.

**Exogenous features** (built by `preprocessing/features.py`): 40 columns total.

| Feature group | Columns | Description |
|---|---|---|
| Daily RBF | 10 | Gaussian basis functions spread over hours 0–23 |
| Weekly RBF | 7 | Basis functions over days 0–6 |
| Annual RBF | 6 | Basis functions over months 1–12 |
| Holiday indicator | 1 | 1 if the hour falls on a German public holiday |
| Weekend flag | 1 | 1 if Saturday or Sunday |
| Weather | 15 | `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `snowfall`, `weather_code`, `pressure_msl`, `surface_pressure`, `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m` |

RBF (Repeating Basis Functions) encoding places Gaussian bumps evenly around each cycle so that the model sees smooth, wrapped representations of "where in the day/week/year are we?" — avoiding the discontinuity that integer encodings create at cycle boundaries.

The calendar features (25 cols) are built for any timestamp range. The weather features (15 cols) are joined from either the archived reanalysis data (training and default predict) or a live Open-Meteo forecast (live predict path).
A Fourier (sin/cos) comparison is planned for Milestone 2.

**Persistence:** models are saved to `models/forecaster_load_mw.joblib` via joblib (following `spotforecast2-safe` conventions).

---

## Evaluation

### Metrics (`evaluation/metrics.py`)

All four metrics are computed for the model, the 7-day naive baseline, and the ENTSO-E DAF on every evaluation run.

| Metric | Formula | Primary use |
|---|---|---|
| **MAE** | mean(|actual − predicted|) | **Primary metric** — interpretable in MW, matches course standard |
| **RMSE** | sqrt(mean((actual − predicted)²)) | Penalises large errors more heavily; useful for catching spikes |
| **MAPE** | mean(|actual − predicted| / actual) × 100 | Scale-independent; easy to communicate ("X% off") |
| **SMAPE** | mean(2|actual − predicted| / (|actual| + |predicted|)) × 100 | Symmetric version of MAPE; avoids asymmetry for over/under-forecasting |

MAE is the primary metric because it is directly interpretable (error in megawatts), matches the metric used in the reference project, and is likely the metric the professor uses for team comparison.

### Benchmarks

Two benchmarks are computed on every evaluation run:

| Benchmark | Description |
|---|---|
| **7-day naive** | `prediction[t] = actual[t − 168h]` — the minimum bar any model must beat |
| **ENTSO-E day-ahead** | The official day-ahead load forecast published by ENTSO-E — a strong external benchmark reflecting what professional forecasters achieve |

Beating the ENTSO-E forecast consistently would be a meaningful result; matching it is already competitive.

### Validation

**Milestone 1:** single temporal split — train on all data before the last `test_days` (default 30), evaluate on those last days. Produces the comparison table from one pass through the pipeline.

**Milestone 2:** full walk-forward (time-series cross-validation) — train up to time T, predict T+1…T+24, advance T by 24h, repeat. This mirrors real deployment and gives statistically robust error estimates.

Random splits are never used — they leak future information into training, which is invalid for time series.

---

## Testing

Tests live in `tests/` and are run with `pytest`.

```bash
uv run pytest                          # run all tests
uv run pytest --cov=ddkast             # with coverage
uv run pytest tests/test_metrics.py   # single file
```

### What is tested

| File | Coverage |
|---|---|
| `test_baseline.py` | 7-day naive forecast correctness |
| `test_metrics.py` | All four metric functions |
| `test_clean.py` | Resampling, outlier removal, gap interpolation, fail-safe rejection |
| `test_features.py` | ExogBuilder output shape, column names, holiday/weekend flags |
| `test_forecaster.py` | fit/forecast roundtrip, output length, index alignment, missing-model error |
| `test_weather.py` | `fetch_weather` with mocked `WeatherService`; both archive and forecast paths; gap error propagation |
| `test_merge.py` | DAF resampling to hourly, separate load/DAF and weather trim logic, no NaN after merge |
| `test_predict_live.py` | Live predict path: continuity guard, 24-row output, submission CSV written to correct path |
| `test_pipeline_e2e.py` | Full train → predict → evaluate on synthetic data (no API key needed) |

### What is not mocked

Tests use real fixture data (synthetic `pd.DataFrame` values) rather than mocking pandas or the ENTSO-E API.
The end-to-end test writes directly to `ParquetStore` and runs all three pipeline stages, giving confidence that the stages wire together correctly.

Mocking the data layer risks the mock drifting from reality — a historical failure mode where tests pass but production breaks on real data shapes.

---

## Roadmap

### Milestone 1 — May 12 (1st Interim Presentation)

- [x] Project scaffold, CI, pre-commit
- [x] `DataStore` abstraction (`ParquetStore`)
- [x] `Config` with pydantic-settings (all pipeline parameters configurable)
- [x] Baseline model (7-day seasonal naive)
- [x] Evaluation metrics (MAE, RMSE, MAPE, SMAPE)
- [x] `data/fetch.py` — actual load + ENTSO-E DAF download
- [x] `preprocessing/clean.py` — resample, IQR outliers, gap interpolation, fail-safe
- [x] `preprocessing/features.py` — `ExogBuilder` with RBF cyclical encoding
- [x] `pipeline/download.py` + `pipeline/merge.py`
- [x] `pipeline/train.py` — temporal split + fit with `lags=168`
- [x] `pipeline/predict.py` + `pipeline/evaluate.py` — 3-column comparison table
- [x] End-to-end test suite (27 tests, no API key required)
- [x] VS Code debug launch configurations for all stages
- [x] `scripts/seed_synthetic.py` — full pipeline runnable without API key
- [x] `pipeline/visualise.py` — Plotly (interactive) and Matplotlib (static) backends; clickable terminal links

### Milestone 2 — June 23 (2nd Interim Presentation)

- [ ] Full walk-forward (time-series cross-validation) replacing single split
- [ ] SHAP explainability — feature importance, lag pruning
- [ ] Long-range named lags `[336, 720, 8760]` evaluated via SHAP
- [ ] Rolling statistics (24h/168h mean and std) as additional exogenous features
- [ ] Fourier (sin/cos) vs. RBF cyclical encoding comparison
- [ ] Hyperparameter optimisation via `spotforecast2` (SpotOptim/Bayesian search)
- [ ] Periodogram analysis for seasonality detection (notebook)

### Milestone 3 — July 21 (Final Presentation)

- [ ] Prediction intervals / uncertainty quantification
- [ ] Automated retraining (GitHub Actions schedule)
- [ ] Model card documentation (EU AI Act)
