# ddkast

Energy consumption forecasting for critical infrastructure.

Built for the *Data-Driven Optimization* course challenge (SoSe 2026, AIT).
The immediate target is electricity load forecasting for Germany using ENTSO-E data.
The long-term target is water consumption prediction for pump optimisation — the same pipeline applies because the underlying problem is identical: predict a time series 24 hours ahead so that an operator can schedule resources.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Architecture](#3-architecture)
4. [Project Structure](#4-project-structure)
5. [Technical Stack](#5-technical-stack)
6. [Configuration](#6-configuration)
7. [Pipeline Stages](#7-pipeline-stages)
8. [Models](#8-models)
9. [Evaluation](#9-evaluation)
10. [Testing](#10-testing)
11. [Development Workflow](#11-development-workflow)
12. [Roadmap](#12-roadmap)

---

## 1. Problem Statement

Germany's electricity grid must balance supply and demand in real time.
Accurate 24-hour-ahead load forecasts allow operators to schedule generation, reduce waste, and prevent outages.

Concretely, we want to answer:

> Given the last several weeks of hourly electricity consumption data for Germany, what will consumption look like for every hour of the next 24 hours?

**Baseline**: the simplest possible answer — assume tomorrow looks exactly like the same 24 hours one week ago (7-day seasonal naive forecast).
Our model must beat this baseline on a held-out test set.

**Constraint**: the software must meet the standards expected of critical infrastructure — reproducible results, explicit error handling, auditable logic, and minimal opaque dependencies.

---

## 2. Solution Overview

The solution is a CLI pipeline with five stages that can be run independently:

```
ddkast download → ddkast merge → ddkast train → ddkast predict → ddkast evaluate
```

Each stage reads its input from disk and writes its output to disk.
This means any stage can be re-run without re-running earlier stages (e.g., retrain without re-downloading, evaluate a different model without retraining).

The core forecasting library is `spotforecast2-safe`, chosen specifically because it is designed for safety-critical production environments: deterministic transformations, explicit failure on incomplete data, EU AI Act compliant, and whitebox-auditable.

---

## 3. Architecture

### Why a pipeline of independent stages?

Each stage has a single clear responsibility and a stable interface (read from `DataStore`, write to `DataStore`).
This means:

- Stages can be developed and tested independently.
- Responsibilities can be split across team members without merge conflicts.
- Future stages (e.g., `visualise`, `publish`, `retrain`) can be added without modifying existing code.
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

This separation means the entire pipeline can be exercised in tests without invoking the CLI, and the CLI can be replaced with a web interface without touching any logic.

```
cli.py  (thin: parse args, call pipeline)
  └── pipeline/X.py  (orchestration: call modules in the right order)
        ├── data/          (data access)
        ├── preprocessing/ (cleaning, feature engineering)
        ├── models/        (forecasting, baseline)
        └── evaluation/    (metrics)
```

---

## 4. Project Structure

```
ddkast/
├── src/ddkast/                  # installable package (src layout)
│   ├── __init__.py
│   ├── __main__.py              # enables: python -m ddkast
│   ├── cli.py                   # typer entry point — thin
│   ├── config.py                # pydantic-settings Config + load()
│   ├── pipeline/
│   │   ├── download.py          # stage 1: fetch raw data
│   │   ├── merge.py             # stage 2: clean + combine
│   │   ├── train.py             # stage 3: engineer features + fit model
│   │   ├── predict.py           # stage 4: load model + forecast
│   │   └── evaluate.py          # stage 5: compute metrics + print report
│   ├── data/
│   │   ├── fetch.py             # ENTSO-E API wrapper (entsoe-py)
│   │   └── store.py             # DataStore protocol + ParquetStore
│   ├── preprocessing/
│   │   ├── clean.py             # curation, resampling, outlier detection
│   │   └── features.py          # lag, rolling, Fourier, calendar, holidays
│   ├── models/
│   │   ├── baseline.py          # 7-day seasonal naive (fully implemented)
│   │   └── forecaster.py        # spotforecast2-safe wrapper
│   └── evaluation/
│       └── metrics.py           # MAE, RMSE, MAPE, SMAPE (fully implemented)
│
├── tests/
│   ├── conftest.py              # shared fixtures (Config, sample data)
│   ├── test_baseline.py
│   └── test_metrics.py
│
├── notebooks/                   # exploratory analysis only — never imported
├── data/                        # gitignored — populated by `ddkast download`
│   ├── raw/
│   └── processed/
├── models/                      # gitignored — populated by `ddkast train`
│
├── .github/workflows/ci.yml     # CI: lint, format, type check, test
├── .vscode/
│   ├── extensions.json          # recommended extensions for teammates
│   └── settings.json            # ruff formatter, pyright strict, auto-format
├── .pre-commit-config.yaml      # ruff + pyright run before every commit
│
├── pyproject.toml               # dependencies, tool config (ruff, pyright, pytest)
├── config.toml                  # non-secret settings (country, horizon, paths)
├── .env.example                 # template — copy to .env and add API key
├── QUESTIONS.md                 # open questions for the professor
└── README.md                    # this file
```

### Why `src/` layout?

Without a `src/` directory, running `pytest` from the project root causes Python to add `.` to `sys.path`, which means `import ddkast` silently resolves to the local directory rather than the installed package.
Tests can pass even if the package is broken or uninstallable, because Python never needed to install it.

With `src/ddkast/`, Python cannot find `ddkast` unless it has been installed (via `uv sync`, which performs an editable install automatically).
Tests always run against the installed package — failures are honest.

---

## 5. Technical Stack

| Layer | Library | Why |
|---|---|---|
| CLI | `typer` | Type-annotated commands, auto-generated `--help`, minimal boilerplate |
| Terminal output | `rich` | Progress bars, formatted tables, coloured errors |
| Configuration | `pydantic-settings` | Validates types at startup, loads `.env` and `config.toml`, fails loudly on misconfiguration |
| Data access | `entsoe-py` | Mature wrapper for the ENTSO-E Transparency Platform API; returns pandas DataFrames |
| Data storage | `pyarrow` (Parquet) | Preserves column types including datetimes, compressed, industry standard for tabular time series |
| Core forecasting | `spotforecast2-safe` | Safety-critical design: deterministic, fail-safe on missing data, EU AI Act compliant |
| Regressor | `lightgbm` | Best accuracy/speed trade-off for tabular time-series; used via `spotforecast2-safe` |
| Feature engineering | `holidays`, `numpy` | German public holidays; Fourier cyclical encodings |
| Package manager | `uv` | Fast, generates a lockfile, manages the virtual environment, `uv sync` is the only setup step |
| Testing | `pytest` | Standard |
| Linting + formatting | `ruff` | Replaces flake8 + isort + black in one fast tool |
| Type checking | `pyright` (strict) | Runs in VS Code via Pylance (inline errors) and in CI; strict mode requires full annotations |
| CI | GitHub Actions | Runs on every push: lint → format → type check → test |
| Pre-commit | `pre-commit` + ruff + pyright | Catches issues before they reach the remote |

### Why `spotforecast2-safe` and not `spotforecast2`?

`spotforecast2` is the full-featured version with AutoML, plotting, and weather integration.
`spotforecast2-safe` deliberately removes these features to produce a minimal, auditable codebase.

The difference matters for critical infrastructure:

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

## 6. Configuration

Configuration has three layers, in increasing priority:

1. **Defaults** — hard-coded in `Config` (e.g., `horizon = 24`)
2. **`config.toml`** — non-secret, version-controlled settings
3. **`.env`** — secrets (API key), gitignored

`cli.py` accepts `--config path/to/config.toml` to override the config file location.
All other settings can be overridden via environment variables (pydantic-settings convention: `COUNTRY_CODE=AT ddkast train`).

### `config.toml` reference

```toml
country_code = "DE_LU"  # ENTSO-E bidding zone code
horizon      = 24       # hours ahead to forecast
resolution   = "1H"     # temporal resolution ("1H", "30min", etc.)
data_dir     = "data"   # root for raw/ and processed/ subdirectories
models_dir   = "models" # where trained models are persisted
```

### `.env` reference

```
ENTSOE_API_KEY=your_security_token_here
```

Get your token at `https://transparency.entsoe.eu` → My Account → Security Token.

---

## 7. Pipeline Stages

### `ddkast download`

Calls `data/fetch.py` which wraps `entsoe-py` to query the ENTSO-E Transparency Platform for actual total load data (the `ActualTotalLoad` series) for the configured country and time range.
Raw data is written to `data/raw/` via `ParquetStore`.

ENTSO-E data for Germany is approximately 8,760 rows per year (one per hour).
The download is incremental: only missing date ranges need to be fetched.

### `ddkast merge`

Reads all raw Parquet files, calls `preprocessing/clean.py` to:
- Resample to the configured resolution
- Detect and handle outliers (using `spotforecast2-safe`'s validated routines)
- Reject windows with too many missing values (fail-safe: no silent imputation for large gaps)

The result is a single, clean, continuous time series written to `data/processed/`.

### `ddkast train`

Reads the processed dataset, calls `preprocessing/features.py` to add:
- **Lag features**: load 1h, 24h, and 168h (1 week) ago
- **Rolling statistics**: 24h and 168h rolling mean and standard deviation
- **Fourier cyclical encodings**: sine/cosine pairs for daily, weekly, and annual seasonality
- **Calendar features**: hour of day, day of week, month, week of year
- **Holiday indicators**: German public holidays via the `holidays` library

Then fits a `spotforecast2-safe` recursive forecaster with a LightGBM regressor using default hyperparameters (tuning comes in a later milestone).
The trained model is persisted via `spotforecast2-safe`'s built-in persistence (`_save_forecasters`) to `models/`.

### `ddkast predict`

Loads the most recently trained model, constructs the feature matrix for the forecast window, and generates predictions for the next `horizon` hours.
Output is written to `data/processed/forecast.parquet`.

### `ddkast evaluate`

Loads the forecast and the ground truth from `data/processed/`, computes all four metrics, and prints a formatted comparison table:

```
┌─────────┬───────────┬────────────┐
│ Metric  │ Baseline  │ Model      │
├─────────┼───────────┼────────────┤
│ MAE     │ 1 842 MW  │   943 MW   │
│ RMSE    │ 2 310 MW  │ 1 201 MW   │
│ MAPE    │   4.2 %   │   2.1 %    │
│ SMAPE   │   4.1 %   │   2.0 %    │
└─────────┴───────────┴────────────┘
```

The **baseline** is always the 7-day seasonal naive forecast (predict = same hour last week).
All model results are reported relative to this baseline.

---

## 8. Models

### Baseline: 7-day seasonal naive (`models/baseline.py`)

```
prediction[t] = actual[t - 168h]
```

This is the simplest meaningful benchmark for hourly energy data.
Electricity consumption has a strong weekly pattern (workdays vs weekends, morning vs evening peaks), so the same hour last week is a surprisingly competitive predictor.
Any model that cannot beat it consistently is not useful.

### Forecasting model: recursive multi-step (`models/forecaster.py`)

The model uses `spotforecast2-safe`'s `ForecasterRecursive` strategy:
predictions are made one step at a time, and each prediction is fed back as an input lag feature for the next step.
The underlying regressor is LightGBM.

This approach is well-suited to the 24-step (24-hour) horizon because:
- It leverages the rich feature set described above
- LightGBM handles non-linear interactions between lag, calendar, and holiday features naturally
- The recursive approach is supported directly by `spotforecast2-safe` with safety guarantees

---

## 9. Evaluation

### Metrics (`evaluation/metrics.py`)

All four metrics are computed for both the model and the baseline on every evaluation run.

| Metric | Formula | Primary use |
|---|---|---|
| **MAE** | mean(|actual − predicted|) | **Primary metric** — interpretable in MW, matches course standard |
| **RMSE** | sqrt(mean((actual − predicted)²)) | Penalises large errors more heavily; useful for catching spikes |
| **MAPE** | mean(|actual − predicted| / actual) × 100 | Scale-independent; easy to communicate ("X% off") |
| **SMAPE** | mean(2|actual − predicted| / (|actual| + |predicted|)) × 100 | Symmetric version of MAPE; avoids asymmetry for over/under-forecasting |

MAE is the primary metric because it is directly interpretable (error in megawatts), matches the metric used in the reference project, and is likely the metric the professor uses for team comparison.

### Walk-forward validation

The model is not evaluated on a single train/test split.
Instead, walk-forward (time-series cross-validation) is used:
- Train on all data up to time T
- Predict hours T+1 to T+24
- Advance T by 24 hours and repeat

This mirrors real deployment: the model is always trained on past data and evaluated on future data.
Random splits are invalid for time series because they leak future information into training.

---

## 10. Testing

Tests live in `tests/` and are run with `pytest`.

```bash
uv run pytest                          # run all tests
uv run pytest --cov=ddkast             # with coverage
uv run pytest tests/test_metrics.py   # single file
```

### What is tested

- **`test_baseline.py`**: correctness of the 7-day naive forecast (length, exact values)
- **`test_metrics.py`**: correctness of all four metric functions (zero error on perfect forecast, known values)

### What is not mocked

Tests use real fixture data (a `pd.Series` of 14 days of synthetic load values) rather than mocking pandas or the ENTSO-E API.
The ENTSO-E integration test uses a committed sample of real downloaded data so CI can run without a live API key.

Mocking the data layer risks the mock drifting from reality — a historical failure mode where tests pass but production breaks on real data shapes.

### Fixtures (`tests/conftest.py`)

- `config` — a `Config` instance with a dummy API key and `tmp_path` for data/model directories; safe to use in any test
- `load_series` — 14 days × 24 hours of realistic synthetic load values with a UTC datetime index

---

## 11. Development Workflow

### First-time setup

**Step 1 — Install uv** (skip if already installed)

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy BypassScope -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Steps 2–5** (same on all platforms — use Terminal on macOS/Linux, PowerShell on Windows):

```bash
# Clone the repo and enter it
git clone <repo-url>
cd ddkast

# Install all dependencies and create the virtual environment
uv sync --group dev

# Activate the pre-commit hooks
uv run pre-commit install

# Set up your ENTSO-E API key
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
# Open .env and replace the placeholder with your token
```

VS Code will prompt to install recommended extensions the first time you open the project.
Select the `.venv` interpreter when prompted — it will be detected automatically on all platforms.

### Running the pipeline

```bash
uv run ddkast download
uv run ddkast merge
uv run ddkast train
uv run ddkast predict
uv run ddkast evaluate
```

Or equivalently:

```bash
python -m ddkast download
```

Override the config file:

```bash
uv run ddkast train --config custom.toml
```

Override a single setting (use the appropriate syntax for your shell):

```bash
# macOS / Linux
HORIZON=48 uv run ddkast predict

# Windows PowerShell
$env:HORIZON = "48"; uv run ddkast predict
```

### Code quality

Pre-commit hooks run automatically on `git commit` and check:
1. **ruff** — linting and import sorting (with auto-fix)
2. **ruff format** — formatting
3. **pyright** — strict type checking

To run them manually:

```bash
uv run ruff check src tests --fix
uv run ruff format src tests
uv run pyright
```

### CI

Every push triggers the GitHub Actions workflow (`.github/workflows/ci.yml`), which runs the same four checks: lint, format, type check, test.
CI must pass before a branch is merged.

---

## 12. Roadmap

### Milestone 1 — May 12 (1st Interim Presentation)

- [x] Project scaffold, CI, pre-commit
- [x] `DataStore` abstraction (`ParquetStore`)
- [x] `Config` with pydantic-settings
- [x] Baseline model
- [x] Evaluation metrics
- [ ] `data/fetch.py` — ENTSO-E download
- [ ] `preprocessing/clean.py` — curation with `spotforecast2-safe`
- [ ] `pipeline/download.py` + `pipeline/merge.py`
- [ ] `pipeline/train.py` — fit with default hyperparameters
- [ ] `pipeline/predict.py` + `pipeline/evaluate.py`
- [ ] Working end-to-end demo beating the baseline

### Milestone 2 — June 23 (2nd Interim Presentation)

- [ ] `preprocessing/features.py` — full feature set (lags, rolling, Fourier, calendar, holidays)
- [ ] Hyperparameter optimisation via `spotforecast2` (SpotOptim/Bayesian search)
- [ ] SHAP explainability (feature importance)
- [ ] Walk-forward validation
- [ ] Periodogram analysis for seasonality detection (notebook)

### Milestone 3 — July 21 (Final Presentation)

- [ ] Prediction intervals / uncertainty quantification
- [ ] Automated retraining (GitHub Actions schedule)
- [ ] Optional: visualisation stage (`ddkast visualise`)
- [ ] Model card documentation (EU AI Act)
