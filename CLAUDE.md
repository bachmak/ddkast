# ddkast — Claude Context

Energy consumption forecasting CLI for the DDMO course challenge (SoSe 2026, AIT). Worth 50% of the course grade.
Immediate target: electricity load forecasting for Germany (ENTSO-E, DE-LU bidding zone).
Long-term target: water consumption prediction for pump optimisation.

Full documentation: see `README.md`.
Open questions for the professor: see `QUESTIONS.md`.

## Architecture

- `src/` layout
- 6-stage CLI pipeline: `download → merge → train → predict → evaluate → visualise`
- Each stage lives in `pipeline/<stage>.py` and exposes a `run(config)` entry point
- `DataStore` protocol (`src/ddkast/data/store.py`) with `ParquetStore` as the concrete implementation
- `Config` is passed explicitly as an argument everywhere — never a global or singleton
- `cli.py` stays thin: argument parsing + `pipeline.X.run(config)` calls only

## Critical constraints

- `spotforecast2-safe` is the **only** library allowed on the production code path. It is EU AI Act compliant and deterministic. `spotforecast2` may be used for development/experimentation only.
- All data access goes through the `DataStore` protocol. Never read/write Parquet files directly in pipeline or model code.
- `cli.py` must contain no business logic.

## Stack

- **Runtime:** Python 3.13+, `uv`
- **CLI/config:** `typer`, `rich`, `pydantic-settings`
- **Data:** `entsoe-py`, `pyarrow`, Parquet via `ParquetStore`
- **Model:** `LightGBM`
- **Visualisation:** `plotly` (interactive HTML), `matplotlib` (static PDF)
- **Quality:** `pyright` strict, `ruff`, `pytest`, pre-commit hooks (ruff + pyright on every commit)
- **CI:** GitHub Actions

## Key decisions already made

- Parquet storage decoupled behind `DataStore` protocol
- MAE is the primary evaluation metric; report all four (MAE, RMSE, MAPE, SMAPE)
- Two benchmarks in `evaluate`: 7-day seasonal naive and the ENTSO-E published day-ahead forecast

### Data

- Download range: Jan 2022 – Apr 2026
- Both actual load and ENTSO-E day-ahead forecast (DAF) are fetched in a single `download` pass and stored separately
- Cleaning: IQR multiplier 3.0, max 3-hour linear interpolation; gaps longer than that cause the surrounding window to be rejected (fail-safe). Both values are configurable.
- All inter-stage filenames (e.g. `raw_load_actual`, `processed_load`) are fields in `Config` — nothing hardcoded in pipeline or model code

### Features

- Cyclical encoding: `ExogBuilder` with RBF (`RepeatingBasisFunction`) from `spotforecast2-safe`. Fourier (sin/cos) is the planned M2 comparison, not the M1 default.
- Rolling statistics (24h/168h mean and std) deferred to M2. With 168 lags, LightGBM captures this structure implicitly.
- Calendar features (hour, day-of-week, month) and German public holidays via `ExogBuilder`. RBF basis function counts are configurable.

### spotforecast2-safe API findings

- `mark_outliers` / `get_outliers` use **IsolationForest**, not IQR. Use `manual_outlier_removal` with IQR-computed bounds (`Q1 - k*IQR`, `Q3 + k*IQR`) for deterministic, interpretable outlier handling.
- `LinearlyInterpolateTS` has no `limit` parameter — use pandas `interpolate(method="linear", limit=N)` for gap-limited interpolation, then check for remaining NaN as the fail-safe step.
- `remove_duplicate_timestamps` operates on a DataFrame where the timestamp is a **column** (named `"Time (UTC)"`), not a DatetimeIndex. Use `agg_and_resample_data` instead — it handles deduplication naturally via resampling.

### Model

- Use `ForecasterRecursive` directly (low-level class from `spotforecast2-safe`), **not** `ForecasterRecursiveModel`. The high-level wrapper does its own data loading that bypasses the `DataStore` abstraction.
- Lags: `lags=168` contiguous (1 week) for M1. SHAP-driven refinement in M2 — long-range named lags (`[336, 720, 8760]`) only added if SHAP confirms their value.
- Model persistence: save via `save_forecaster` from `spotforecast2_safe.manager.persistence` (writes `forecaster_{target}.joblib`); load via `joblib.load` directly (no public load counterpart exists in the library).

### Validation

- M1: single temporal split — train on all data except last `test_days` (default 30), evaluate on those last days
- M2: full walk-forward (time-series cross-validation) replaces the single split

### Visualisation

- `visualise` is a **standalone stage** — not part of the linear pipeline chain; can be called independently or from a future `report` stage via `visualise.run(config)`
- Assumes `evaluate` has always been run first; reads `evaluation_series.parquet` written by `evaluate` (contains: `actual`, `forecast`, `entso_daf`, `residuals_forecast`, `residuals_daf`)
- Backend abstraction: `PlotBackend` protocol in `src/ddkast/visualisation/protocol.py` with a single method `render(data: VisualisationData, config: Config) -> list[str]` — returns a list of output URIs printed as clickable terminal links by the stage
- Each backend has full autonomy over layout; the protocol does not prescribe which plots or how to arrange them
- **Plotly backend** (`src/ddkast/visualisation/plotly_backend.py`): interactive HTML, two linked subplots (load panel + residuals panel), toggleable traces, range slider
- **Matplotlib backend** (`src/ddkast/visualisation/matplotlib_backend.py`): static PDF, two-panel figure (forecast vs actual / residuals), suitable for reports
- Both backend files carry `# pyright: reportUnknownMemberType = false` at the top — plotly and matplotlib have incomplete type stubs; this suppresses the resulting noise without affecting other files
- `config.plots: list[str]` controls which data series are included (`"forecast"`, `"daf"`, `"residuals"`); backends read this field to decide what to render
- CLI: `ddkast visualise [--backend] [--from] [--to] [--plots]`; `--backend` and `--plots` override their `Config` defaults via `config.model_copy(update={...})`

## Milestones

- **2026-05-12**: working end-to-end demo (download → merge → train → predict → evaluate)
- **2026-06-23**: full feature engineering, hyperparameter tuning, SHAP
- **2026-07-21**: final presentation — uncertainty quantification, model card

## Deferred decisions

- **Pipeline stage caching**: revisit whether stages should cache their outputs to avoid re-running expensive steps (e.g. `train`, `predict`) when inputs haven't changed.

## Working with Claude

Before scaffolding or implementing any new module, feature, or significant design decision, interview the user about every relevant aspect — **one question at a time**, with a recommended answer and reasoning for each. Walk down the dependency tree of decisions in order. This process surfaces assumptions early and prevents rework. Do not batch questions; do not jump straight to code.

When moving files (refactoring, reorganising), always use `git mv` instead of a plain file move so git preserves history.
