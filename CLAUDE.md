# ddkast — Claude Context

Energy consumption forecasting CLI for the DDMO course challenge (SoSe 2026, AIT). Worth 50% of the course grade.
Immediate target: electricity load forecasting for Germany (ENTSO-E, DE-LU bidding zone).
Long-term target: water consumption prediction for pump optimisation.

Full documentation: see `README.md`.
Open questions for the professor: see `QUESTIONS.md`.

## Architecture

- `src/` layout
- 5-stage CLI pipeline: `download → merge → train → predict → evaluate`
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
- **Quality:** `pyright` strict, `ruff`, `pytest`, pre-commit hooks (ruff + pyright on every commit)
- **CI:** GitHub Actions

## Key decisions already made

- Parquet storage decoupled behind `DataStore` protocol
- Walk-forward validation only — no random train/test splits
- MAE is the primary evaluation metric; report all four (MAE, RMSE, MAPE, SMAPE)
- Two benchmarks in `evaluate`: 7-day seasonal naive and the ENTSO-E published day-ahead forecast
- Cyclical encoding defaults to Fourier (sine/cosine); RBF decomposition is a planned Milestone 2 alternative to evaluate

## Milestones

- **2026-05-12**: working end-to-end demo (download → merge → train → predict → evaluate)
- **2026-06-23**: full feature engineering, hyperparameter tuning, SHAP
- **2026-07-21**: final presentation — uncertainty quantification, model card

## Working with Claude

Before scaffolding or implementing any new module, feature, or significant design decision, interview the user about every relevant aspect — **one question at a time**, with a recommended answer and reasoning for each. Walk down the dependency tree of decisions in order. This process surfaces assumptions early and prevents rework. Do not batch questions; do not jump straight to code.
