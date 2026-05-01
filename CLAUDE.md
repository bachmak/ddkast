# ddkast — Claude Context

Energy consumption forecasting CLI for the DDMO course challenge (SoSe 2026, AIT).
Long-term target: water consumption prediction for pump optimisation.

Full documentation: see `README.md`.
Open questions for the professor: see `QUESTIONS.md`.

## Critical constraints

- `spotforecast2-safe` is the **only** library allowed on the production code path. It is EU AI Act compliant and deterministic. `spotforecast2` may be used for development/experimentation only.
- All data access goes through the `DataStore` protocol (`src/ddkast/data/store.py`). Never read/write Parquet files directly in pipeline or model code.
- `Config` is always passed as an explicit argument to pipeline functions. It is never a global variable or module-level singleton.
- `cli.py` must stay thin — only argument parsing and calls to `pipeline.X.run(config)`. No business logic.

## Key decisions already made

- Python 3.13+, `uv` for package management
- `typer` for CLI, `rich` for terminal output, `pydantic-settings` for config
- Parquet storage (via `ParquetStore`), decoupled behind `DataStore` protocol
- `pyright` strict mode — all code must be fully type-annotated
- `ruff` for linting and formatting
- Pre-commit hooks run ruff + pyright before every commit
- Walk-forward validation only — no random train/test splits
- MAE is the primary evaluation metric; report all four (MAE, RMSE, MAPE, SMAPE)

## Milestones

- **2026-05-12**: working end-to-end demo (download → merge → train → predict → evaluate)
- **2026-06-23**: full feature engineering, hyperparameter tuning, SHAP
- **2026-07-21**: final presentation — uncertainty quantification, model card
