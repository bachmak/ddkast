## Critical constraints

- `spotforecast2-safe` is the **only** library allowed on the production code path. It is EU AI Act compliant and deterministic. `spotforecast2` may be used for development/experimentation only.
- All data access goes through the `DataStore` protocol. Never read/write Parquet files directly in pipeline or model code.
- `cli.py` must contain no business logic.

## Key decisions already made

- Parquet storage decoupled behind `DataStore` protocol
- MAE is the primary evaluation metric; report all four (MAE, RMSE, MAPE, SMAPE)
- Two benchmarks in `evaluate`: 7-day seasonal naive and the ENTSO-E published day-ahead forecast

### Data

- Download range: Jan 2022 – Apr 2026
- Both actual load and ENTSO-E day-ahead forecast (DAF) are fetched in a single `download` pass and stored separately
- Cleaning: IQR multiplier 3.0, max 3-hour linear interpolation; gaps longer than that cause the surrounding window to be rejected (fail-safe). Both values are configurable.
- All inter-stage filenames (e.g. `raw_load_actual`, `processed_load`) are fields in `Config` — nothing hardcoded in pipeline or model code

## Working with Claude

Before scaffolding or implementing any new module, feature, or significant design decision, interview the user about every relevant aspect — **one question at a time**, with a recommended answer and reasoning for each. Walk down the dependency tree of decisions in order. This process surfaces assumptions early and prevents rework. Do not batch questions; do not jump straight to code.

When moving files (refactoring, reorganising), always use `git mv` instead of a plain file move so git preserves history.
