# Model Card — ddkast

This card describes the **toolkit**: the pipeline, its scope, and how to audit it.
For the forecasting model itself (recipe, training data, evaluation numbers), see the per-target model card.

## Details

- **Type**: a 7-command CLI (`download`, `merge`, `train`, `predict`, `evaluate`, `format-submission`, `visualise`) built on Typer, orchestrating a deterministic time-series forecasting pipeline.
  Note: this is one more stage than the "6-stage pipeline" described elsewhere in this repo's documentation — `format-submission` was added later to support leaderboard submission and is not yet reflected there.
- **Production-path dependency**: [`spotforecast2-safe`](https://pypi.org/project/spotforecast2-safe/) is the only forecasting library permitted in `src/`. It is EU AI Act compliant and deterministic. The unconstrained `spotforecast2` package may be used for development/experimentation only, never on the code path that produces a submitted or evaluated forecast.
- **Domain**: electricity load forecasting for Germany (ENTSO-E), classified as a high-risk AI system under EU AI Act Annex III No. 2 (critical infrastructure).

## Intended Use and Scope

ddkast is a **live leaderboard competition entry**, not a course demo. A scheduled GitHub Actions job (`.github/workflows/daily-forecast.yml`, cron `0 17 * * *` UTC) runs the pipeline end-to-end every day: it downloads the freshest actuals, pins a single forecast origin to the latest available timestamp, trains, predicts, and submits tomorrow's 24-hour forecast as a CSV pull request against the upstream leaderboard repository.

This is **explicitly not validated for real grid operations**. It has not been assessed for the accuracy, latency, or failure-mode guarantees that operational grid forecasting would require, and it must not be used to inform actual dispatch, balancing, or infrastructure decisions.

## Architecture

The pipeline is a linear chain of CLI stages, each reading and writing through the `DataStore` protocol (no direct Parquet access in pipeline or model code):

```
download → merge → train → predict → evaluate
                                    → format-submission
                      (any stage) → visualise
```

Two structures shape how the later stages operate:

- **`folds.py`** is the single source of truth for forecast origins. `build_folds()` deterministically derives a list of `Fold` objects (rolling-origin, half-open forecast blocks) from the training history and `Config`. `evaluate` uses the full rolling-origin fold set (365 daily folds spanning 2025-05-02 → 2026-05-01); the daily-forecast CI job overrides `Config` to produce exactly one fold per run, pinned to the latest actual.
- **Per-fold model directory layout**: `train` persists one model per fold under `models/folds/<fold_id>/`, rather than shipping a single model artifact. `predict` reads from the same path. This is why `models/` (gitignored) never contains one canonical forecaster — every fold gets its own.

## How to Audit

```bash
uv run pytest        # test suite — every function/branch should be exercised (CR-1)
uv run ruff check     # linting
uv run pyright        # static type checking (strict mode)
uv run pre-commit run --all-files   # ruff + ruff-format + pyright, as enforced pre-commit
```

The CR-1–CR-4 rules enforced on `spotforecast2-safe` usage (no dead code, determinism, fail-safe processing, minimal CVE attack surface) are documented in `CLAUDE.md` and are exercised by the test suite above, not by a separate audit tool.

## Disclaimer

This model card describes the system as implemented at the time of writing. It is provided for transparency and audit purposes under EU AI Act Art. 13 and does not constitute a safety or performance guarantee. The system is a competition/research artifact and has not been validated for production grid operations.
