# ddkast

Energy consumption forecasting for critical infrastructure.

Built for the *Data-Driven Optimization* course challenge (SoSe 2026, AIT).
The immediate target is electricity load forecasting for Germany using ENTSO-E data.
The long-term target is water consumption prediction for pump optimisation — the same pipeline applies because the underlying problem is identical: predict a time series 24 hours ahead so that an operator can schedule resources.

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

### Verify the setup

```bash
uv run pytest   # all tests should pass
```

### Run the pipeline

In VS Code, go to `Run and Debug` and run `pipeline (all stages)`

## Determinism audit

`scripts/audit_determinism.py` is the evidence generator for **CR-2** (same input →
bit-identical output). It runs the full pipeline — `download → merge → train →
predict → evaluate → visualise` — **twice from clean state** and compares every
artifact the two runs produced, so a single command proves both that all six
stages still work and that they reproduce.

```bash
uv run python scripts/audit_determinism.py
```

The two runs are sandboxed under `audit/<timestamp>/run_a/` and `run_b/`
(`data_dir`, `models_dir` and `plots_dir` are redirected there in memory), so your
real `data/` and `models/` are never touched, and run B cannot cache-hit off run A.
The raw data is downloaded **once** into `audit/<timestamp>/snapshot/` and copied
into both runs — that pins the inputs so an upstream ENTSO-E revision mid-audit
cannot make the runs disagree for reasons outside our code.

Only the forecast window is overridden (`n_forecasts=2` — two days is enough to
exercise the rolling-origin machinery without paying for a full year). The
download range and every model and cleaning knob keep their production values from
`config.py` / `config.toml`.

### Which days are audited

**If you pass nothing, the audit uses the last two complete days in the cleaned
load series** — it takes the end of the actuals, walks back to the most recent
midnight whose full `horizon` block is still covered by ground truth, and uses that
origin plus the one 24 h before it. That guarantees `evaluate` has actuals to score
both folds against. The chosen window is always printed before the runs start:

```
running audit on 2026-07-16 00:00 UTC → 2026-07-17 00:00 UTC (2 forecast origins, 24h each)
```

To audit specific days, pass both origins (ISO 8601, exactly 24 h apart):

```bash
uv run python scripts/audit_determinism.py --start 2026-07-14 --end 2026-07-15
```

The script refuses (CR-3) any window that is not two consecutive days or that
extends past the actuals, rather than producing an audit that silently compares two
empty evaluations.

### Reading the result

The claim being audited is that the **forecasts** reproduce. Every artifact gets a
SHA-256 byte hash; parquet files are additionally compared on content — values,
dtypes, index and column order, with the largest numeric delta reported — which
covers the per-fold predictions, the cleaned inputs and the evaluation.

| Status | Meaning | Verdict |
|---|---|---|
| `identical` | Byte-for-byte equal | pass |
| `byte-differs` | Parquet: same table, different bytes. Models/plots: hash moved | warn |
| `content-differs` | A parquet table actually moved | **fail** |
| `missing` | Produced by only one run | **fail** |

Persisted `.joblib` models and rendered plots are hashed and listed for the record
but never fail the audit: `spotforecast2-safe` stamps `fit_date`/`creation_date`
onto every fitted forecaster and plotly embeds a fresh div id per render, so those
files can never be byte-identical no matter how deterministic the training was. A
model that genuinely changed shows up where it matters — in the predictions.

Results land in `audit/<timestamp>/report.json` and `report.md` with full
provenance (git commit and dirty flag, Python and package versions, the exact
window used) — the Art. 11/12 record. Exit code is 0 on pass, 1 on fail, so the
script can gate CI.

The generated `audit/` directory is gitignored — it holds full copies of the data
and models. To keep a report as evidence, force-add just the report files:
`git add -f audit/<timestamp>/report.md`.