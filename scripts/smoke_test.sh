#!/usr/bin/env bash
#
# Offline end-to-end smoke test for the ddkast CLI pipeline.
#
# Runs all six stages (download → merge → train → predict → evaluate →
# visualise) against parquet fixtures generated on demand — no network, no API
# key — then asserts every expected artifact was produced. Catches "the pipeline
# is broken" regressions: a crashing stage, a mis-wired CLI command, Config
# plumbing drift, or a broken inter-stage contract.
#
# Every knob is an env var so CI (and you) can resize the run without touching
# code. The fixtures are generated into the temp dir and all writes are isolated
# there, leaving the repo pristine. Run from anywhere:
#
#     ./scripts/smoke_test.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- knobs (override by exporting before invoking) ---
export ENTSOE_API_KEY="${ENTSOE_API_KEY:-dummy}"  # required Config field, unused offline
export DATA_SOURCE=fixtures
export FIXTURES_DIR="$WORK/fixtures"
export DOWNLOAD_START="${DOWNLOAD_START:-2024-01-01}"
export DOWNLOAD_END="${DOWNLOAD_END:-2024-01-18}"
export LAGS="${LAGS:-48}"
export FOLDS="${FOLDS:-3}"  # historical folds; n_forecasts = FOLDS + 1 (live origin at the tail)
export HORIZON="${HORIZON:-24}"
export DATA_DIR="$WORK/data"
export MODELS_DIR="$WORK/models"
export PLOTS_DIR="$WORK/plots"
export BACKEND=matplotlib
export FIGURE_FORMAT=pdf

# Run from repo root so config.toml is found; env vars override its defaults.
cd "$ROOT"

echo "▶ generating offline fixtures → $FIXTURES_DIR"
uv run python tests/fixtures/generate.py "$FIXTURES_DIR"

echo "▶ smoke test (fixtures: $FIXTURES_DIR, work: $WORK)"
for stage in download merge; do
  echo "── $stage ─────────────────────────────────────────"
  uv run ddkast "$stage"
done

# Pin the rolling-origin fold window onto the freshly merged data: FOLDS historical
# origins plus one live origin at the data tail (mirrors the production daily path). Done
# here, not via Config, because the live tail is data-relative — the fixed forecasts_start/
# forecasts_end defaults target real 2025–26 data, not these 2024 fixtures.
echo "── pin fold window (FOLDS=$FOLDS historical + 1 live) ──"
eval "$(uv run python - <<'EOF'
import os
import pandas as pd
from ddkast.config import load
from ddkast.data.store import ParquetStore
config = load()
step = pd.Timedelta(config.resolution)
historical = int(os.environ["FOLDS"])
forecasts_end = ParquetStore(config.processed_dir).read(config.processed_load).index.max() + step
forecasts_start = end - historical * pd.Timedelta(hours=24)
fmt = "%Y-%m-%dT%H:%M:%SZ"
print(f"export N_FORECASTS={historical + 1}")
print(f"export FORECASTS_START={forecasts_start.strftime(fmt)}")
print(f"export FORECASTS_END={forecasts_end.strftime(fmt)}")
EOF
)"
echo "  $N_FORECASTS forecasts: $FORECASTS_START … $FORECASTS_END"

for stage in train predict evaluate visualise; do
  echo "── $stage ─────────────────────────────────────────"
  uv run ddkast "$stage"
done

assert_file() {
  if [[ ! -f "$1" ]]; then
    echo "✗ missing expected artifact: $1" >&2
    exit 1
  fi
  if [[ "$1" == *.parquet ]]; then
    local rows
    rows=$(uv run python -c "import pyarrow.parquet as pq; print(pq.read_metadata('$1').num_rows)")
    if [[ "$rows" == "0" ]]; then
      echo "✗ empty parquet (0 rows): $1" >&2
      exit 1
    fi
  fi
  echo "  ✓ $1"
}

echo "── asserting artifacts ────────────────────────────"
# Derive artifact names from Config defaults so renames stay in sync.
eval "$(uv run python - <<'EOF'
from ddkast.config import Config
c = Config(entsoe_api_key="x")
print(f"RAW_LOAD_ACTUAL={c.raw_load_actual}")
print(f"RAW_LOAD_FORECAST={c.raw_load_forecast}")
print(f"RAW_WEATHER={c.raw_weather}")
print(f"PROCESSED_LOAD={c.processed_load}")
print(f"PROCESSED_ENTSO={c.processed_entso_forecast}")
print(f"PROCESSED_WEATHER={c.processed_weather}")
print(f"PREDICTIONS_SUBDIR={c.predictions_subdir}")
print(f"EVALUATION_METRICS={c.evaluation_metrics}")
print(f"EVALUATION_SUMMARY={c.evaluation_summary}")
print(f"EVAL_SERIES={c.evaluation_series}")
print(f"MODEL_TARGET={c.model_target}")
EOF
)"

assert_file "$DATA_DIR/raw/$RAW_LOAD_ACTUAL.parquet"
assert_file "$DATA_DIR/raw/$RAW_LOAD_FORECAST.parquet"
assert_file "$DATA_DIR/raw/$RAW_WEATHER.parquet"
assert_file "$DATA_DIR/processed/$PROCESSED_LOAD.parquet"
assert_file "$DATA_DIR/processed/$PROCESSED_ENTSO.parquet"
assert_file "$DATA_DIR/processed/$PROCESSED_WEATHER.parquet"
assert_file "$DATA_DIR/processed/$EVALUATION_METRICS.parquet"
assert_file "$DATA_DIR/processed/$EVALUATION_SUMMARY.parquet"
assert_file "$DATA_DIR/processed/$EVAL_SERIES.parquet"
assert_file "$PLOTS_DIR/forecast_analysis.pdf"

assert_glob() {
  # Assert at least one file matches the glob pattern in $1.
  local matches=("$1")
  if [[ ! -e "${matches[0]}" ]]; then
    echo "✗ no files matching: $1" >&2
    exit 1
  fi
  echo "  ✓ ${matches[0]} (+others)"
}

# Per-fold predictions and per-fold models are written under nested dirs.
assert_glob "$DATA_DIR/processed/$PREDICTIONS_SUBDIR/"*.parquet
assert_glob "$MODELS_DIR/folds/"*/"forecaster_${MODEL_TARGET}.joblib"

echo "✅ smoke test passed"
