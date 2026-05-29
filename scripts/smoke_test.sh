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
export DOWNLOAD_END="${DOWNLOAD_END:-2024-01-10}"
export LAGS="${LAGS:-48}"
export TEST_DAYS="${TEST_DAYS:-1}"
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
for stage in download merge train predict evaluate visualise; do
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
print(f"PROCESSED_TEST={c.processed_test}")
print(f"PROCESSED_PREDICTIONS={c.processed_predictions}")
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
assert_file "$DATA_DIR/processed/$PROCESSED_TEST.parquet"
assert_file "$DATA_DIR/processed/$PROCESSED_PREDICTIONS.parquet"
assert_file "$DATA_DIR/processed/$EVAL_SERIES.parquet"
assert_file "$MODELS_DIR/forecaster_${MODEL_TARGET}.joblib"
assert_file "$PLOTS_DIR/forecast_analysis.pdf"

echo "✅ smoke test passed"
