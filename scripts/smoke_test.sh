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
  echo "  ✓ $1"
}

echo "── asserting artifacts ────────────────────────────"
assert_file "$DATA_DIR/raw/load_actual.parquet"
assert_file "$DATA_DIR/raw/load_forecast.parquet"
assert_file "$DATA_DIR/raw/weather_raw.parquet"
assert_file "$DATA_DIR/processed/load_clean.parquet"
assert_file "$DATA_DIR/processed/forecast_entso.parquet"
assert_file "$DATA_DIR/processed/weather_processed.parquet"
assert_file "$DATA_DIR/processed/load_test.parquet"
assert_file "$DATA_DIR/processed/load_predicted.parquet"
assert_file "$DATA_DIR/processed/evaluation_series.parquet"
assert_file "$MODELS_DIR/forecaster_load_mw.joblib"
assert_file "$PLOTS_DIR/forecast_analysis.pdf"

echo "✅ smoke test passed"
