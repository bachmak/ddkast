"""End-to-end smoke test for the fold-based train → predict → evaluate stages.

Uses synthetic data written directly to the processed store (the ``write_processed``
conftest fixture) — no API call needed. This module only checks the stages wire
together and produce the expected fold artifacts.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ddkast.config import Config
from ddkast.data.store import ParquetStore
from ddkast.folds import build_folds
from ddkast.pipeline import evaluate, predict, train
from tests.synthetic import fold_window

# Tiny lags + a couple of daily folds so the test stays fast.
_FOLDS = 2
_PERIODS = 24 * 12  # 12 days: covers 168h naive lookback + training history


@pytest.fixture
def e2e_config(config: Config) -> Config:
    return config.model_copy(
        update={"lags": 24, "horizon": 24, **fold_window(_PERIODS, _FOLDS)}
    )


@pytest.fixture
def seeded(
    e2e_config: Config, write_processed: Callable[[Config, int], None]
) -> Config:
    write_processed(e2e_config, _PERIODS)
    return e2e_config


def test_train_creates_a_model_per_fold(seeded: Config) -> None:
    train.run(seeded)
    folds = build_folds(
        ParquetStore(seeded.processed_dir).read(seeded.processed_load).index,  # type: ignore[arg-type]
        seeded,
    )
    assert len(folds) == _FOLDS + 1  # historical folds + the latest origin
    for fold in folds:
        model_path = (
            seeded.models_dir
            / "folds"
            / fold.fold_id
            / f"forecaster_{seeded.model_target}.joblib"
        )
        assert model_path.exists()


def test_predict_writes_per_fold(seeded: Config) -> None:
    train.run(seeded)
    predict.run(seeded)
    processed = ParquetStore(seeded.processed_dir)
    folds = build_folds(processed.read(seeded.processed_load).index, seeded)  # type: ignore[arg-type]

    # Every fold is forecast identically to predictions/<fold_id>; predict flags no
    # operational fold — format-submission selects that itself.
    for fold in folds:
        preds = processed.read(f"{seeded.predictions_subdir}/{fold.fold_id}")
        assert len(preds) == seeded.horizon


def test_evaluate_writes_evaluation_artifacts(seeded: Config) -> None:
    train.run(seeded)
    predict.run(seeded)
    evaluate.run(seeded)
    processed = ParquetStore(seeded.processed_dir)

    metrics = processed.read(seeded.evaluation_metrics)
    # 2 scored folds × {model, naive, daf} forecasters (the latest origin is future).
    assert set(metrics["forecaster"]) == {"model", "naive", "daf"}
    assert len(metrics) == _FOLDS * 3
    assert set(metrics.columns) >= {"fold_id", "origin", "forecaster", "MAE"}

    summary = processed.read(seeded.evaluation_summary)
    assert "skill_vs_naive" in set(summary["metric"])
    assert "skill_vs_daf" in set(summary["metric"])
