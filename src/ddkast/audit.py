"""Append-only audit log — mandatory record-keeping (EU AI Act Art. 12).

`record_run` writes one JSON line per pipeline-stage run to ``config.audit_dir``,
capturing the provenance needed to reconstruct a result later (Art. 11): UTC timestamp,
git SHA, package versions, a hash of the effective config, the data window, and the
stage's headline metrics. This is plain-text run metadata — deliberately *not* Parquet —
so it correctly lives outside the ``DataStore``.

Provenance fields (timestamp, git SHA) are intentionally non-deterministic and are
kept out of the deterministic data artifacts; the audit log is never golden-compared.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from ddkast.config import Config

AUDIT_FILENAME = "runs.jsonl"

# Packages whose versions are pinned into every record for reproducibility.
_TRACKED_PACKAGES = ("spotforecast2-safe", "lightgbm", "ddkast")


def _git_sha() -> str:
    """Current commit SHA, or ``"unknown"`` if git is unavailable (failsafe)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _package_versions() -> dict[str, str]:
    """Resolve installed versions of the tracked packages; ``"unknown"`` if absent."""
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _config_hash(config: Config) -> str:
    """sha256 of the effective config (sorted JSON) — pins the modelling inputs."""
    payload = json.dumps(config.model_dump(), default=str, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_run(
    config: Config,
    stage: str,
    status: str,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one provenance record for ``stage`` to the audit log and return it.

    The record carries a UTC timestamp, the git SHA, tracked package versions, the
    config hash, the data window, the stage ``status`` and headline ``metrics``, plus
    any ``extra`` fields (e.g. ``n_folds``, ``seed``). Created on first write.

    Args:
        config: Effective run config (hashed; supplies the data window and audit dir).
        stage: Pipeline stage name, e.g. ``"train"`` / ``"evaluate"``.
        status: Outcome marker, e.g. ``"ok"`` / ``"failed"``.
        metrics: Headline numeric metrics for the run (may be empty).
        extra: Optional additional provenance (n_folds, seeds, ...).

    Returns:
        The record dict that was appended (handy for tests/logging).
    """
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "status": status,
        "git_sha": _git_sha(),
        "config_hash": _config_hash(config),
        "packages": _package_versions(),
        "data_window": {
            "start": config.download_start.isoformat(),
            "end": config.download_end.isoformat(),
        },
        "metrics": metrics,
        "extra": extra or {},
    }

    config.audit_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with open(config.audit_dir / AUDIT_FILENAME, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return record
