from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from stocks.ai.active_swing_panel import (
    INFERENCE_PATH,
    infer_current_active_swing_candidates,
)
from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.p3.io import atomic_write_json, read_json


MODEL_MANIFEST_PATH = Path("output/ai/decision-intelligence/model-manifest.json")
TOURNAMENT_PATH = Path(
    "output/ai/decision-intelligence/active-swing-tournament.json"
)
MODEL_PATH = Path(
    "output/ai/decision-intelligence/active-swing-shadow-model.joblib"
)
MACHINE_STATUS_PATH = Path("output/operations/machine-status.json")


def infer_active_swing_fast_path(project_root: Path) -> dict[str, Any]:
    """Run exact candidate inference from a hash-verified frozen local bundle."""

    root = project_root.resolve()
    machine = read_json(root / MACHINE_STATUS_PATH)
    if machine and machine.get("enabled") is not True:
        inference = {
            "schema": "current_active_swing_candidate_inference_v1",
            "status": "SKIPPED_MACHINE_DISABLED",
            "generated_at": datetime.now(UTC).isoformat(),
            "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
            "current_candidate_count": 0,
            "evidence_count": 0,
            "model_evidence": [],
            "model_load_status": "NOT_ATTEMPTED_MACHINE_DISABLED",
            "model_artifact_hash_verified": False,
            "fast_path_retraining_performed": False,
            "financial_fields_mutated": False,
            "automatic_promotion": False,
            "authority": "SHADOW_ONLY",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "broker_writes": 0,
        }
        inference["content_hash"] = stable_hash(inference)
        atomic_write_json(root / INFERENCE_PATH, inference)
        return inference
    manifest = read_json(root / MODEL_MANIFEST_PATH)
    tournament = read_json(root / TOURNAMENT_PATH)
    model_path = root / MODEL_PATH
    bundle: dict[str, Any] = {}
    load_status = "NO_CURRENT_MODEL_ARTIFACT"
    expected_hash = str(manifest.get("active_swing_model_sha256") or "").upper()
    if manifest.get("active_swing_model_artifact_current") is True:
        if not model_path.is_file() or not expected_hash:
            load_status = "CURRENT_MODEL_ARTIFACT_MISSING"
        elif sha256_file(model_path).upper() != expected_hash:
            load_status = "CURRENT_MODEL_ARTIFACT_HASH_MISMATCH"
        else:
            try:
                loaded = joblib.load(model_path)
            except (OSError, ValueError, EOFError, TypeError):
                load_status = "CURRENT_MODEL_ARTIFACT_LOAD_FAILED"
            else:
                if (
                    isinstance(loaded, dict)
                    and loaded.get("candidate_unit")
                    == "ONE_NATURAL_STRATEGY_SETUP"
                    and loaded.get("model_version")
                    == manifest.get("active_swing_model_version")
                ):
                    bundle = loaded
                    load_status = "HASH_VERIFIED_FROZEN_MODEL_LOADED"
                else:
                    load_status = "CURRENT_MODEL_BUNDLE_IDENTITY_MISMATCH"
    inference = infer_current_active_swing_candidates(
        root,
        bundle,
        tournament,
        publish=False,
    )
    inference["model_load_status"] = load_status
    inference["model_artifact_hash_verified"] = (
        load_status == "HASH_VERIFIED_FROZEN_MODEL_LOADED"
    )
    inference["fast_path_retraining_performed"] = False
    inference["content_hash"] = stable_hash(
        {key: value for key, value in inference.items() if key != "content_hash"}
    )
    atomic_write_json(root / INFERENCE_PATH, inference)
    return inference


__all__ = ["infer_active_swing_fast_path"]
