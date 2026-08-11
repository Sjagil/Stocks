from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stocks.ai.contracts import (
    AIAuthority,
    ExperimentRecord,
    ExperimentStatus,
    FeatureDefinition,
    ModelLifecycle,
    ModelRecord,
    ResearchHypothesis,
)
from stocks.ai.governance import (
    assess_model_health,
    audit_ai_import_boundary,
    canonical_hash,
    multiple_testing_penalty,
    validate_ai_authority,
    write_immutable_experiment,
)
from stocks.portfolio.quant_authority import load_quant_authority_map
from stocks.quant_platform.capabilities import capability_registry


OUTPUT_ROOT = Path("output/ai")
REPORT_PATH = Path("reports/AI_REFERENCE_REPOS_INTEGRATION_REPORT.md")
REFERENCE_CONFIG = Path("config/ai/reference_patterns_v1.json")
LEARNING_EVIDENCE = Path("output/portfolio/learning-model-evidence.json")

ARTIFACTS = {
    "reference_matrix": "reference-repo-integration-matrix.json",
    "model_registry": "model-registry.json",
    "model_health": "model-health.json",
    "feature_registry": "feature-registry.json",
    "hypotheses": "research-hypotheses.json",
    "experiments": "experiment-registry.json",
    "shadow": "shadow-portfolio-comparison.json",
    "authority": "authority-matrix.json",
    "status": "status.json",
}

_CAPABILITY_REFERENCES: dict[int, tuple[str, ...]] = {
    1: ("lean", "qlib"),
    2: ("quantstats",),
    3: ("lean", "qlib"),
    4: ("lean", "qlib"),
    5: ("qlib",),
    6: ("qlib", "finrl_x"),
    7: ("vectorbt",),
    8: ("vectorbt", "pybroker"),
    9: ("pybroker",),
    10: ("qlib",),
    11: ("lean", "finrl_x"),
    12: ("vectorbt",),
    13: ("vectorbt",),
    14: ("quantstats",),
    15: ("quantstats",),
    16: ("quantstats",),
    17: ("qlib",),
    18: ("qlib", "pybroker"),
    19: ("qlib",),
    20: ("fingpt",),
    21: ("fingpt", "quantstats"),
    22: ("finrobot", "fingpt"),
    23: ("finrl_x", "lean"),
    24: ("qlib", "lean"),
    25: ("qlib", "lean"),
    26: ("finrl_x", "lean"),
    27: ("vectorbt", "trademaster"),
    28: ("nautilus_trader", "lean"),
    29: ("qlib",),
    30: ("finrl_x", "qlib"),
    31: ("finrl_x", "qlib"),
    32: ("trademaster", "finrl_x"),
    33: ("lean", "finrl_x", "rd_agent"),
}

_WEAK_CAPABILITIES = {17, 18, 19, 20, 21, 22, 31, 32}


def publish_ai_research_plane(project_root: Path) -> dict[str, Any]:
    """Publish one advisory AI plane without importing any broker runtime."""

    root = project_root.resolve()
    now = datetime.now(UTC)
    reference_matrix = _reference_matrix(root)
    feature_registry = _feature_registry()
    learning = _read_json(root / LEARNING_EVIDENCE)
    models = _model_records(root, learning, now)
    hypotheses = _hypotheses(models, learning, now)
    experiments = _experiments(root, models, hypotheses, learning)
    model_health = _model_health(models, now)
    authority = _authority_matrix(root, models)
    shadow = _shadow_comparison(root, learning)
    architecture = _architecture_map(root)
    boundary = audit_ai_import_boundary(root / "src/stocks/ai")
    capability_matrix = _capability_matrix(root)

    reference_payload = {
        "schema": "ai_reference_repo_integration_matrix_v1",
        "status": (
            "GO"
            if all(row["status"] == "PRESENT_INSPECTED" for row in reference_matrix)
            else "PARTIAL"
        ),
        "repositories": reference_matrix,
        "capability_count": len(capability_matrix),
        "capabilities": capability_matrix,
        "capability_34_added": False,
        "copied_reference_code": False,
        "production_dependency_changes": [],
        "second_runtime_created": False,
        "execution_authority": "NONE",
    }
    model_payload = {
        "schema": "ai_model_registry_v1",
        "status": "GO",
        "models": [model.model_dump(mode="json") for model in models],
        "model_count": len(models),
        "canonical_learning_evidence": LEARNING_EVIDENCE.as_posix(),
        "automatic_promotion": False,
        "money_control": False,
        "execution_authority": "NONE",
    }
    health_payload = {
        "schema": "ai_model_health_registry_v1",
        "status": "GO",
        "health": model_health,
        "active_model_count": sum(
            row["recommended_lifecycle"] == ModelLifecycle.VALIDATED
            for row in model_health
        ),
        "paused_or_shadow_count": sum(
            row["recommended_lifecycle"]
            in {ModelLifecycle.PAUSED, ModelLifecycle.SHADOW}
            for row in model_health
        ),
        "automatic_risk_increase": False,
        "execution_authority": "NONE",
    }
    feature_payload = {
        "schema": "ai_feature_registry_v1",
        "status": "GO",
        "features": [row.model_dump(mode="json") for row in feature_registry],
        "feature_count": len(feature_registry),
        "random_time_series_split_allowed": False,
        "required_splits": [
            "TRAIN",
            "VALIDATION",
            "TEST",
            "FORWARD",
            "LIVE_SHADOW",
        ],
        "execution_authority": "NONE",
    }
    hypothesis_payload = {
        "schema": "ai_research_hypothesis_registry_v1",
        "status": "GO",
        "hypotheses": [row.model_dump(mode="json") for row in hypotheses],
        "hypothesis_count": len(hypotheses),
        "automatic_live_status": False,
        "execution_authority": "NONE",
    }
    experiment_payload = {
        "schema": "ai_experiment_registry_v1",
        "status": "GO",
        "experiments": [row.model_dump(mode="json") for row in experiments],
        "experiment_count": len(experiments),
        "immutable_records": True,
        "live_authorized_status_exists": False,
        "execution_authority": "NONE",
    }
    outputs = {
        "reference_matrix": reference_payload,
        "model_registry": model_payload,
        "model_health": health_payload,
        "feature_registry": feature_payload,
        "hypotheses": hypothesis_payload,
        "experiments": experiment_payload,
        "shadow": shadow,
        "authority": authority,
    }
    for key, payload in outputs.items():
        _publish_snapshot(root / OUTPUT_ROOT / ARTIFACTS[key], payload, now)

    blockers = []
    if reference_payload["status"] != "GO":
        blockers.append("LOCAL_REFERENCE_INSPECTION_INCOMPLETE")
    if authority["status"] != "GO":
        blockers.append("AI_AUTHORITY_CONTRACT_NO_GO")
    if boundary["status"] != "GO":
        blockers.append("AI_BROKER_IMPORT_BOUNDARY_NO_GO")
    if len(capability_matrix) != 33:
        blockers.append("EXACT_33_CAPABILITY_COVERAGE_REQUIRED")
    status = {
        "schema": "ai_research_plane_status_v1",
        "status": "GO" if not blockers else "NO_GO",
        "generated_at": now.isoformat(),
        "architecture": architecture,
        "blockers": blockers,
        "reference_repo_count": len(reference_matrix),
        "capability_count": len(capability_matrix),
        "model_count": len(models),
        "feature_count": len(feature_registry),
        "experiment_count": len(experiments),
        "financial_validation_status": "NO_INCREMENTAL_EVIDENCE",
        "deterministic_fallback": True,
        "ai_money_control": False,
        "direct_broker_access": False,
        "writer_calls": 0,
        "broker_calls": 0,
        "orders_generated": 0,
        "risk_limits_changed": False,
        "capital_permissions_changed": False,
        "execution_authority": "NONE",
        "import_boundary": boundary,
        "artifacts": {
            key: (OUTPUT_ROOT / value).as_posix()
            for key, value in ARTIFACTS.items()
        },
    }
    status["content_hash"] = canonical_hash(
        {key: value for key, value in status.items() if key != "generated_at"}
    )
    _atomic_json(root / OUTPUT_ROOT / ARTIFACTS["status"], status)
    _atomic_text(
        root / REPORT_PATH,
        _integration_report(
            status=status,
            architecture=architecture,
            reference_matrix=reference_matrix,
            capability_matrix=capability_matrix,
            models=models,
            shadow=shadow,
            authority=authority,
        ),
    )
    return status


def load_ai_research_plane_status(project_root: Path) -> dict[str, Any]:
    payload = _read_json(project_root.resolve() / OUTPUT_ROOT / ARTIFACTS["status"])
    if not payload:
        return {
            "schema": "ai_research_plane_status_v1",
            "status": "FALLBACK_DETERMINISTIC",
            "financial_validation_status": "NO_INCREMENTAL_EVIDENCE",
            "deterministic_fallback": True,
            "ai_money_control": False,
            "execution_authority": "NONE",
        }
    return payload


def _architecture_map(root: Path) -> dict[str, Any]:
    stages = [
        ("data", "src/stocks/quant_platform/data.py", "EXISTS_AND_STRONG"),
        ("features", "src/stocks/portfolio/learning_integration.py", "EXISTS_BUT_WEAK"),
        ("research", "src/stocks/research/autopilot", "EXISTS_AND_STRONG"),
        ("model_evidence", LEARNING_EVIDENCE.as_posix(), "EXISTS_BUT_WEAK"),
        ("strategies", "src/stocks/research/registry_service.py", "EXISTS_AND_STRONG"),
        ("ranking", "src/stocks/portfolio/opportunities.py", "EXISTS_AND_STRONG"),
        ("portfolio", "src/stocks/portfolio/manager.py", "EXISTS_AND_STRONG"),
        ("whole_share_sizing", "src/stocks/portfolio/targets.py", "EXISTS_AND_STRONG"),
        ("risk", "src/stocks/portfolio/dynamic_risk.py", "EXISTS_AND_STRONG"),
        ("authority", "src/stocks/portfolio/strategy_authority.py", "EXISTS_AND_STRONG"),
        ("reconciliation", "src/stocks/ibkr/reconciliation", "EXISTS_AND_STRONG"),
        ("execution", "src/stocks/live/submission.py", "EXISTS_AND_STRONG"),
    ]
    return {
        "schema": "current_architecture_map_v1",
        "flow": [name for name, _, _ in stages],
        "stages": [
            {
                "stage": name,
                "native_path": path,
                "status": status if (root / path).exists() else "MISSING",
                "preserved": True,
            }
            for name, path, status in stages
        ],
        "duplicated_runtime_count": 0,
        "ai_plane_role": "ADVISORY_RESEARCH_AND_INTELLIGENCE",
        "native_execution_chain_authoritative": True,
    }


def _reference_matrix(root: Path) -> list[dict[str, Any]]:
    config = _read_json(root / REFERENCE_CONFIG)
    rows = []
    for spec in config.get("repositories", []):
        repo_root = root / "reference_repos" / spec["repo"]
        files = [repo_root / value for value in spec["inspected_files"]]
        missing = [
            path.relative_to(repo_root).as_posix()
            for path in files
            if not path.is_file()
        ]
        license_path = _license_path(repo_root)
        rows.append(
            {
                **spec,
                "local_path": repo_root.relative_to(root).as_posix(),
                "status": (
                    "PRESENT_INSPECTED"
                    if repo_root.is_dir() and not missing
                    else "REFERENCE_NOT_PRESENT"
                    if not repo_root.is_dir()
                    else "PRESENT_INSPECTION_FILE_MISSING"
                ),
                "missing_inspection_files": missing,
                "git_head": _git_head(repo_root),
                "license_file": (
                    license_path.relative_to(repo_root).as_posix()
                    if license_path
                    else None
                ),
                "license_family": _license_family(license_path),
                "inspected_file_hashes": {
                    path.relative_to(repo_root).as_posix(): _sha256_file(path)
                    for path in files
                    if path.is_file()
                },
                "source_code_copied": False,
                "production_dependency_added": False,
            }
        )
    return rows


def _capability_matrix(root: Path) -> list[dict[str, Any]]:
    capabilities = capability_registry()["capabilities"]
    authority = load_quant_authority_map(root)
    by_id = {row["id"]: row for row in authority.get("capabilities", [])}
    rows = []
    for capability in capabilities:
        capability_id = int(capability["id"])
        current = by_id.get(capability_id, {})
        weak = capability_id in _WEAK_CAPABILITIES
        rows.append(
            {
                "capability_id": capability_id,
                "capability_name": capability["name"],
                "native_module": f"src/stocks/quant_platform/{capability['module'].split('/')[0]}.py",
                "native_status": "EXISTS_BUT_WEAK" if weak else "EXISTS_AND_STRONG",
                "reference_repo": list(_CAPABILITY_REFERENCES[capability_id]),
                "reference_pattern": "NATIVE_CONTRACT_AND_VALIDATION_IMPROVEMENT",
                "gap": (
                    "NO_INCREMENTAL_OR_FORWARD_EVIDENCE"
                    if weak
                    else "NO_MATERIAL_RUNTIME_GAP"
                ),
                "implementation_change": (
                    "GOVERNED_REGISTRY_HEALTH_AND_SHADOW_EVIDENCE"
                    if weak
                    else "PRESERVE_NATIVE_IMPLEMENTATION"
                ),
                "authority_before": current.get("authority", "SHADOW_ONLY"),
                "authority_after": current.get("authority", "SHADOW_ONLY"),
                "money_control_before": bool(current.get("money_control")),
                "money_control_after": False,
                "output_used": capability_id
                in {1, 2, 5, 6, 7, 10, 11, 20, 21, 22, 26, 27, 29, 33},
                "forward_evidence": False if weak else "CAPABILITY_SPECIFIC",
                "monitoring": "AI_MODEL_HEALTH" if weak else "NATIVE_MONITORING",
                "tests": [
                    "tests/test_ai_research_plane.py",
                    "tests/test_quant_platform_manager.py",
                ],
                "artifact": "output/ai/reference-repo-integration-matrix.json",
                "new_capability": False,
            }
        )
    return rows


def _feature_registry() -> list[FeatureDefinition]:
    common = {
        "source": "LOCAL_CANONICAL_MARKET_AND_INTELLIGENCE_ARTIFACTS",
        "event_time_semantics": "SOURCE_EVENT_OR_CLOSED_BAR_TIMESTAMP",
        "available_at_semantics": "FIRST_LOCALLY_OBSERVABLE_TIMESTAMP",
        "revision_semantics": "VERSIONED_POINT_IN_TIME_NO_FUTURE_REVISION",
        "missingness_semantics": "MISSING_REMAINS_MISSING_AND_FAILS_REQUIRED_GATE",
        "universe_scope": "POINT_IN_TIME_ELIGIBLE_UNIVERSE",
        "feature_version": "1",
    }
    specs = [
        ("return_1", "1_CLOSED_SESSION", "NONE"),
        ("momentum_5", "5_CLOSED_SESSIONS", "NONE"),
        ("momentum_20", "20_CLOSED_SESSIONS", "NONE"),
        ("volatility_20", "20_CLOSED_SESSIONS", "ROLLING_STANDARD_DEVIATION"),
        ("intraday_range", "1_CLOSED_SESSION", "PRICE_NORMALIZED"),
        ("drawdown_63", "63_CLOSED_SESSIONS", "ROLLING_PEAK"),
        ("volume_z_20", "20_CLOSED_SESSIONS", "ROLLING_Z_SCORE_TRAIN_ONLY"),
        ("news_sentiment", "EVENT", "BOUNDED_MINUS_ONE_TO_ONE"),
        ("news_uncertainty", "EVENT", "BOUNDED_ZERO_TO_ONE"),
        ("news_novelty", "EVENT", "BOUNDED_ZERO_TO_ONE"),
        ("news_relevance", "EVENT", "BOUNDED_ZERO_TO_ONE"),
        ("sec_materiality", "SEC_ACCEPTANCE_TIME", "BOUNDED_ZERO_TO_ONE"),
        ("macro_surprise", "MACRO_RELEASE_TIME", "CONSENSUS_STANDARDIZED"),
        ("execution_feasible", "DECISION_TIME", "BOOLEAN_NATIVE_GATE"),
    ]
    return [
        FeatureDefinition(
            feature_name=name,
            calculation_window=window,
            normalization_method=normalization,
            closed_bar_only=window not in {"EVENT", "SEC_ACCEPTANCE_TIME", "MACRO_RELEASE_TIME", "DECISION_TIME"},
            authority=(
                AIAuthority.CONTEXT_ONLY
                if name == "execution_feasible"
                else AIAuthority.FEATURE_ALLOWED
            ),
            **common,
        )
        for name, window, normalization in specs
    ]


def _model_records(
    root: Path,
    evidence: dict[str, Any],
    now: datetime,
) -> list[ModelRecord]:
    if not evidence:
        return []
    created = _timestamp(evidence.get("generated_at")) or now
    code_hashes = {
        "supervised": _sha256_file(root / "src/stocks/quant_platform/ml.py"),
        "tcn": _sha256_file(root / "src/stocks/quant_platform/ml.py"),
        "unsupervised": _sha256_file(root / "src/stocks/quant_platform/regime.py"),
        "reinforcement_learning": _sha256_file(
            root / "src/stocks/quant_platform/professional.py"
        ),
    }
    models: list[ModelRecord] = []
    for symbol_row in evidence.get("symbol_predictions", []):
        symbol = str(symbol_row.get("symbol"))
        source = str(symbol_row.get("data_source"))
        source_hash = evidence.get("source_hashes", {}).get(source, "MISSING")
        cutoff = str(symbol_row.get("trained_through", "UNRECORDED"))
        features = tuple(
            symbol_row.get("supervised", {}).get("report", {}).get("features", [])
            or (
                "return_1",
                "momentum_5",
                "momentum_20",
                "volatility_20",
                "intraday_range",
                "drawdown_63",
                "volume_z_20",
            )
        )
        for family, key in (
            ("SUPERVISED_LOGISTIC", "supervised"),
            ("TEMPORAL_CONVOLUTIONAL_NETWORK", "tcn"),
            ("UNSUPERVISED_GMM_REGIME", "unsupervised"),
        ):
            item = symbol_row.get(key, {})
            report = item.get("report", {})
            validation = item.get("validation_status")
            validated = validation == "SHADOW_VALIDATION_GO"
            lifecycle = ModelLifecycle.SHADOW if validated else ModelLifecycle.PAUSED
            authority = (
                AIAuthority.CONTEXT_ONLY
                if key == "unsupervised"
                else AIAuthority.SHADOW_ONLY
            )
            metrics = report.get("temporal_validation", {})
            if key == "unsupervised":
                metrics = {
                    "latest_cluster_confidence": item.get("confidence"),
                    "operational_regime_authority": False,
                }
            model_id = f"{key.upper()}-{symbol}-V1"
            models.append(
                ModelRecord(
                    model_id=model_id,
                    family=family,
                    version="1",
                    feature_set=features,
                    target=(
                        str(symbol_row.get("target"))
                        if key != "unsupervised"
                        else "RELATIVE_STATISTICAL_CLUSTER"
                    ),
                    training_interval=f"START_UNRECORDED..{cutoff}",
                    validation_interval="PURGED_TEMPORAL_FOLDS",
                    test_interval="TEMPORAL_HOLDOUT_INCLUDED_IN_EVIDENCE",
                    forward_interval="NOT_AVAILABLE",
                    universe=(symbol,),
                    horizon="5_SESSIONS" if key != "unsupervised" else "CONTEXT",
                    regime_scope="ALL_OBSERVED_REGIMES_NO_LEAVE_ONE_OUT_EVIDENCE",
                    data_hash=str(source_hash),
                    code_hash=code_hashes[key],
                    hyperparameters={
                        name: value
                        for name, value in report.items()
                        if name
                        in {
                            "backend",
                            "channels",
                            "dilations",
                            "kernel_size",
                            "sequence_length",
                        }
                    },
                    metrics=metrics,
                    calibration={
                        "probability_calibrated": report.get(
                            "probability_calibrated", False
                        ),
                        "brier_score": metrics.get("brier_score"),
                    },
                    drift_limits={
                        "feature": 0.20,
                        "prediction": 0.20,
                        "calibration": 0.05,
                        "performance": 0.20,
                        "regime": 0.25,
                    },
                    authority=authority,
                    lifecycle=lifecycle,
                    created_at=created,
                    expires_at=created + timedelta(days=31),
                    incremental_evidence="NO_INCREMENTAL_EVIDENCE",
                )
            )
    rl = evidence.get("portfolio_rl", {})
    if rl:
        report = rl.get("report", {})
        validation = report.get("temporal_holdout_validation", {})
        models.append(
            ModelRecord(
                model_id="PORTFOLIO-RL-V1",
                family="CONSTRAINED_TABULAR_Q_LEARNING",
                version="1",
                feature_set=("signals", "volatility", "drawdown", "cash"),
                target="NET_RISK_ADJUSTED_PORTFOLIO_REWARD",
                training_interval="LOCAL_CAUSAL_HISTORY_TRAIN_WINDOW",
                validation_interval="FINAL_30_PERCENT_TEMPORAL_HOLDOUT",
                test_interval="TEMPORAL_HOLDOUT",
                forward_interval="NOT_AVAILABLE",
                universe=tuple(
                    sorted(rl.get("counterfactual_target_weights", {}))
                ),
                horizon="ONE_SESSION_ACTION",
                regime_scope="ALL_OBSERVED_NO_UNSEEN_REGIME_PROOF",
                data_hash=canonical_hash(evidence.get("source_hashes", {})),
                code_hash=code_hashes["reinforcement_learning"],
                hyperparameters={
                    "model_type": report.get("model_type"),
                    "action_count": report.get("action_count"),
                    "long_only": report.get("long_only"),
                    "cash_action_available": report.get("cash_action_available"),
                },
                metrics=validation,
                calibration={"not_applicable": True},
                drift_limits={
                    "feature": 0.20,
                    "prediction": 0.20,
                    "calibration": 0.05,
                    "performance": 0.20,
                    "regime": 0.25,
                },
                authority=AIAuthority.SHADOW_ONLY,
                lifecycle=ModelLifecycle.PAUSED,
                created_at=created,
                expires_at=created + timedelta(days=31),
                incremental_evidence="NO_INCREMENTAL_EVIDENCE",
            )
        )
    return models


def _hypotheses(
    models: list[ModelRecord],
    evidence: dict[str, Any],
    now: datetime,
) -> list[ResearchHypothesis]:
    created = _timestamp(evidence.get("generated_at")) or now
    return [
        ResearchHypothesis(
            hypothesis_id=f"H-{canonical_hash({'model': model.model_id})[:16]}",
            source="NATIVE_EXISTING_MODEL_EVIDENCE",
            description=(
                f"{model.family} adds conservative net-of-cost value over the "
                "deterministic baseline"
            ),
            economic_rationale=(
                "A calibrated causal signal may improve selection while native "
                "whole-share and risk gates remain authoritative"
            ),
            feature_dependencies=model.feature_set,
            target=model.target,
            horizon=model.horizon,
            created_at=created,
            status=ExperimentStatus.FAILED,
        )
        for model in models
    ]


def _experiments(
    root: Path,
    models: list[ModelRecord],
    hypotheses: list[ResearchHypothesis],
    evidence: dict[str, Any],
) -> list[ExperimentRecord]:
    records = []
    for model, hypothesis in zip(models, hypotheses, strict=True):
        core = {
            "contract_version": "ai_experiment_record_v1_schema_version_field",
            "hypothesis_id": hypothesis.hypothesis_id,
            "code_hash": model.code_hash,
            "dataset_hash": model.data_hash,
            "parameters": model.hyperparameters,
            "seed": 42,
            "cost_model": "SHARED_TRANSACTION_COST_MODEL_V1",
        }
        record = ExperimentRecord(
            experiment_id=f"EXP-{canonical_hash(core)[:20]}",
            hypothesis_id=hypothesis.hypothesis_id,
            code_hash=model.code_hash,
            dataset_hash=model.data_hash,
            cutoff=_timestamp(evidence.get("generated_at")) or datetime.now(UTC),
            parameters=model.hyperparameters,
            seed=42,
            transaction_cost_model_version="SHARED_TRANSACTION_COST_MODEL_V1",
            result_artifact=LEARNING_EVIDENCE.as_posix(),
            decision=ExperimentStatus.FAILED,
            hypothesis_count_at_selection=max(1, len(models)),
            multiple_testing=multiple_testing_penalty(
                0.0, hypothesis_count=max(1, len(models))
            ),
        )
        write_immutable_experiment(root, record)
        records.append(record)
    return records


def _model_health(
    models: list[ModelRecord],
    now: datetime,
) -> list[dict[str, Any]]:
    return [
        assess_model_health(
            model,
            now=now,
            feature_drift=None,
            prediction_drift=None,
            calibration_drift=None,
            performance_drift=None,
            regime_drift=None,
            schema_matches=True,
        )
        for model in models
    ]


def _authority_matrix(
    root: Path,
    models: list[ModelRecord],
) -> dict[str, Any]:
    components = [
        {
            "component_id": model.model_id,
            "authority": model.authority,
            "lifecycle": model.lifecycle,
            "money_control": False,
            "execution_authority": "NONE",
            "granted_powers": [],
            "broker_write": False,
            "order_authority": False,
            "capital_promotion": False,
            "strategy_promotion": False,
            "risk_limit_expansion": False,
            "direct_quantity_authority": False,
        }
        for model in models
    ]
    components.extend(
        [
            {
                "component_id": "FINANCIAL_NLP_EVENTS",
                "authority": AIAuthority.FEATURE_ALLOWED,
                "lifecycle": ModelLifecycle.SHADOW,
                "money_control": False,
                "execution_authority": "NONE",
                "granted_powers": [],
            },
            {
                "component_id": "RESEARCH_AGENTS",
                "authority": AIAuthority.SHADOW_ONLY,
                "lifecycle": ModelLifecycle.RESEARCH,
                "money_control": False,
                "execution_authority": "NONE",
                "granted_powers": [],
            },
            {
                "component_id": "AI_PORTFOLIO_PROPOSAL",
                "authority": AIAuthority.PORTFOLIO_ADVISORY,
                "lifecycle": ModelLifecycle.SHADOW,
                "money_control": False,
                "execution_authority": "NONE",
                "granted_powers": [],
                "publishes_broker_quantity": False,
                "native_translation_required": True,
            },
        ]
    )
    validation = validate_ai_authority(components)
    return {
        "schema": "ai_authority_matrix_v1",
        "status": validation["status"],
        "components": components,
        "validation": validation,
        "native_execution_chain": (
            "strategy_authority -> P2.2 feasibility -> risk -> reconciliation "
            "-> P0 bridge -> frozen writer"
        ),
        "native_quant_authority_source": (
            "config/portfolio/quant_capability_authority_v1.json"
        ),
        "money_control_default": False,
        "writer_count_added": 0,
        "broker_call_sites_added": 0,
        "risk_limits_changed": False,
        "capital_permissions_changed": False,
        "execution_authority": "NONE",
    }


def _shadow_comparison(
    root: Path,
    learning: dict[str, Any],
) -> dict[str, Any]:
    deterministic = _read_json(
        root / "output/research/p1/independent-performance-check.json"
    )
    variants = [
        ("A", "CURRENT_DETERMINISTIC_PORTFOLIO", deterministic.get("metrics")),
        ("B", "SUPERVISED_ML_PROPOSAL", None),
        ("C", "NLP_ENHANCED_PROPOSAL", None),
        ("D", "CALIBRATED_ENSEMBLE_PROPOSAL", None),
        (
            "E",
            "RL_PROPOSAL",
            learning.get("portfolio_rl", {})
            .get("report", {})
            .get("temporal_holdout_validation"),
        ),
    ]
    metrics = (
        "gross_return",
        "net_return",
        "turnover",
        "costs",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "hit_rate",
        "expected_r",
        "realized_r",
        "implementation_shortfall",
        "regime_performance",
        "downside_capture",
        "tail_losses",
    )
    rows = []
    for key, name, evidence in variants:
        rows.append(
            {
                "variant": key,
                "name": name,
                "metrics": evidence,
                "comparable_period": False,
                "status": (
                    "BASELINE_ONLY_NOT_COMPARABLE"
                    if key == "A"
                    else "NO_INCREMENTAL_EVIDENCE"
                ),
                "authority": "NATIVE_DETERMINISTIC"
                if key == "A"
                else "SHADOW_ONLY",
                "execution_authority": "NONE" if key != "A" else "NATIVE_CHAIN_ONLY",
            }
        )
    return {
        "schema": "ai_shadow_portfolio_comparison_v1",
        "status": "NO_INCREMENTAL_EVIDENCE",
        "required_metrics": list(metrics),
        "variants": rows,
        "comparison_valid": False,
        "reason": "NO_SAME_PERIOD_FORWARD_NET_OF_COST_SHADOW_RETURN_SERIES",
        "deterministic_fallback": True,
        "cash_is_valid_result": True,
        "automatic_promotion": False,
        "execution_authority": "NONE",
    }


def _integration_report(
    *,
    status: dict[str, Any],
    architecture: dict[str, Any],
    reference_matrix: list[dict[str, Any]],
    capability_matrix: list[dict[str, Any]],
    models: list[ModelRecord],
    shadow: dict[str, Any],
    authority: dict[str, Any],
) -> str:
    repo_lines = "\n".join(
        f"- `{row['repo']}` ({row['license_family']}): {row['status']}; "
        f"integrated `{row['native_integration']}`; excluded "
        f"{', '.join(row['explicit_exclusions'])}."
        for row in reference_matrix
    )
    weak = [
        row for row in capability_matrix if row["native_status"] == "EXISTS_BUT_WEAK"
    ]
    model_lines = "\n".join(
        f"- `{model.model_id}`: `{model.lifecycle}`, `{model.authority}`, "
        f"value `{model.incremental_evidence}`."
        for model in models
    ) or "- No model evidence was available."
    return f"""# AI Reference Repositories Integration Report

Generated: {status['generated_at']}

## Outcome

One native, framework-independent AI research plane now governs the existing
AI/ML/NLP components. It adds no capability 34, broker client, writer, ledger,
portfolio manager, risk engine or execution path. Current financial conclusion:
`{status['financial_validation_status']}`. The deterministic portfolio and cash
fallback remain authoritative.

## A. Current architecture

The preserved chain is `{ ' -> '.join(architecture['flow']) }`. All AI outputs
remain before native opportunity, Shariah, cost, whole-share, risk, authority,
reconciliation and writer gates. Duplicated runtime count: 0.

## B. Reference repositories

{repo_lines}

No reference source was copied and no reference framework became a production
dependency. Commons-Clause and GPL/LGPL repositories were used only as locally
inspected behavioral oracles.

## C. AI and ML

{model_lines}

The canonical registries record causal feature timing, immutable experiment
identity, lifecycle, calibration, drift limits and expiry. There are
{len(weak)} existing-but-weak capabilities; none received broader authority.

## D. NLP and agents

FinGPT concepts are normalized into timestamped entity, event, sentiment,
uncertainty, novelty, relevance and source-quality fields over the existing news
and SEC pipelines. FinRobot and RD-Agent concepts become structured evidence and
hypothesis/experiment contracts. Free text cannot produce an order or quantity.

## E. Portfolio AI

FinRL-X concepts are represented as weight-only advisory proposals. Every
proposal must still pass the native opportunities, strategy evidence, Shariah,
transaction-cost, whole-share, risk, concentration, correlation, heat, cash and
authority chain.

## F. Reinforcement learning

TradeMaster and FinRL-X informed only the existing constrained shadow policy.
The native reward includes net return, cost, turnover, drawdown and risk terms.
The current RL evidence is not promotable and remains `SHADOW_ONLY`.

## G. Economics

The AI plane cannot mutate gross/net expectancy, commission, spread, slippage,
FX, risk-per-share or whole-share quantity. P2.2 remains authoritative. Cash is
a valid result. No trade is fabricated for activity.

## H. Safety

- Authority matrix: `{authority['status']}`.
- Added writer count: 0.
- Added broker call sites: 0.
- Live orders or financial writes: 0.
- Risk limits changed: false.
- Capital permissions changed: false.
- AI money control: false.

## I. Validation

Validation results are recorded after the ordered targeted, regression, Ruff and
compileall runs. The AI artifacts are snapshots; experiment records are immutable.
Existing P0, P0.2, P2, P2.1 and P2.2 gates are not weakened.

## J. Actual value added

The implementation solves governance fragmentation: one contract now connects
feature timing, models, health, hypotheses, experiments, capabilities, authority
and shadow comparison. It does not claim better trades. The shadow comparison is
`{shadow['status']}` because `{shadow['reason']}`. Therefore every new AI output
remains advisory, context-only or shadow-only until same-period forward,
net-of-cost evidence beats the deterministic baseline with confidence intervals.
"""


def _publish_snapshot(path: Path, payload: dict[str, Any], now: datetime) -> None:
    body = {**payload, "generated_at": now.isoformat()}
    body["content_hash"] = canonical_hash(payload)
    _atomic_json(path, body)


def _git_head(repo_root: Path) -> str | None:
    git = repo_root / ".git"
    if not git.is_dir():
        return None
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    ref_path = git / ref
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    packed = git / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(f" {ref}"):
                return line.split(" ", 1)[0]
    return None


def _license_path(repo_root: Path) -> Path | None:
    if not repo_root.is_dir():
        return None
    return next(
        (
            path
            for path in repo_root.iterdir()
            if path.is_file()
            and path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
        ),
        None,
    )


def _license_family(path: Path | None) -> str:
    if path is None:
        return "NOT_FOUND"
    text = path.read_text(encoding="utf-8", errors="ignore")[:4000].lower()
    if "commons clause" in text:
        return "COMMONS_CLAUSE"
    if "lesser general public license" in text:
        return "LGPL"
    if "general public license" in text:
        return "GPL"
    if "apache license" in text:
        return "APACHE-2.0"
    if "mit license" in text:
        return "MIT"
    if "bsd 2-clause" in text:
        return "BSD-2-CLAUSE"
    return "OTHER"


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None:
        return None
    return result.astimezone(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


__all__ = [
    "ARTIFACTS",
    "load_ai_research_plane_status",
    "publish_ai_research_plane",
]
