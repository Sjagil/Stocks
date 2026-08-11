from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from stocks.capital.canary import load_level_one_canary_policy
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.p0_readiness import inspect_p0_execution_readiness_gate
from stocks.live.integrity import normalized_file_hash


P02_VERSION = "P0.2_LEVEL1_REAUTHORIZATION_V1"
PREPARE_TTL_SECONDS = 600
MAX_PREPARE_PRICE_DRIFT = Decimal("0.01")
AUTHORITATIVE_SOURCES = (
    "main.py",
    "config/capital_scaling/levels_v1.json",
    "src/stocks/capital/canary.py",
    "src/stocks/capital/service.py",
    "src/stocks/portfolio/manager.py",
    "src/stocks/portfolio/dynamic_risk.py",
    "src/stocks/portfolio/targets.py",
    "src/stocks/portfolio/strategy_authority.py",
    "config/portfolio/strategy_authority_registry_v1.json",
    "scripts/install_windows_service.ps1",
    "scripts/run_stocks_service.ps1",
    "scripts/start_bot.ps1",
    "scripts/stop_bot.ps1",
    "scripts/restart_bot.ps1",
    "scripts/status_bot.ps1",
    "src/stocks/live/config.py",
    "src/stocks/live/level_one_reauthorization.py",
    "src/stocks/live/models.py",
    "src/stocks/live/authority.py",
    "src/stocks/live/automatic.py",
    "src/stocks/live/autonomous_policy.py",
    "src/stocks/live/service.py",
    "src/stocks/live/portfolio_targets.py",
    "src/stocks/live/approvals.py",
    "src/stocks/live/adapter.py",
    "src/stocks/live/submission.py",
    "src/stocks/operations/service.py",
    "src/stocks/operations/launcher.py",
)


def policy_integrity(project_root: Path) -> dict[str, Any]:
    artifact = _read_json(
        project_root / "output/ibkr/live/whole-share-canary-policy-v1.json"
    )
    blockers: list[str] = []
    try:
        configured = load_level_one_canary_policy(project_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        configured = None
        blockers.append("WHOLE_SHARE_POLICY_CONFIG_INVALID")
    content = {
        key: value for key, value in artifact.items() if key != "content_hash"
    }
    if not artifact or artifact.get("content_hash") != stable_hash(content):
        blockers.append("WHOLE_SHARE_POLICY_ARTIFACT_HASH_MISMATCH")
    if configured is not None:
        expected_policy_hash = stable_hash(
            {
                "configured_policy": configured.jsonable(),
                "resolved_limits": artifact.get("level_one_limits", {}),
            }
        )
        checks = {
            "policy_version": artifact.get("policy_version")
            == configured.policy_version,
            "policy_hash": artifact.get("policy_hash")
            == expected_policy_hash,
            "fractional_forbidden": artifact.get(
                "fractional_shares_allowed"
            )
            is False,
            "whole_share_required": artifact.get("whole_share_required")
            is True,
            "risk_first": artifact.get("primary_sizing_authority")
            == "RISK_PER_WHOLE_SHARE",
            "notional_secondary": artifact.get("notional_cap_role")
            == "SECONDARY_EMERGENCY_BACKSTOP",
            "level_one": artifact.get("capital_level") == 1,
            "five_roundtrips": artifact.get("promotion_requirements", {}).get(
                "minimum_verified_whole_share_round_trips"
            )
            == 5,
        }
        blockers.extend(
            f"POLICY_{name.upper()}_MISMATCH"
            for name, passed in checks.items()
            if not passed
        )
    else:
        checks = {}
        expected_policy_hash = None
    return {
        "schema": "p02_whole_share_policy_integrity_v1",
        "status": "GO" if not blockers else "NO_GO",
        "policy_version": artifact.get("policy_version"),
        "policy_hash": artifact.get("policy_hash"),
        "expected_policy_hash": expected_policy_hash,
        "checks": checks,
        "blockers": sorted(set(blockers)),
    }


def build_p02_freeze(project_root: Path) -> dict[str, Any]:
    policy = policy_integrity(project_root)
    prior = _read_json(project_root / "output/ibkr/live/freeze-status.json")
    blockers = list(policy["blockers"])
    prior_hashes = prior.get("source_hashes", {})
    for relative, expected in prior_hashes.items():
        path = project_root / relative
        if not path.exists() or normalized_file_hash(path).lower() != str(expected).lower():
            blockers.append("WRITER_FREEZE_MISMATCH")
            break
    source_hashes: dict[str, str] = {}
    for relative in AUTHORITATIVE_SOURCES:
        path = project_root / relative
        if not path.exists():
            blockers.append(f"AUTHORITATIVE_SOURCE_MISSING:{relative}")
        else:
            source_hashes[relative] = normalized_file_hash(path).upper()
    body: dict[str, Any] = {
        "schema": "p02_level_one_whole_share_freeze_v1",
        "status": "GO" if not blockers else "NO_GO",
        "p0_version": "IBKR_EXECUTION_P0_SAFETY_MATRIX_V1",
        "p01_version": policy.get("policy_version"),
        "p02_version": P02_VERSION,
        "policy_hash": policy.get("policy_hash"),
        "prior_writer_manifest_hash": prior.get("manifest_hash"),
        "source_hashes": source_hashes,
        "config_hashes": {
            path: digest
            for path, digest in source_hashes.items()
            if path.startswith("config/")
        },
        "account_fingerprint_requirement": "EXACT_CONFIGURED_MATCH",
        "capital_level": 1,
        "capital_level_name": "WHOLE_SHARE_EXECUTION_CANARY",
        "whole_share_only": True,
        "fractional_allowed": False,
        "risk_first": True,
        "manual_approval_required": True,
        "automatic_submission": False,
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "broker_writes": 0,
    }
    body["freeze_hash"] = stable_hash(body)
    artifact = (
        project_root
        / "output/ibkr/live/p0-2-freezes"
        / f"level-one-{body['freeze_hash']}.json"
    )
    if artifact.exists() and _read_json(artifact) != body:
        raise ValueError("IMMUTABLE_P02_FREEZE_CONFLICT")
    _write_json(artifact, body, overwrite=False)
    index = {
        "schema": "p02_level_one_freeze_index_v1",
        "status": body["status"],
        "artifact_path": artifact.relative_to(project_root).as_posix(),
        "freeze_hash": body["freeze_hash"],
        "policy_hash": body["policy_hash"],
        "blockers": body["blockers"],
    }
    _write_json(
        project_root / "output/ibkr/live/p0-2-level-one-freeze.json", index
    )
    return {**body, "artifact_path": index["artifact_path"]}


def verify_p02_freeze(project_root: Path) -> dict[str, Any]:
    index = _read_json(
        project_root / "output/ibkr/live/p0-2-level-one-freeze.json"
    )
    artifact = _read_json(project_root / str(index.get("artifact_path", "")))
    blockers: list[str] = []
    if not artifact or artifact.get("freeze_hash") != index.get("freeze_hash"):
        blockers.append("P02_FREEZE_MISSING_OR_INDEX_MISMATCH")
    frozen_body = {
        key: value for key, value in artifact.items() if key != "freeze_hash"
    }
    if artifact and artifact.get("freeze_hash") != stable_hash(frozen_body):
        blockers.append("P02_FREEZE_HASH_MISMATCH")
    for relative, expected in artifact.get("source_hashes", {}).items():
        path = project_root / relative
        if not path.exists() or normalized_file_hash(path).upper() != expected:
            blockers.append("WRITER_FREEZE_MISMATCH")
            break
    policy = policy_integrity(project_root)
    if policy["status"] != "GO" or artifact.get("policy_hash") != policy.get(
        "policy_hash"
    ):
        blockers.append("POLICY_HASH_MISMATCH")
    return {
        "schema": "p02_level_one_freeze_verification_v1",
        "status": "GO" if not blockers else "NO_GO",
        "freeze_hash": artifact.get("freeze_hash"),
        "policy_hash": artifact.get("policy_hash"),
        "artifact_path": index.get("artifact_path"),
        "blockers": sorted(set(blockers)),
    }


def build_prepare_artifact(
    candidate: dict[str, Any],
    *,
    bindings: dict[str, str],
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    blockers: list[str] = []
    quantity = _decimal(candidate.get("canary_qty"))
    if quantity < 1 or quantity != quantity.to_integral_value():
        blockers.append("WHOLE_SHARE_CANARY_REQUIRED")
    if candidate.get("fractional_allowed") is not False:
        blockers.append("FRACTIONAL_POLICY_BINDING_REQUIRED")
    for field in (
        "candidate_id",
        "strategy_id",
        "symbol",
        "con_id",
        "asset_class",
        "entry_price",
        "stop_price",
        "target_price",
        "account_snapshot_hash",
        "market_data_timestamp",
    ):
        if candidate.get(field) in (None, ""):
            blockers.append(f"{field.upper()}_REQUIRED")
    for gate in (
        "strategy_authorized",
        "shariah_allowed",
        "contract_resolved",
        "economics_go",
        "risk_go",
        "liquidity_go",
    ):
        if candidate.get(gate) is not True:
            blockers.append(f"{gate.upper()}_REQUIRED")
    if blockers:
        return {
            "schema": "level1_canary_prepare_v1",
            "status": "NO_GO",
            "state": "CANDIDATE",
            "blockers": sorted(set(blockers)),
            "transmission_allowed_by_this_artifact": False,
        }
    artifact = {
        "schema": "level1_canary_prepare_v1",
        "status": "GO",
        "state": "PREPARED",
        "prepared_at": current.isoformat(),
        "expires_at": (current + timedelta(seconds=PREPARE_TTL_SECONDS)).isoformat(),
        "max_prepare_price_drift": str(MAX_PREPARE_PRICE_DRIFT),
        **candidate,
        "parent_qty": str(quantity),
        "stop_qty": str(quantity),
        "target_qty": str(quantity),
        "authority_level": "LIVE_LEVEL_ONE_MANUAL",
        "p0_hash": bindings["p0_hash"],
        "writer_freeze_hash": bindings["writer_freeze_hash"],
        "policy_hash": bindings["policy_hash"],
        "approval_required": True,
        "transmission_allowed_by_this_artifact": False,
        "intent_created": False,
        "submitted": False,
        "broker_writes": 0,
        "blockers": [],
    }
    artifact["prepare_hash"] = stable_hash(artifact)
    return artifact


def rank_manual_review_candidates(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rank gated cross-asset candidates; share price is feasibility only."""

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        asset_class = str(candidate.get("asset_class", "")).upper()
        quantity = _decimal(candidate.get("canary_qty"))
        gates = all(
            candidate.get(name) is True
            for name in (
                "strategy_authorized",
                "shariah_allowed",
                "contract_resolved",
                "economics_go",
                "risk_go",
                "liquidity_go",
            )
        )
        if (
            asset_class not in {"STOCK", "ETF", "COMMODITY_VEHICLE"}
            or not gates
            or quantity < 1
            or quantity != quantity.to_integral_value()
            or candidate.get("fractional_allowed") is not False
        ):
            continue
        expected_net = _decimal(candidate.get("expected_net_return"))
        score = (
            expected_net * Decimal("0.30")
            + _decimal(candidate.get("risk_adjusted_opportunity"))
            * Decimal("0.25")
            + _decimal(candidate.get("validation_quality"))
            * Decimal("0.15")
            + _decimal(candidate.get("diversification")) * Decimal("0.10")
            + _decimal(candidate.get("liquidity")) * Decimal("0.10")
            + _decimal(candidate.get("regime_fit")) * Decimal("0.10")
            - _decimal(candidate.get("event_risk")) * Decimal("0.10")
        )
        if expected_net > 0 and score > 0:
            ranked.append({**candidate, "cross_asset_rank_score": str(score)})
    ranked.sort(
        key=lambda row: (
            _decimal(row["cross_asset_rank_score"]),
            _decimal(row.get("expected_net_return")),
            str(row.get("candidate_id", "")),
        ),
        reverse=True,
    )
    winner = ranked[0] if ranked else None
    return {
        "schema": "level1_cross_asset_candidate_ranking_v1",
        "status": "GO",
        "selected_action": "MANUAL_REVIEW_READY" if winner else "NO_TRADE",
        "cash_competes": True,
        "selected_candidate": winner,
        "ranked_candidates": ranked,
        "share_price_used_for_ranking": False,
    }


def publish_current_multi_asset_funnel(project_root: Path) -> dict[str, Any]:
    ranking = _read_json(project_root / "output/portfolio/opportunity_ranking.json")
    funnel = _read_json(project_root / "output/portfolio/opportunity-funnel.json")
    coverage = _read_json(project_root / "output/portfolio/coverage-waterfall.json")
    private = _read_json(project_root / "data/portfolio/private/current-state.json")
    opportunities = [
        row for row in ranking.get("opportunities", []) if isinstance(row, dict)
    ]
    sizing_rows = private.get("whole_share_sizing", {}).get(
        "candidate_preflight", {}
    ).get("candidate_results", [])
    classes = ("STOCK", "ETF", "COMMODITY_VEHICLE")
    coverage_keys = {
        "STOCK": "EQUITY",
        "ETF": "ETF",
        "COMMODITY_VEHICLE": "COMMODITY_EXPOSURE",
    }
    class_rows: dict[str, dict[str, int]] = {}
    for asset_class in classes:
        rows = [
            row
            for row in opportunities
            if _asset_class(row.get("asset_type")) == asset_class
        ]
        sized = [
            row
            for row in sizing_rows
            if str(row.get("asset_class", "")).upper() == asset_class
        ]
        stages = coverage.get("funnels", {}).get(
            coverage_keys[asset_class], {}
        ).get("stages", {})
        class_rows[asset_class] = {
            "universe": int(stages.get("discovered", 0)),
            "analyzable": int(stages.get("data_analyzable", 0)),
            "research_eligible": int(stages.get("research_eligible", 0)),
            "shariah_allowed": int(stages.get("shariah_eligible", 0)),
            "broker_resolvable": int(stages.get("broker_resolvable", 0)),
            "strategy_signal": int(stages.get("strategy_signal", 0)),
            "qualified": int(stages.get("qualified", 0)),
            "ranked": int(stages.get("ranked", len(rows))),
            "portfolio_candidate": int(stages.get("portfolio_candidate", 0)),
            "normal_whole_share_feasible": sum(
                _decimal(row.get("normal_allowed_qty")) >= 1 for row in sized
            ),
            "level1_canary_feasible": sum(
                _decimal(row.get("level1_canary_qty")) >= 1 for row in sized
            ),
            "manual_review_ready": 0,
        }
    top = []
    for row in opportunities[:10]:
        blockers = sorted(
            set(
                list(row.get("deployment_blockers") or [])
                + list(row.get("execution_blockers") or [])
            )
        )
        top.append(
            {
                "symbol": row.get("ticker"),
                "asset_class": _asset_class(row.get("asset_type")),
                "strategy_ids": row.get("strategy_ids", []),
                "strategy_status": (
                    "LIVE_ELIGIBLE"
                    if row.get("deployment_eligible") is True
                    else "RESEARCH_ONLY"
                ),
                "score": row.get("opportunity_score"),
                "confidence": row.get("components", {}).get("signal_quality"),
                "regime_fit": row.get("components", {}).get("regime_fit"),
                "event_risk": row.get("event_risk"),
                "shariah_status": row.get("shariah_status"),
                "contract_resolved": row.get("contract_resolved"),
                "blocking_reason": blockers,
            }
        )
    report = {
        "schema": "p02_current_multi_asset_funnel_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_generated_at": funnel.get("generated_at"),
        "universe": funnel.get("universe_instrument_count", 0),
        "analyzable": funnel.get("analyzable_instrument_count", 0),
        "strategy_signal": funnel.get("current_signal_count", 0),
        "qualified": funnel.get("qualified_screener_record_count", 0),
        "ranked": funnel.get("ranked_opportunity_count", len(opportunities)),
        "portfolio_candidate": funnel.get("portfolio_candidate_count", 0),
        "execution_candidate": funnel.get("execution_candidate_count", 0),
        "by_asset_class": class_rows,
        "top_current_opportunities": top,
        "coverage_source": coverage.get("schema"),
        "cash_competes": True,
        "natural_setup_available": False,
        "manual_review_ready": False,
        "selected_action": "NO_TRADE",
        "prepare_artifact": None,
        "orders_generated": 0,
        "broker_writes": 0,
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(report)
    _write_json(project_root / "output/ibkr/live/p0-2-multi-asset-funnel.json", report)
    return report


def validate_prepare_artifact(
    prepared: dict[str, Any],
    *,
    current_price: Decimal,
    current_stop: Decimal,
    current_account_snapshot_hash: str,
    current_account_fingerprint: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    blockers: list[str] = []
    try:
        expires = datetime.fromisoformat(str(prepared["expires_at"]))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
    except (KeyError, TypeError, ValueError):
        expires = current
    if current >= expires:
        blockers.append("PREPARE_EXPIRED")
    entry = _decimal(prepared.get("entry_price"))
    if entry <= 0 or abs(current_price - entry) / entry > MAX_PREPARE_PRICE_DRIFT:
        blockers.append("REPREPARE_REQUIRED_PRICE_DRIFT")
    if current_stop != _decimal(prepared.get("stop_price")):
        blockers.append("REPREPARE_REQUIRED_STOP_DRIFT")
    if current_account_snapshot_hash != prepared.get("account_snapshot_hash"):
        blockers.append("REVALIDATION_REQUIRED_ACCOUNT_DRIFT")
    if current_account_fingerprint != prepared.get("account_fingerprint"):
        blockers.append("ACCOUNT_FINGERPRINT_MISMATCH")
    quantity = _decimal(prepared.get("canary_qty"))
    quantities = [_decimal(prepared.get(key)) for key in ("parent_qty", "stop_qty", "target_qty")]
    if (
        quantity < 1
        or quantity != quantity.to_integral_value()
        or any(value != quantity for value in quantities)
        or prepared.get("fractional_allowed") is not False
    ):
        blockers.append("WHOLE_SHARE_FINAL_CHECK_FAILED")
    return {
        "schema": "level1_canary_prepare_validation_v1",
        "status": "GO" if not blockers else "NO_GO",
        "blockers": sorted(set(blockers)),
        "transmission_allowed": False,
        "broker_writes": 0,
    }


def publish_p02_readiness(project_root: Path) -> dict[str, Any]:
    policy = policy_integrity(project_root)
    freeze = verify_p02_freeze(project_root)
    p0 = inspect_p0_execution_readiness_gate(project_root)
    reconciliation = _read_json(project_root / "output/ibkr/live/reconciliation.json")
    authority = _read_json(project_root / "output/ibkr/live/authority-status.json")
    blockers = [
        *(policy["blockers"] if policy["status"] != "GO" else []),
        *(freeze["blockers"] if freeze["status"] != "GO" else []),
        *(p0.get("blockers", []) if p0.get("status") != "GO" else []),
    ]
    if reconciliation.get("status") != "GO":
        blockers.append("LIVE_RECONCILIATION_REQUIRED")
    report = {
        "schema": "p02_level_one_reauthorization_readiness_v1",
        "status": "GO" if not blockers else "NO_GO",
        "technical_state": "P0_TECHNICAL_READY" if not blockers else "BLOCKED",
        "level1_authorized": authority.get("execution_authority")
        == "LIVE_LEVEL_ONE",
        "authority_binding": {
            "active_status": authority.get("execution_authority")
            == "LIVE_LEVEL_ONE",
            "execution_authority": authority.get("execution_authority", "NONE"),
            "capital_level": authority.get("capital_level", 0),
            "capital_level_name": "WHOLE_SHARE_EXECUTION_CANARY",
            "activation_timestamp": authority.get("activated_at"),
            "account_fingerprint_masked": authority.get(
                "account_fingerprint_masked"
            ),
            "p0_hash": p0.get("attestation_hash"),
            "writer_freeze_hash": freeze.get("freeze_hash"),
            "policy_hash": policy.get("policy_hash"),
            "whole_share_only": True,
            "fractional_allowed": False,
            "risk_first": True,
            "canary_downscale_allowed": True,
            "minimum_quantity": 1,
            "manual_approval_required": True,
            "automatic_submission": False,
            "limits": authority.get("limits", {}),
        },
        "natural_setup_available": False,
        "manual_review_ready": False,
        "order_approved": False,
        "order_submitted": False,
        "policy": policy,
        "freeze": freeze,
        "p0_status": p0.get("status"),
        "reconciliation_status": reconciliation.get("reconciliation_status"),
        "blockers": sorted(set(blockers)),
        "orders_submitted": 0,
        "orders_cancelled": 0,
        "orders_modified": 0,
        "fx_transactions": 0,
        "level_two_promotion": False,
    }
    report["content_hash"] = stable_hash(report)
    _write_json(project_root / "output/ibkr/live/p0-2-readiness.json", report)
    _write_json(
        project_root / "output/ibkr/live/p0-2-level-one-authority.json",
        report["authority_binding"],
    )
    return report


def _decimal(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() else Decimal("0")


def _asset_class(value: Any) -> str:
    normalized = str(value or "").upper()
    if normalized == "STOCK":
        return "STOCK"
    if normalized in {"ETF", "FUND"}:
        return "ETF"
    if normalized in {
        "COMMODITY",
        "COMMODITY_VEHICLE",
        "REAL_ASSET",
        "REAL_ASSET_VEHICLE",
    }:
        return "COMMODITY_VEHICLE"
    return normalized or "UNKNOWN"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


__all__ = [
    "AUTHORITATIVE_SOURCES",
    "MAX_PREPARE_PRICE_DRIFT",
    "P02_VERSION",
    "PREPARE_TTL_SECONDS",
    "build_p02_freeze",
    "build_prepare_artifact",
    "policy_integrity",
    "publish_current_multi_asset_funnel",
    "publish_p02_readiness",
    "rank_manual_review_candidates",
    "validate_prepare_artifact",
    "verify_p02_freeze",
]
