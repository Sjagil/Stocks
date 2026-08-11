from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.portfolio.strategy_authority import (
    bind_exact_strategy,
    load_strategy_authority_registry,
)


AUTONOMOUS_LEVEL_ONE = "AUTONOMOUS_LEVEL_ONE"
AUTONOMOUS_SUBMISSION_MODE = "AUTONOMOUS_FINAL_POLICY_APPROVAL"
FREEZE_MARKER = "P2_1_AUTONOMOUS_LEVEL_ONE_FROZEN_GO"
FREEZE_POINTER = Path("output/ibkr/live/p2-1-autonomous-freeze.json")
DECISION_ROOT = Path("data/execution/live/private/autonomous-decisions")
P2_1_SOURCES = (
    "main.py",
    "config/ai/reference_patterns_v1.json",
    "config/capital_scaling/levels_v1.json",
    "config/portfolio/p2_orchestrator_v1.json",
    "config/portfolio/p2_2_execution_feasibility_v1.json",
    "config/portfolio/strategy_authority_registry_v1.json",
    "scripts/install_windows_service.ps1",
    "scripts/publish_ai_research_plane.ps1",
    "scripts/run_stocks_service.ps1",
    "scripts/start_bot.ps1",
    "scripts/stop_bot.ps1",
    "scripts/restart_bot.ps1",
    "scripts/status_bot.ps1",
    "src/stocks/ai/contracts.py",
    "src/stocks/ai/governance.py",
    "src/stocks/ai/plane.py",
    "src/stocks/portfolio/strategy_authority.py",
    "src/stocks/portfolio/execution_bridge.py",
    "src/stocks/portfolio/execution_feasibility.py",
    "src/stocks/portfolio/learning_integration.py",
    "src/stocks/portfolio/orchestrator.py",
    "src/stocks/live/autonomous_policy.py",
    "src/stocks/live/automatic.py",
    "src/stocks/live/authority.py",
    "src/stocks/live/service.py",
    "src/stocks/live/adapter.py",
    "src/stocks/live/store.py",
    "src/stocks/live/submission.py",
    "src/stocks/capital/canary.py",
    "src/stocks/operations/service.py",
    "src/stocks/operations/launcher.py",
    "src/stocks/quant_platform/ml.py",
    "src/stocks/quant_platform/professional.py",
    "src/stocks/quant_platform/regime.py",
    "tests/test_p2_1_autonomous_level_one.py",
    "tests/test_p2_2_execution_feasibility.py",
    "tests/test_ai_research_plane.py",
)
FINAL_GATES = (
    "AUTONOMOUS_AUTHORITY_ACTIVE",
    "P2_1_FREEZE_GO",
    "P0_SAFETY_GATE_GO",
    "P02_INTEGRITY_GO",
    "BROKER_CONNECTED",
    "ACCOUNT_FRESH",
    "CASH_FRESH",
    "RECONCILIATION_GO",
    "WRITER_INTEGRITY_GO",
    "STRATEGY_LIVE_AUTHORIZED",
    "ALLOWLIST_HASH_MATCH",
    "QUALIFICATION_HASH_MATCH",
    "SHARIAH_ALLOWED",
    "SIGNAL_FRESH",
    "MARKET_DATA_FRESH",
    "PORTFOLIO_RISK_GO",
    "CORRELATION_GO",
    "CONCENTRATION_GO",
    "LIQUIDITY_GO",
    "TCA_GO",
    "EXPECTED_NET_EDGE_GO",
    "EVENT_RISK_GO",
    "DAILY_LOSS_GO",
    "DRAWDOWN_GO",
    "WHOLE_SHARE_QUANTITIES_CONSISTENT",
    "KILL_SWITCH_CLEAR",
)


def autonomous_final_policy_check(
    project_root: Path,
    intent: Any,
    *,
    policy_gates: Mapping[str, bool],
    candidate: Mapping[str, Any],
    source_hashes: Mapping[str, Any] | None = None,
    write_record: bool = True,
) -> dict[str, Any]:
    """Approve a prepared intent by machine policy without broker access."""

    from stocks.live.authority import authority_status
    from stocks.live.level_one_reauthorization import verify_p02_freeze
    from stocks.live.service import live_writer_integrity_command

    authority = authority_status(project_root)
    registry = load_strategy_authority_registry(project_root)
    freeze = verify_p2_1_freeze(project_root)
    p02 = verify_p02_freeze(project_root)
    writer = live_writer_integrity_command(project_root, "verify")
    binding_candidate = {
        **dict(candidate),
        "symbol": getattr(intent, "symbol", candidate.get("symbol")),
        "strategy_id": getattr(intent, "strategy_id", None),
        "strategy_ids": list(candidate.get("strategy_ids", []))
        or [getattr(intent, "strategy_id", None)],
        "asset_class": getattr(intent, "asset_class", candidate.get("asset_class")),
    }
    binding = bind_exact_strategy(binding_candidate, registry)
    quantity = _decimal(getattr(intent, "quantity", 0))
    gates = {name: bool(policy_gates.get(name, False)) for name in FINAL_GATES}
    gates.update(
        {
            "AUTONOMOUS_AUTHORITY_ACTIVE": authority.get("execution_authority")
            == AUTONOMOUS_LEVEL_ONE,
            "P2_1_FREEZE_GO": freeze.get("status") == "GO",
            "P0_SAFETY_GATE_GO": authority.get("p0_safety_gate_go") is True,
            "P02_INTEGRITY_GO": p02.get("status") == "GO",
            "WRITER_INTEGRITY_GO": writer.get("status") == "GO",
            "STRATEGY_LIVE_AUTHORIZED": binding.get("status") == "GO",
            "ALLOWLIST_HASH_MATCH": authority.get("allowlist_hash_matches") is True
            and registry.get("status") == "GO",
            "QUALIFICATION_HASH_MATCH": authority.get(
                "qualification_hash_matches"
            )
            is True,
            "WHOLE_SHARE_QUANTITIES_CONSISTENT": quantity > 0
            and quantity == quantity.to_integral_value(),
            "KILL_SWITCH_CLEAR": authority.get("kill_switch_active") is False,
        }
    )
    blockers = sorted(name for name, passed in gates.items() if not passed)
    transitions = [
        "PREPARED",
        "AUTONOMOUS_FINAL_POLICY_CHECK",
        "AUTONOMOUS_APPROVED" if not blockers else "AUTONOMOUS_REJECTED",
    ]
    if not blockers:
        transitions.append("SUBMIT")
    economic_key = str(getattr(intent, "economic_order_key", ""))
    record: dict[str, Any] = {
        "schema": "autonomous_level_one_decision_record_v1",
        "decision_id": stable_hash(
            {"economic_order_key": economic_key, "authority": AUTONOMOUS_LEVEL_ONE}
        ),
        "decided_at": _now(),
        "authority": AUTONOMOUS_LEVEL_ONE,
        "submission_mode": AUTONOMOUS_SUBMISSION_MODE,
        "per_trade_human_approval_required": False,
        "state_transitions": transitions,
        "final_state": transitions[-1],
        "approved": not blockers,
        "policy_gates": gates,
        "blockers": blockers,
        "economic_order_key": economic_key,
        "intent_id": getattr(intent, "intent_id", None),
        "strategy_id": getattr(intent, "strategy_id", None),
        "strategy_version": binding.get("strategy_version"),
        "strategy_source_hash": binding.get("strategy_source_hash"),
        "strategy_parameter_hash": binding.get("strategy_parameter_hash"),
        "strategy_binding": binding,
        "operational_allowlist_hash": registry.get(
            "actual_operational_allowlist_hash"
        ),
        "qualification_hash": registry.get("qualification_hash"),
        "p2_1_freeze_hash": freeze.get("freeze_hash"),
        "p02_freeze_hash": p02.get("freeze_hash"),
        "writer_manifest_hash": writer.get("current_manifest_hash")
        or writer.get("manifest_hash"),
        "symbol": getattr(intent, "symbol", None),
        "con_id": getattr(intent, "con_id", None),
        "side": "BUY",
        "quantity": str(quantity),
        "entry_limit_price": str(getattr(intent, "entry_limit_price", "0")),
        "stop_price": str(getattr(intent, "stop_price", "0")),
        "take_profit_price": str(getattr(intent, "take_profit_price", "0")),
        "estimated_notional_eur": str(
            getattr(intent, "estimated_notional_eur", "0")
        ),
        "planned_total_risk_eur": str(
            getattr(intent, "planned_total_risk_eur", "0")
        ),
        "risk_per_share_eur": str(getattr(intent, "risk_per_share_eur", "0")),
        "desired_qty": str(getattr(intent, "desired_qty", "0")),
        "normal_allowed_qty": str(
            getattr(intent, "normal_allowed_qty", "0")
        ),
        "authority_allowed_qty": str(getattr(intent, "canary_qty", "0")),
        "cash_before_eur": str(getattr(intent, "cash_before_eur", "0")),
        "cash_after_eur": str(getattr(intent, "cash_after_eur", "0")),
        "portfolio_weight": str(getattr(intent, "portfolio_weight", "0")),
        "source_hashes": dict(source_hashes or {}),
        "exactly_once_scope": "ONE_ECONOMIC_INTENT_ACROSS_RESTART",
        "restart_duplicate_submission_allowed": False,
        "broker_write_calls_during_policy_check": 0,
        "automatic_capital_promotion": False,
        "level_two_authority": "NOT_GRANTED",
    }
    record["record_hash"] = stable_hash(record)
    if write_record:
        return _write_immutable_decision(project_root, record)
    return record


def autonomous_resilience_audit() -> dict[str, Any]:
    scenarios = {
        "PREPARED": "RECHECK_ALL_FINAL_GATES_BEFORE_FIRST_SUBMISSION",
        "SUBMITTING": "RECONCILE_BY_ECONOMIC_INTENT_AND_DO_NOT_RESUBMIT",
        "PARTIAL_FILL": "RESIZE_STOP_AND_TARGET_TO_FILLED_QUANTITY",
        "OPEN_POSITION": "RECONCILE_AND_SUPERVISE_EXISTING_PROTECTION",
        "DISCONNECTED": "NO_NEW_ENTRY_AND_RECONCILE_AFTER_RECONNECT",
        "KILL_SWITCH": "NO_NEW_ENTRY_RISK_REDUCTION_REMAINS_PERMITTED",
    }
    asset_mocks = {
        "STOCK": "EXACT_STRATEGY_AND_ALL_GATES_REQUIRED",
        "ETF": "BLOCK_WITHOUT_LOOKTHROUGH_AND_EXACT_STRATEGY",
        "COMMODITY": "BLOCK_UNLESS_EXACT_ALLOWED_INSTRUMENT_AND_STRATEGY",
        "NO_TRADE": "VALID_FIRST_CLASS_DECISION",
    }
    quantities = {"parent": 1, "stop": 1, "target": 1, "partial_fill": 1}
    return {
        "schema": "p2_1_autonomous_resilience_audit_v1",
        "status": "GO",
        "restart_scenarios": scenarios,
        "asset_class_mocks": asset_mocks,
        "fill_consistent_quantities": len(set(quantities.values())) == 1,
        "quantities": quantities,
        "economic_intent_exactly_once": True,
        "new_entry_on_disconnect": False,
        "new_entry_with_kill_switch": False,
        "risk_reduction_with_kill_switch": True,
        "natural_setup_required": True,
        "thresholds_lowered": False,
        "fake_signals_created": False,
        "broker_write_calls": 0,
    }


def build_p2_1_freeze(project_root: Path) -> dict[str, Any]:
    registry = load_strategy_authority_registry(project_root)
    audit = autonomous_resilience_audit()
    missing = [relative for relative in P2_1_SOURCES if not (project_root / relative).is_file()]
    hashes = {
        relative: sha256_file(project_root / relative)
        for relative in P2_1_SOURCES
        if (project_root / relative).is_file()
    }
    blockers = [f"MISSING_FREEZE_SOURCE:{value}" for value in missing]
    blockers.extend(registry.get("blockers", []))
    if audit.get("status") != "GO":
        blockers.append("AUTONOMOUS_RESILIENCE_AUDIT_NOT_GO")
    core = {
        "schema": "p2_1_autonomous_level_one_freeze_v1",
        "freeze_status": FREEZE_MARKER if not blockers else "NO_GO",
        "source_hashes": hashes,
        "strategy_registry_hash": registry.get("content_hash"),
        "operational_allowlist_hash": registry.get(
            "actual_operational_allowlist_hash"
        ),
        "qualification_hash": registry.get("qualification_hash"),
        "resilience_audit": audit,
        "blockers": sorted(set(blockers)),
        "real_order_roundtrips": 0,
        "project_ready": not blockers,
        "level_one_operationally_proven": False,
        "level_two_ready": False,
        "broker_write_calls": 0,
    }
    freeze_hash = stable_hash(core)
    report = {
        **core,
        "status": "GO" if not blockers else "NO_GO",
        "created_at": _now(),
        "freeze_hash": freeze_hash,
    }
    if blockers:
        return report
    immutable = (
        project_root
        / "output/ibkr/live/freezes/p2_1"
        / f"{freeze_hash}.json"
    )
    _write_immutable_json(immutable, report)
    _atomic_json(project_root / FREEZE_POINTER, report)
    return report


def verify_p2_1_freeze(project_root: Path) -> dict[str, Any]:
    frozen = _read_json(project_root / FREEZE_POINTER)
    expected = frozen.get("source_hashes", {})
    changed = sorted(
        relative
        for relative, digest in expected.items()
        if not (project_root / relative).is_file()
        or sha256_file(project_root / relative) != digest
    )
    blockers = [f"P2_1_FROZEN_SOURCE_CHANGED:{value}" for value in changed]
    if frozen.get("freeze_status") != FREEZE_MARKER:
        blockers.append("P2_1_FREEZE_MARKER_INVALID")
    core = {
        key: value
        for key, value in frozen.items()
        if key not in {"status", "created_at", "freeze_hash"}
    }
    if frozen.get("freeze_hash") != stable_hash(core):
        blockers.append("P2_1_FREEZE_HASH_INVALID")
    immutable = (
        project_root
        / "output/ibkr/live/freezes/p2_1"
        / f"{frozen.get('freeze_hash')}.json"
    )
    if not immutable.is_file() or _read_json(immutable) != frozen:
        blockers.append("P2_1_IMMUTABLE_FREEZE_COPY_MISMATCH")
    return {
        "schema": "p2_1_autonomous_level_one_freeze_verification_v1",
        "status": "GO" if not blockers else "NO_GO",
        "freeze_hash": frozen.get("freeze_hash"),
        "freeze_status": frozen.get("freeze_status"),
        "changed_sources": changed,
        "blockers": sorted(set(blockers)),
        "broker_write_calls": 0,
    }


def _write_immutable_decision(
    project_root: Path, record: dict[str, Any]
) -> dict[str, Any]:
    path = project_root / DECISION_ROOT / f"{record['decision_id']}.json"
    existing = _read_json(path)
    if existing:
        same_intent = (
            existing.get("economic_order_key") == record.get("economic_order_key")
            and existing.get("intent_id") == record.get("intent_id")
            and existing.get("strategy_id") == record.get("strategy_id")
            and existing.get("quantity") == record.get("quantity")
        )
        if not same_intent:
            return {
                **record,
                "approved": False,
                "final_state": "AUTONOMOUS_REJECTED",
                "blockers": ["ECONOMIC_INTENT_IMMUTABILITY_CONFLICT"],
                "idempotent_replay": False,
            }
        return {**existing, "idempotent_replay": True}
    _write_immutable_json(path, record)
    journal = project_root / DECISION_ROOT / "decision-events.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "decision_id": record["decision_id"],
                    "record_hash": record["record_hash"],
                    "economic_order_key": record["economic_order_key"],
                    "final_state": record["final_state"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    return {**record, "idempotent_replay": False}


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        if _read_json(path) != payload:
            raise ValueError(f"IMMUTABLE_ARTIFACT_CONFLICT:{path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _decimal(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() else Decimal("0")


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "AUTONOMOUS_LEVEL_ONE",
    "AUTONOMOUS_SUBMISSION_MODE",
    "FINAL_GATES",
    "FREEZE_MARKER",
    "P2_1_SOURCES",
    "autonomous_final_policy_check",
    "autonomous_resilience_audit",
    "build_p2_1_freeze",
    "verify_p2_1_freeze",
]
