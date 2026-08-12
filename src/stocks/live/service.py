from __future__ import annotations

import json
import socket
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import dotenv_values

from stocks.capital.canary import (
    default_level_one_canary_policy,
    evaluate_whole_share_canary,
    load_level_one_canary_policy,
)
from stocks.capital.service import daily_profit_target
from stocks.data.phase5_common import sha256_file
from stocks.domain.assets import IbkrSecurityType
from stocks.execution.idempotency import economic_order_key
from stocks.ibkr.callbacks import CallbackState
from stocks.ibkr.contract_cache import contract_cache_ttl
from stocks.ibkr.paper_execution.order_ids import allocate_order_id
from stocks.ibkr.paper_execution import PHASE9_FREEZE_MARKER, PHASE9_MARKER
from stocks.ibkr.p0_readiness import (
    P0_READINESS_BLOCKER,
    inspect_p0_execution_readiness_gate,
    write_p0_execution_readiness,
)
from stocks.ibkr.reconciliation.requests import Phase8Config
from stocks.ibkr.reconciliation.account_state import (
    derive_economic_account_state,
)
from stocks.ibkr.reconciliation.snapshots import capture_snapshot
from stocks.ibkr.reconciliation.storage import (
    BrokerObservationStore,
    public_snapshot_summary,
)
from stocks.live.adapter import (
    LiveCanaryApp,
    build_bracket_orders,
    build_stock_contract,
    validate_whole_share_intent,
)
from stocks.live.approvals import (
    approval_challenge as live_approval_challenge,
)
from stocks.live.approvals import approve as approve_live_intent
from stocks.live.approvals import consume as consume_live_approval
from stocks.live.authority import authority_status
from stocks.live.config import load_live_canary_config
from stocks.live.integrity import (
    freeze_manifest,
    inspect_manifest,
    verify_manifest,
)
from stocks.live.models import LiveCanaryConfig, ManualLiveBracketIntent
from stocks.live.store import LiveExecutionStore
from stocks.live.submission import submit_bracket_once
from stocks.research.autopilot.contracts import stable_hash
from stocks.universe import broad_asset_metadata


LIVE_PORTS = {7496, 4001}
LIVE_WRITER_FREEZE_MARKER = "LIVE_CANARY_WRITER_OFFLINE_FROZEN_GO"
LIVE_WRITER_SOURCES = (
    "main.py",
    "config/operations/autonomous_multi_asset_v1.json",
    "config/portfolio/strategy_authority_registry_v1.json",
    "config/research/evidence_throughput_v1.json",
    "scripts/install_windows_service.ps1",
    "scripts/run_stocks_service.ps1",
    "scripts/start_bot.ps1",
    "scripts/stop_bot.ps1",
    "scripts/restart_bot.ps1",
    "scripts/status_bot.ps1",
    "src/stocks/operations/background_jobs.py",
    "src/stocks/operations/primary_refresh.py",
    "src/stocks/ibkr/paper_execution/adapter.py",
    "src/stocks/ibkr/paper_execution/approvals.py",
    "src/stocks/ibkr/paper_execution/audit.py",
    "src/stocks/ibkr/paper_execution/authority.py",
    "src/stocks/ibkr/paper_execution/callbacks.py",
    "src/stocks/ibkr/paper_execution/canary_a_evidence.py",
    "src/stocks/ibkr/paper_execution/canary_b_evidence.py",
    "src/stocks/ibkr/paper_execution/cancellation.py",
    "src/stocks/ibkr/paper_execution/commissions.py",
    "src/stocks/ibkr/paper_execution/config.py",
    "src/stocks/ibkr/paper_execution/errors.py",
    "src/stocks/ibkr/paper_execution/executions.py",
    "src/stocks/ibkr/paper_execution/historical_quarantine.py",
    "src/stocks/ibkr/paper_execution/known_fill.py",
    "src/stocks/ibkr/paper_execution/models.py",
    "src/stocks/ibkr/paper_execution/order_ids.py",
    "src/stocks/ibkr/paper_execution/reconciliation.py",
    "src/stocks/ibkr/paper_execution/restart_recovery.py",
    "src/stocks/ibkr/paper_execution/risk.py",
    "src/stocks/ibkr/paper_execution/state_machine.py",
    "src/stocks/ibkr/paper_execution/state_mapping.py",
    "src/stocks/ibkr/paper_execution/storage.py",
    "src/stocks/ibkr/paper_execution/submission.py",
    "src/stocks/ibkr/contract_cache.py",
    "src/stocks/ibkr/contract_resolver.py",
    "src/stocks/ibkr/contracts.py",
    "src/stocks/ibkr/live_contract_refresh.py",
    "src/stocks/live/automatic.py",
    "src/stocks/live/autonomous_policy.py",
    "src/stocks/live/authority.py",
    "src/stocks/live/adapter.py",
    "src/stocks/live/approvals.py",
    "src/stocks/live/config.py",
    "src/stocks/live/evidence.py",
    "src/stocks/live/integrity.py",
    "src/stocks/live/level_one_reauthorization.py",
    "src/stocks/live/models.py",
    "src/stocks/live/portfolio_targets.py",
    "src/stocks/live/quote.py",
    "src/stocks/live/service.py",
    "src/stocks/live/store.py",
    "src/stocks/live/submission.py",
    "src/stocks/capital/service.py",
    "src/stocks/capital/canary.py",
    "src/stocks/portfolio/dynamic_risk.py",
    "src/stocks/portfolio/strategy_authority.py",
    "src/stocks/portfolio/targets.py",
    "config/capital_scaling/levels_v1.json",
    "config/portfolio/active_manager_v1.json",
    "src/stocks/operations/launcher.py",
    "src/stocks/operations/manual_positions.py",
    "src/stocks/operations/service.py",
    "src/stocks/research/evidence_throughput.py",
    "src/stocks/signals/storage.py",
    "tests/test_live_authority.py",
    "tests/test_live_canary_preflight.py",
    "tests/test_live_capability_commands.py",
    "tests/test_live_contract_refresh.py",
    "tests/test_live_writer.py",
    "tests/test_live_level_two_evidence.py",
    "tests/test_level_one_reauthorization.py",
    "tests/test_whole_share_canary_policy.py",
    "tests/test_evidence_throughput.py",
    "tests/test_live_automatic_cycle.py",
    "tests/test_p2_1_autonomous_level_one.py",
    "tests/test_manual_position_lifecycle.py",
    "tests/test_manual_position_broker_match.py",
    "tests/test_operations_launcher.py",
    "tests/test_no_order_authority.py",
    "tests/test_phase9_canary_a_evidence.py",
    "tests/test_phase9_canary_b_evidence.py",
    "tests/test_phase9_capital_reservations.py",
    "tests/test_phase9_fill_close_reconciliation.py",
    "tests/test_phase9_historical_quarantine.py",
    "tests/test_phase9_paper_execution.py",
    "tests/test_phase9_state_machine.py",
)


def live_strategy_allowlist(project_root: Path) -> dict[str, Any]:
    try:
        level_one_policy = load_level_one_canary_policy(project_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        level_one_policy = default_level_one_canary_policy()
    research_root = project_root / "output" / "research" / "phase11_13"
    schema = _json_file(research_root / "schema.json")
    boundary = _json_file(research_root / "qualification-boundary.json")
    qualification = _json_file(research_root / "qualification.json")
    observation = _json_file(
        research_root / "latest-forward-observation.json"
    )
    attestation_path = (
        project_root
        / "config"
        / "screener"
        / "shariah_attestations_v1.json"
    )
    attestations = _active_attestations(attestation_path)
    specifications = schema.get("strategies", {})
    observation_rows = {
        str(row.get("strategy_id")): row
        for row in observation.get("observations", [])
        if row.get("strategy_id")
    }
    robust_rows = {
        str(row.get("strategy_id")): row
        for row in qualification.get("strategies", [])
        if row.get("strategy_id") and bool(row.get("robust_pass"))
    }
    candidates = []
    allowed = []
    for strategy_id in sorted(robust_rows):
        specification = specifications.get(strategy_id, {})
        observed = observation_rows.get(strategy_id, {})
        attested_targets = sorted(
            set(observed.get("current_attested_target_weights", {}))
            & attestations
        )
        blockers = []
        if not boundary.get("qualification_hash"):
            blockers.append("FROZEN_QUALIFICATION_HASH_REQUIRED")
        if not bool(observed.get("independent_forward_session")):
            blockers.append("INDEPENDENT_FORWARD_SESSION_REQUIRED")
        if not attested_targets:
            blockers.append("CURRENT_PIT_ATTESTED_TARGET_REQUIRED")
        if observed.get("observation_status") != "OBSERVATION_COMPLETE":
            blockers.append("COMPLETE_FORWARD_OBSERVATION_REQUIRED")
        candidate = {
            "strategy_id": strategy_id,
            "version": "PHASE11_13_V1",
            "source_hash": (
                sha256_file(
                    project_root
                    / "src"
                    / "stocks"
                    / "research"
                    / "phase11_13.py"
                )
                if (
                    project_root
                    / "src"
                    / "stocks"
                    / "research"
                    / "phase11_13.py"
                ).exists()
                else None
            ),
            "parameter_hash": stable_hash(specification),
            "qualification_hash": boundary.get("qualification_hash"),
            "qualified_at": boundary.get("qualified_at"),
            "training_data_end": boundary.get(
                "data_end_by_strategy", {}
            ).get(strategy_id),
            "allowed_timeframes": [specification.get("timeframe")],
            "allowed_asset_classes": ["STK"],
            "allowed_symbols": attested_targets,
            "attestation_hash": (
                sha256_file(attestation_path)
                if attestation_path.exists()
                else None
            ),
            "maximum_position_weight": (
                str(level_one_policy.maximum_stock_weight)
                if level_one_policy is not None
                else None
            ),
            "canary_notional_hard_cap_eur": (
                str(level_one_policy.hard_notional_cap_eur)
                if level_one_policy is not None
                else None
            ),
            "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
            "maximum_daily_orders": 1,
            "maximum_daily_loss_eur": "5",
            "status": "PIT_LIVE_ALLOWLISTED" if not blockers else "BLOCKED",
            "blockers": blockers,
        }
        candidates.append(candidate)
        if not blockers:
            allowed.append(candidate)
    mtf_observation = _json_file(
        project_root
        / "output"
        / "research"
        / "phase11_10"
        / "latest-pit-forward-observation.json"
    )
    for observed in mtf_observation.get("observations", []):
        if (
            observed.get("qualification_status")
            != "ROBUST_SHORTLIST_FROZEN"
        ):
            continue
        attested_targets = sorted(
            set(observed.get("current_attested_target_weights", {}))
            & attestations
        )
        candidate_blockers = []
        if not observed.get("qualification_hash"):
            candidate_blockers.append(
                "FROZEN_QUALIFICATION_HASH_REQUIRED"
            )
        if not bool(observed.get("independent_forward_session")):
            candidate_blockers.append(
                "INDEPENDENT_FORWARD_SESSION_REQUIRED"
            )
        if not attested_targets:
            candidate_blockers.append(
                "CURRENT_PIT_ATTESTED_TARGET_REQUIRED"
            )
        if (
            observed.get("observation_status")
            != "PIT_OBSERVATION_COMPLETE"
        ):
            candidate_blockers.append(
                "COMPLETE_FORWARD_OBSERVATION_REQUIRED"
            )
        if (
            observed.get("provider_continuity_status")
            != "SAME_PRIMARY_PROVIDER_GO"
        ):
            candidate_blockers.append("PROVIDER_CONTINUITY_REQUIRED")
        candidate = {
            "strategy_id": observed.get("strategy_id"),
            "version": observed.get("version"),
            "source_hash": observed.get("source_hash"),
            "parameter_hash": observed.get("parameter_hash"),
            "qualification_hash": observed.get("qualification_hash"),
            "qualified_at": observed.get("qualified_at"),
            "training_data_end": observed.get("training_data_end"),
            "allowed_timeframes": observed.get(
                "allowed_timeframes",
                [],
            ),
            "allowed_asset_classes": ["STK"],
            "allowed_symbols": attested_targets,
            "attestation_hash": (
                sha256_file(attestation_path)
                if attestation_path.exists()
                else None
            ),
            "maximum_position_weight": (
                str(level_one_policy.maximum_stock_weight)
                if level_one_policy is not None
                else None
            ),
            "canary_notional_hard_cap_eur": (
                str(level_one_policy.hard_notional_cap_eur)
                if level_one_policy is not None
                else None
            ),
            "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
            "maximum_daily_orders": 1,
            "maximum_daily_loss_eur": "5",
            "status": (
                "PIT_LIVE_ALLOWLISTED"
                if not candidate_blockers
                else "BLOCKED"
            ),
            "blockers": candidate_blockers,
        }
        candidates.append(candidate)
        if not candidate_blockers:
            allowed.append(candidate)
    survivor_root = (
        project_root / "output" / "research" / "phase11_14"
    )
    survivor_boundary = _json_file(
        survivor_root / "qualification-boundary.json"
    )
    survivor_qualification = _json_file(
        survivor_root / "qualification.json"
    )
    survivor_observation = _json_file(
        survivor_root / "latest-forward-observation.json"
    )
    survivor_rows = {
        str(row.get("strategy_id")): row
        for row in survivor_qualification.get("strategies", [])
        if row.get("strategy_id")
        and row.get("robust_pass")
        and row.get("portfolio_invariants_go")
        and row.get("forward_observer_candidate")
    }
    survivor_observations = {
        str(row.get("strategy_id")): row
        for row in survivor_observation.get("observations", [])
        if row.get("strategy_id")
    }
    survivor_hash = survivor_boundary.get("qualification_hash")
    survivor_frozen_ids = {
        str(strategy_id)
        for strategy_id in survivor_boundary.get(
            "robust_strategy_ids",
            [],
        )
        if strategy_id
    }
    survivor_hash_matches = bool(
        survivor_hash
        and survivor_observation.get("qualification_hash")
        == survivor_hash
    )
    survivor_source = (
        project_root
        / "src"
        / "stocks"
        / "research"
        / "phase11_14.py"
    )
    for strategy_id in sorted(survivor_rows):
        qualification_row = survivor_rows[strategy_id]
        observed = survivor_observations.get(strategy_id, {})
        attested_targets = sorted(
            set(observed.get("current_attested_target_weights", {}))
            & attestations
        )
        candidate_blockers = []
        if survivor_boundary.get("status") != "FROZEN":
            candidate_blockers.append(
                "FROZEN_QUALIFICATION_BOUNDARY_REQUIRED"
            )
        if strategy_id not in survivor_frozen_ids:
            candidate_blockers.append(
                "FROZEN_ROBUST_STRATEGY_ID_REQUIRED"
            )
        if not survivor_hash_matches:
            candidate_blockers.append(
                "FROZEN_QUALIFICATION_HASH_REQUIRED"
            )
        if survivor_observation.get("status") != "GO":
            candidate_blockers.append(
                "COMPLETE_FORWARD_OBSERVATION_REQUIRED"
            )
        if not bool(observed.get("independent_forward_session")):
            candidate_blockers.append(
                "INDEPENDENT_FORWARD_SESSION_REQUIRED"
            )
        if not attested_targets:
            candidate_blockers.append(
                "CURRENT_PIT_ATTESTED_TARGET_REQUIRED"
            )
        if observed.get("data_freshness") != "FRESH_CLOSED_BAR":
            candidate_blockers.append(
                "FRESH_CLOSED_BAR_OBSERVATION_REQUIRED"
            )
        if qualification_row.get("asset_class") != "STOCK":
            candidate_blockers.append(
                "CONTROLLED_LIVE_STOCK_ONLY"
            )
        candidate = {
            "strategy_id": strategy_id,
            "version": "PHASE11_14_V1",
            "source_hash": (
                sha256_file(survivor_source)
                if survivor_source.exists()
                else None
            ),
            "parameter_hash": stable_hash(
                {
                    "source_strategy_id": qualification_row.get(
                        "source_strategy_id"
                    ),
                    "formula": qualification_row.get("formula"),
                    "frozen_profile": qualification_row.get(
                        "frozen_profile"
                    ),
                    "timeframe": qualification_row.get("timeframe"),
                }
            ),
            "qualification_hash": survivor_hash,
            "qualified_at": survivor_boundary.get("frozen_at"),
            "training_data_end": (
                survivor_boundary.get("data_end_by_strategy", {})
                .get(strategy_id)
            ),
            "allowed_timeframes": [
                qualification_row.get("timeframe")
            ],
            "allowed_asset_classes": ["STK"],
            "allowed_symbols": attested_targets,
            "attestation_hash": (
                sha256_file(attestation_path)
                if attestation_path.exists()
                else None
            ),
            "maximum_position_weight": (
                str(level_one_policy.maximum_stock_weight)
                if level_one_policy is not None
                else None
            ),
            "canary_notional_hard_cap_eur": (
                str(level_one_policy.hard_notional_cap_eur)
                if level_one_policy is not None
                else None
            ),
            "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
            "maximum_daily_orders": 1,
            "maximum_daily_loss_eur": "5",
            "evidence_scope": qualification_row.get(
                "evidence_scope"
            ),
            "independent_forward_session": bool(
                observed.get("independent_forward_session")
            ),
            "status": (
                "PIT_LIVE_ALLOWLISTED"
                if not candidate_blockers
                else "BLOCKED"
            ),
            "blockers": candidate_blockers,
        }
        candidates.append(candidate)
        if not candidate_blockers:
            allowed.append(candidate)
    blockers = []
    if not candidates:
        blockers.append("NO_ROBUST_FROZEN_STRATEGY")
    if not allowed:
        blockers.append("NO_PIT_LIVE_ELIGIBLE_STRATEGY")
    qualification_hashes = sorted(
        {
            str(row["qualification_hash"])
            for row in candidates
            if row.get("qualification_hash")
        }
    )
    combined_qualification_hash = (
        qualification_hashes[0]
        if len(qualification_hashes) == 1
        else stable_hash(qualification_hashes)
        if qualification_hashes
        else None
    )
    payload = {
        "schema": "ibkr_live_pit_strategy_allowlist_v1",
        "status": "GO" if not blockers else "NO_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "qualification_hash": combined_qualification_hash,
        "attestation_count": len(attestations),
        "candidate_count": len(candidates),
        "strategy_count": len(allowed),
        "strategies": allowed,
        "candidates": candidates,
        "blockers": blockers,
        "strategy_authority": (
            "FROZEN_PIT_LIVE_ALLOWLIST" if allowed else "NONE"
        ),
        "execution_authority": "NONE",
        "live_place_order_calls": 0,
    }
    _write_live_artifact(project_root, "strategy-allowlist.json", payload)
    return payload


def live_preflight(
    project_root: Path,
    *,
    env_file: str | Path = ".env.ibkr.live",
    strategy_id: str | None = None,
    symbol: str | None = None,
    max_order_eur: Decimal | None = None,
    approval: str | None = None,
    probe_socket: bool = True,
) -> dict[str, Any]:
    path = Path(env_file)
    if not path.is_absolute():
        path = project_root / path
    values = {
        key: str(value)
        for key, value in dotenv_values(path).items()
        if value is not None
    } if path.exists() else {}
    blockers: list[str] = []
    checks: dict[str, Any] = {}
    try:
        level_one_policy = load_level_one_canary_policy(project_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        level_one_policy = None
        blockers.append("LEVEL_ONE_WHOLE_SHARE_POLICY_REQUIRED")
    p0_gate = inspect_p0_execution_readiness_gate(project_root)
    checks["p0_safety_matrix_go"] = p0_gate.get("status") == "GO"
    checks["p0_safety_attestation_hash"] = p0_gate.get(
        "attestation_hash"
    )
    checks["p0_safety_gate_blockers"] = p0_gate.get("blockers", [])
    if not checks["p0_safety_matrix_go"]:
        blockers.append(P0_READINESS_BLOCKER)
    checks["dedicated_live_env"] = path.name == ".env.ibkr.live" and path.exists()
    if not checks["dedicated_live_env"]:
        blockers.append("DEDICATED_LIVE_ENV_REQUIRED")
    environment = values.get("IBKR_ENVIRONMENT", "")
    port = _int(values.get("IBKR_PORT"), -1)
    host = values.get("IBKR_HOST", "")
    checks["live_environment"] = environment == "LIVE" and port in LIVE_PORTS
    if not checks["live_environment"]:
        blockers.append("LIVE_ENVIRONMENT_OR_PORT_MISMATCH")
    checks["write_authority"] = (
        not _bool(values.get("IBKR_READ_ONLY"), True)
        and values.get("IBKR_ORDER_AUTHORITY") == "CANARY"
        and _bool(values.get("IBKR_ALLOW_ORDER_TRANSMISSION"), False)
        and _bool(values.get("IBKR_LIVE_TRADING_ENABLED"), False)
    )
    if not checks["write_authority"]:
        blockers.append("LIVE_WRITE_AUTHORITY_NOT_EXPLICIT")
    checks["autoscaling_disabled"] = not _bool(
        values.get("IBKR_LIVE_AUTOSCALE_ENABLED"), True
    )
    if not checks["autoscaling_disabled"]:
        blockers.append("AUTOSCALING_MUST_BE_DISABLED")
    max_config_order = _decimal(
        values.get("IBKR_MAX_ORDER_EUR"), Decimal("999999")
    )
    max_total = _decimal(
        values.get("IBKR_MAX_TOTAL_EXPOSURE_EUR"), Decimal("999999")
    )
    max_risk = _decimal(
        values.get("IBKR_MAX_RISK_EUR"), Decimal("999999")
    )
    max_positions = _int(values.get("IBKR_MAX_OPEN_POSITIONS"), 999)
    max_daily = _int(values.get("IBKR_MAX_NEW_ORDERS_PER_DAY"), 999)
    requested_backstop_valid = (
        max_order_eur is None
        or (
            level_one_policy is not None
            and Decimal("0") < max_order_eur
            <= level_one_policy.hard_notional_cap_eur
        )
    )
    checks["level_one_caps"] = bool(
        level_one_policy is not None
        and Decimal("0") < max_config_order
        <= level_one_policy.hard_notional_cap_eur
        and requested_backstop_valid
        and Decimal("0") < max_total
        <= level_one_policy.hard_notional_cap_eur
        and Decimal("0") < max_risk <= level_one_policy.maximum_risk_eur
        and max_positions == 1
        and max_daily == 1
    )
    if not checks["level_one_caps"]:
        blockers.append("LIVE_LEVEL_ONE_CAPS_BLOCKED")
    checks["whole_shares_only"] = not _bool(
        values.get("IBKR_ALLOW_FRACTIONAL_SHARES"), False
    )
    if not checks["whole_shares_only"]:
        blockers.append("FRACTIONAL_SHARES_MUST_BE_DISABLED")
    checks["forbidden_products_disabled"] = (
        not _bool(values.get("IBKR_ALLOW_FUTURES"), True)
        and not _bool(values.get("IBKR_ALLOW_SHORTS"), True)
        and not _bool(values.get("IBKR_ALLOW_MARGIN"), True)
        and not _bool(values.get("IBKR_ALLOW_OPTIONS"), True)
        and not _bool(values.get("IBKR_ALLOW_FOREX_SPECULATION"), True)
    )
    if not checks["forbidden_products_disabled"]:
        blockers.append("FORBIDDEN_PRODUCT_OR_LEVERAGE_FLAG")
    expected_phrase = values.get("IBKR_MANUAL_APPROVAL_PHRASE", "")
    checks["approval_phrase"] = bool(expected_phrase) and (
        approval is None or approval == expected_phrase
    )
    if not checks["approval_phrase"]:
        blockers.append("EXACT_OPERATOR_APPROVAL_REQUIRED")
    kill_switch = _kill_switch_state(project_root)
    checks["kill_switch_clear"] = not kill_switch["active"]
    if not checks["kill_switch_clear"]:
        blockers.append("KILL_SWITCH_ACTIVE")
    phase9 = _json_file(project_root / "output" / "ibkr" / "phase9" / "status.json")
    checks["phase9_status_integrity"] = _artifact_content_hash_valid(
        phase9
    )
    if not checks["phase9_status_integrity"]:
        blockers.append("PHASE9_STATUS_INTEGRITY_BLOCKED")
    checks["paper_fill_close_proven"] = bool(
        checks["phase9_status_integrity"]
        and phase9.get("checks", {}).get("submit_cancel_canary")
        and phase9.get("checks", {}).get("fill_canary")
        and phase9.get("checks", {}).get("closing_sell_canary")
        and phase9.get("status") == PHASE9_MARKER
    )
    if not checks["paper_fill_close_proven"]:
        blockers.append("PHASE9_FILL_CLOSE_CANARY_REQUIRED")
    paper_session = _json_file(
        project_root
        / "output"
        / "operations"
        / "paper-session-audit.json"
    )
    phase9_freeze = _json_file(
        project_root / "output" / "ibkr" / "phase9" / "freeze-status.json"
    )
    checks["phase9_freeze_integrity"] = _artifact_content_hash_valid(
        phase9_freeze
    )
    checks["phase9_adapter_frozen"] = bool(
        checks["phase9_freeze_integrity"]
        and phase9_freeze.get("freeze_status") == PHASE9_FREEZE_MARKER
        and phase9_freeze.get("phase9_status") == PHASE9_MARKER
    )
    dedicated_paper_session_proven = (
        paper_session.get("status") == "GO"
        and paper_session.get("marker") == "ONE_COMPLETE_PAPER_SESSION_GO"
    )
    phase9_lifecycle_proven = bool(
        checks["paper_fill_close_proven"] and checks["phase9_adapter_frozen"]
    )
    checks["complete_paper_session_proven"] = bool(
        dedicated_paper_session_proven or phase9_lifecycle_proven
    )
    checks["paper_session_evidence_source"] = (
        "DEDICATED_NATURAL_PAPER_SESSION"
        if dedicated_paper_session_proven
        else "FROZEN_PHASE9_SUBMIT_FILL_CLOSE_LIFECYCLE"
        if phase9_lifecycle_proven
        else "UNPROVEN"
    )
    if not checks["complete_paper_session_proven"]:
        blockers.append("ONE_COMPLETE_PAPER_SESSION_REQUIRED")
    live_allowlist = live_strategy_allowlist(project_root)
    checks["pit_strategy_allowlist_go"] = (
        live_allowlist.get("status") == "GO"
        and int(live_allowlist.get("strategy_count", 0)) > 0
    )
    if not checks["pit_strategy_allowlist_go"]:
        blockers.append("PIT_STRATEGY_ALLOWLIST_REQUIRED")
    allowed_strategies = {
        str(row.get("strategy_id"))
        for row in live_allowlist.get("strategies", [])
    }
    allowed_symbols = {
        str(allowed_symbol).upper()
        for row in live_allowlist.get("strategies", [])
        for allowed_symbol in row.get("allowed_symbols", [])
    }
    checks["strategy_allowlisted"] = (
        strategy_id is None or strategy_id in allowed_strategies
    )
    if strategy_id and not checks["strategy_allowlisted"]:
        blockers.append("STRATEGY_NOT_PIT_LIVE_ALLOWLISTED")
    checks["symbol_allowlisted"] = (
        symbol is None or symbol.upper() in allowed_symbols
    )
    if symbol and not checks["symbol_allowlisted"]:
        blockers.append("SYMBOL_NOT_PIT_LIVE_ALLOWLISTED")
    checks["live_reconciliation_empty"] = _live_reconciliation_empty(project_root)
    if not checks["live_reconciliation_empty"]:
        blockers.append("LIVE_RECONCILIATION_EMPTY_NOT_PROVEN")
    reconciliation = _json_file(
        project_root / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    approved_fingerprint = values.get("IBKR_LIVE_ACCOUNT_FINGERPRINT", "")
    observed_fingerprints = reconciliation.get("account_fingerprints", [])
    checks["live_account_fingerprint_matches"] = bool(
        approved_fingerprint
        and observed_fingerprints == [approved_fingerprint]
    )
    if not checks["live_account_fingerprint_matches"]:
        blockers.append("LIVE_ACCOUNT_FINGERPRINT_MISMATCH")
    capital_safety = publish_live_capital_safety(
        project_root,
        env_file=path,
    )
    checks["live_buying_power_proven"] = (
        capital_safety.get("status") == "GO"
        and capital_safety.get("buying_power_sufficient") is True
    )
    if not checks["live_buying_power_proven"]:
        blockers.append("LIVE_BUYING_POWER_NOT_PROVEN")
    strategy_eligibility = _strategy_eligibility(project_root, strategy_id)
    checks["strategy_eligible"] = strategy_eligibility["eligible"]
    checks["strategy_eligibility_status"] = strategy_eligibility["status"]
    checks["strategy_eligibility_source"] = strategy_eligibility["source"]
    checks["recommended_strategy_authority"] = strategy_eligibility.get(
        "recommended_strategy_authority"
    )
    if strategy_id and not checks["strategy_eligible"]:
        blockers.append("PAPER_OR_LIVE_CANARY_STRATEGY_REQUIRED")
    contract, contract_resolution_status = _fresh_stock_contract(
        project_root,
        symbol=symbol,
    )
    checks["contract_resolution_evaluated"] = symbol is not None
    checks["contract_resolution_status"] = contract_resolution_status
    checks["contract_resolved"] = contract_resolution_status == "FRESH_RESOLVED"
    if symbol and not checks["contract_resolved"]:
        blockers.append("EXACT_RESOLVED_CONTRACT_REQUIRED")
    checks["exit_path_implemented"] = _writer_frozen(project_root)
    checks["live_writer_frozen"] = checks["exit_path_implemented"]
    if not checks["live_writer_frozen"]:
        blockers.append("LIVE_EXECUTION_WRITER_NOT_FROZEN")
    automatic_cycle = _json_file(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "automatic-cycle-freeze.json"
    )
    checks["automatic_live_cycle_frozen"] = (
        automatic_cycle.get("status") == "GO"
        and automatic_cycle.get("freeze_status")
        == "LIVE_LEVEL_ONE_AUTOMATIC_CYCLE_FROZEN_GO"
    )
    if not checks["automatic_live_cycle_frozen"]:
        blockers.append("LIVE_AUTOMATIC_EXECUTION_CYCLE_NOT_FROZEN")
    socket_reachable: bool | None = False
    if (
        probe_socket
        and checks["live_environment"]
        and host in {"127.0.0.1", "localhost"}
    ):
        socket_reachable = _socket_reachable(host, port)
        if not socket_reachable:
            blockers.append("LIVE_TWS_SOCKET_UNREACHABLE")
    elif not probe_socket:
        socket_reachable = None
    checks["live_tws_socket_reachable"] = socket_reachable
    report: dict[str, Any] = {
        "schema": "ibkr_live_canary_preflight_v1",
        "status": "GO" if not blockers else "NO_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "safe_config": {
            "environment": environment or "UNCONFIGURED",
            "host_local_only": host in {"127.0.0.1", "localhost"},
            "port_class": "LIVE" if port in LIVE_PORTS else "INVALID",
            "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
            "configured_notional_backstop_eur": str(max_config_order),
            "requested_legacy_backstop_eur": (
                str(max_order_eur) if max_order_eur is not None else None
            ),
            "notional_cap_role": "SECONDARY_EMERGENCY_BACKSTOP",
            "canary_risk_cap_eur": str(max_risk),
            "max_total_exposure_eur": str(max_total),
            "max_open_positions": max_positions,
            "max_new_orders_per_day": max_daily,
            "autoscaling_enabled": not checks["autoscaling_disabled"],
            "futures_allowed": _bool(values.get("IBKR_ALLOW_FUTURES"), True),
        },
        "strategy_id": strategy_id,
        "symbol": symbol,
        "contract": contract,
        "account_masked": True,
        "credentials_logged": False,
        "order_sent": False,
        "live_place_order_calls": 0,
    }
    _write_report(project_root, "live_preflight.json", report)
    return report


def live_canary(
    project_root: Path,
    *,
    env_file: str | Path,
    strategy_id: str,
    symbol: str,
    max_order_eur: Decimal,
    approval: str,
) -> dict[str, Any]:
    try:
        policy = load_level_one_canary_policy(project_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        policy = default_level_one_canary_policy()
    preflight = live_preflight(
        project_root,
        env_file=env_file,
        strategy_id=strategy_id,
        symbol=symbol,
        max_order_eur=max_order_eur,
        approval=approval,
    )
    preview = {
        "schema": "ibkr_live_canary_preview_v1",
        "status": "NO_GO" if preflight["status"] != "GO" else "PREVIEW_READY",
        "strategy_id": strategy_id,
        "symbol": symbol.upper(),
        "requested_legacy_backstop_eur": str(max_order_eur),
        "canary_notional_hard_cap_eur": str(
            policy.hard_notional_cap_eur
        ),
        "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
        "notional_cap_role": "SECONDARY_EMERGENCY_BACKSTOP",
        "contract": preflight["contract"],
        "order_type": "BOUNDED_LIMIT_DAY_RTH_ONLY",
        "maximum_positions": 1,
        "maximum_orders_today": 1,
        "maximum_canary_risk_eur": str(policy.maximum_risk_eur),
        "automatic_retry": False,
        "autoscaling": False,
        "futures": False,
        "approval_verified": preflight["checks"]["approval_phrase"],
        "blockers": preflight["blockers"],
        "order_sent": False,
        "live_place_order_calls": 0,
    }
    _write_report(project_root, "live_canary_preview.json", preview)
    result = {
        **preview,
        "schema": "ibkr_live_canary_result_v1",
        "status": "NO_GO",
        "reason": (
            "LIVE_EXECUTION_WRITER_NOT_FROZEN"
            if preflight["status"] == "GO"
            else "PREFLIGHT_BLOCKED"
        ),
        "manual_review_required": True,
    }
    _write_report(project_root, "live_canary_result.json", result)
    return result


def live_reconcile(
    project_root: Path,
    *,
    env_file: str | Path = ".env.ibkr.live",
) -> dict[str, Any]:
    config, errors = _load_live_observer_config(project_root, env_file)
    if config is None or errors:
        report: dict[str, Any] = {
            "schema": "ibkr_live_reconciliation_v1",
            "status": "NO_GO",
            "reconciliation_status": "LIVE_OBSERVER_CONFIG_BLOCKED",
            "blockers": errors,
            "position_count": None,
            "open_order_count": None,
            "unknown_positions": None,
            "unknown_orders": None,
            "broker_observation_authority": "READ_ONLY",
            "execution_authority": "NONE",
            "broker_write_calls": 0,
        }
        _write_live_artifact(project_root, "reconciliation.json", report)
        return report
    if not _socket_reachable(config.host, config.port):
        report = {
            "schema": "ibkr_live_reconciliation_v1",
            "status": "NO_GO",
            "reconciliation_status": "LIVE_TWS_SOCKET_UNREACHABLE",
            "blockers": ["LIVE_TWS_SOCKET_UNREACHABLE"],
            "position_count": None,
            "open_order_count": None,
            "unknown_positions": None,
            "unknown_orders": None,
            "broker_observation_authority": "READ_ONLY",
            "execution_authority": "NONE",
            "broker_write_calls": 0,
        }
        _write_live_artifact(project_root, "reconciliation.json", report)
        return report
    snapshot, read_counters, write_counters = capture_snapshot(config)
    private_db = (
        project_root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "broker_observation.sqlite3"
    )
    store = BrokerObservationStore(private_db)
    private_hash = store.write_snapshot(snapshot)
    summary = public_snapshot_summary(snapshot, private_hash, private_db)
    timed_out = any(
        component.request_status == "CALLBACK_TIMEOUT"
        for component in snapshot.component_audits
    )
    position_count = len(snapshot.positions.positions)
    same_count = len(snapshot.same_client_open_orders.open_orders)
    all_count = len(snapshot.all_api_open_orders.open_orders)
    open_order_count = same_count + all_count
    fingerprints = sorted(
        {
            item.account_fingerprint
            for item in snapshot.account.values
            if not item.tag.startswith("$LEDGER-")
        }
    )
    empty = (
        not timed_out
        and len(fingerprints) == 1
        and position_count == 0
        and open_order_count == 0
    )
    report = {
        "schema": "ibkr_live_reconciliation_v1",
        "status": "GO" if empty else "NO_GO",
        "reconciliation_status": (
            "LIVE_RECONCILED_EMPTY"
            if empty
            else "LIVE_RECONCILIATION_REVIEW_REQUIRED"
        ),
        "position_count": position_count,
        "open_order_count": open_order_count,
        "same_client_open_order_count": same_count,
        "all_api_open_order_count": all_count,
        "execution_count": len(snapshot.executions.executions),
        "commission_count": len(snapshot.executions.commissions),
        "unknown_positions": position_count,
        "unknown_orders": open_order_count,
        "account_fingerprint_count": len(fingerprints),
        "account_fingerprints": fingerprints,
        "component_statuses": {
            item.name: item.request_status
            for item in snapshot.component_audits
        },
        **summary,
        "read_counters": read_counters,
        "write_counters": write_counters,
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "broker_write_calls": sum(write_counters.values()),
    }
    _write_live_artifact(project_root, "reconciliation.json", report)
    publish_live_capital_safety(project_root, env_file=env_file)
    publish_live_daily_profit_target(project_root, env_file=env_file)
    write_p0_execution_readiness(project_root)
    return report


def publish_live_capital_safety(
    project_root: Path,
    *,
    env_file: str | Path = ".env.ibkr.live",
) -> dict[str, Any]:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = project_root / env_path
    values = {
        key: str(value).strip()
        for key, value in dotenv_values(env_path).items()
        if value is not None
    } if env_path.exists() else {}
    expected_currency = values.get("IBKR_ACCOUNT_BASE_CURRENCY", "").upper()
    required_eur = _decimal(
        values.get("IBKR_MAX_TOTAL_EXPOSURE_EUR"),
        Decimal("25"),
    )
    private_db = (
        project_root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "broker_observation.sqlite3"
    )
    reconciliation = _json_file(
        project_root / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    blockers = []
    buying_power_sufficient = False
    financial_tag_count = 0
    snapshot_hash_matches = False
    if expected_currency != "EUR":
        blockers.append("EXPLICIT_EUR_ACCOUNT_BASE_CURRENCY_REQUIRED")
    snapshot = _latest_private_snapshot(private_db)
    if snapshot is None:
        blockers.append("PRIVATE_LIVE_SNAPSHOT_REQUIRED")
    else:
        snapshot_hash_matches = (
            snapshot.get("snapshot_hash")
            == reconciliation.get("private_snapshot_hash")
        )
        if not snapshot_hash_matches:
            blockers.append("PRIVATE_LIVE_SNAPSHOT_HASH_MISMATCH")
        payload = snapshot.get("payload", {})
        values_list = payload.get("account", {}).get("values", [])
        financial_tag_count = len(
            {
                str(row.get("tag"))
                for row in values_list
                if str(row.get("currency", "")).upper() in {"EUR", "BASE"}
            }
        )
        account_state = derive_economic_account_state(
            payload,
            expected_base_currency=expected_currency,
            snapshot_hash_verified=snapshot_hash_matches,
        )
        execution_capacity = account_state.get(
            "execution_sizing_capacity_eur"
        )
        buying_power_sufficient = (
            account_state.get("execution_status")
            == "EXECUTION_ACCOUNT_READY"
            and execution_capacity is not None
            and Decimal(str(execution_capacity)) >= required_eur
            and reconciliation.get("reconciliation_status")
            == "LIVE_RECONCILED_EMPTY"
        )
        if not buying_power_sufficient:
            blockers.append("INSUFFICIENT_OR_UNVERIFIED_LIVE_BUYING_POWER")
    report = {
        "schema": "ibkr_live_private_capital_safety_v1",
        "status": "GO" if not blockers else "NO_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_currency": (
            expected_currency if expected_currency else "UNCONFIGURED"
        ),
        "required_exposure_limit_eur": str(required_eur),
        "private_snapshot_present": snapshot is not None,
        "private_snapshot_hash_matches": snapshot_hash_matches,
        "financial_tag_count": financial_tag_count,
        "buying_power_sufficient": buying_power_sufficient,
        "execution_account_ready": bool(
            snapshot is not None
            and account_state.get("execution_status")
            == "EXECUTION_ACCOUNT_READY"
        ),
        "spendable_eur_proven": bool(
            snapshot is not None
            and account_state.get("spendable_eur") is not None
        ),
        "buying_power_is_cash": False,
        "implicit_fx_conversion_assumed": False,
        "financial_values_public": False,
        "account_masked": True,
        "blockers": sorted(set(blockers)),
        "broker_writes": 0,
    }
    _write_live_artifact(project_root, "capital-safety.json", report)
    return report


def publish_live_daily_profit_target(
    project_root: Path,
    *,
    env_file: str | Path = ".env.ibkr.live",
    now: datetime | None = None,
) -> dict[str, Any]:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = project_root / env_path
    env_values = (
        {
            key: str(value).strip()
            for key, value in dotenv_values(env_path).items()
            if value is not None
        }
        if env_path.exists()
        else {}
    )
    expected_currency = env_values.get(
        "IBKR_ACCOUNT_BASE_CURRENCY", ""
    ).upper()
    policy = _json_file(
        project_root
        / "config"
        / "capital_scaling"
        / "levels_v1.json"
    ).get("daily_profit_target", {})
    timezone_name = str(policy.get("timezone", "Europe/Amsterdam"))
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    local_date = current_time.astimezone(
        ZoneInfo(timezone_name)
    ).date()
    private_db = (
        project_root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "broker_observation.sqlite3"
    )
    reconciliation = _json_file(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "reconciliation.json"
    )
    usable = []
    blockers = []
    if expected_currency != "EUR":
        blockers.append("EXPLICIT_EUR_ACCOUNT_BASE_CURRENCY_REQUIRED")
    if not policy:
        blockers.append("DAILY_PROFIT_TARGET_POLICY_REQUIRED")
    for snapshot in _private_snapshots(private_db):
        observation = _snapshot_net_liquidation(snapshot)
        if observation is not None:
            usable.append({**snapshot, **observation})
    usable.sort(key=lambda row: row["created_at_parsed"])
    current = usable[-1] if usable else None
    baseline = None
    if current is not None:
        for candidate in reversed(usable[:-1]):
            candidate_date = candidate[
                "created_at_parsed"
            ].astimezone(ZoneInfo(timezone_name)).date()
            if candidate_date < local_date:
                baseline = candidate
                break
    if current is None:
        blockers.append("CURRENT_PRIVATE_NET_LIQUIDATION_REQUIRED")
    else:
        age_seconds = (
            current_time - current["created_at_parsed"]
        ).total_seconds()
        if age_seconds < -5 or age_seconds > 300:
            blockers.append("CURRENT_LIVE_EQUITY_SNAPSHOT_STALE")
        if (
            current.get("snapshot_hash")
            != reconciliation.get("private_snapshot_hash")
        ):
            blockers.append(
                "CURRENT_LIVE_EQUITY_SNAPSHOT_HASH_MISMATCH"
            )
    if baseline is None:
        blockers.append("PRIOR_SESSION_EQUITY_BASELINE_REQUIRED")
    elif current is not None:
        baseline_age_days = (
            current["created_at_parsed"].date()
            - baseline["created_at_parsed"].date()
        ).days
        if baseline_age_days > 7:
            blockers.append("PRIOR_SESSION_EQUITY_BASELINE_STALE")
        if (
            baseline["account_fingerprint"]
            != current["account_fingerprint"]
        ):
            blockers.append("LIVE_ACCOUNT_FINGERPRINT_CHANGED")
    if blockers:
        report = {
            "schema": "live_daily_profit_target_v1",
            "status": "NO_GO",
            "generated_at": current_time.isoformat(),
            "session_date": local_date.isoformat(),
            "target_type": "SOFT_RISK_THROTTLE",
            "input_source": (
                "BROKER_RECONCILED_NET_LIQUIDATION_CHANGE"
            ),
            "enforcement_active": True,
            "new_entries_allowed": False,
            "risk_increasing_actions_allowed": False,
            "risk_reducing_exits_allowed": True,
            "force_liquidation": False,
            "risk_chasing_allowed": False,
            "financial_values_public": False,
            "blockers": sorted(set(blockers)),
            "execution_authority": "NONE",
        }
    else:
        assert current is not None and baseline is not None
        equity_change = (
            current["net_liquidation"] - baseline["net_liquidation"]
        )
        current_cash = current.get("total_cash_value")
        baseline_cash = baseline.get("total_cash_value")
        current_gross = current.get("gross_position_value")
        baseline_gross = baseline.get("gross_position_value")
        broker_state_empty = (
            int(reconciliation.get("position_count", 0) or 0) == 0
            and int(reconciliation.get("open_order_count", 0) or 0) == 0
            and int(reconciliation.get("execution_count", 0) or 0) == 0
            and current_gross == 0
            and baseline_gross == 0
        )
        cash_flow_detected = False
        if current_cash is None or baseline_cash is None:
            blockers.append("TOTAL_CASH_VALUE_BASELINE_REQUIRED")
            net_daily_pnl = Decimal("0")
            cash_flow_status = "UNAVAILABLE_FAIL_CLOSED"
        elif broker_state_empty:
            # With no positions, orders or executions, an equity change cannot
            # be trading P&L. Treat it as external cash movement instead.
            cash_flow_detected = equity_change != 0
            net_daily_pnl = Decimal("0")
            cash_flow_status = (
                "INFERRED_EMPTY_ACCOUNT_CASH_FLOW"
                if cash_flow_detected
                else "NO_CASH_FLOW_OBSERVED_EMPTY_ACCOUNT"
            )
        elif current_cash == baseline_cash:
            net_daily_pnl = equity_change
            cash_flow_status = "NO_CASH_FLOW_OBSERVED"
        else:
            blockers.append("EXTERNAL_CASH_FLOW_ADJUSTMENT_UNAVAILABLE")
            net_daily_pnl = Decimal("0")
            cash_flow_status = "UNAVAILABLE_FAIL_CLOSED"
        if blockers:
            report = {
                "schema": "live_daily_profit_target_v1",
                "status": "NO_GO",
                "generated_at": current_time.isoformat(),
                "session_date": local_date.isoformat(),
                "target_type": "SOFT_RISK_THROTTLE",
                "input_source": "BROKER_RECONCILED_EQUITY_AND_CASH_CHANGE",
                "enforcement_active": True,
                "new_entries_allowed": False,
                "risk_increasing_actions_allowed": False,
                "risk_reducing_exits_allowed": True,
                "force_liquidation": False,
                "risk_chasing_allowed": False,
                "financial_values_public": False,
                "cash_flow_adjustment_status": cash_flow_status,
                "cash_flow_detected": cash_flow_detected,
                "blockers": sorted(set(blockers)),
                "execution_authority": "NONE",
            }
        else:
            report = daily_profit_target(
                current["net_liquidation"],
                net_daily_pnl,
                policy,
                enforcement_active=True,
            )
            report.update(
                {
                    "schema": "live_daily_profit_target_v1",
                    "generated_at": current_time.isoformat(),
                    "session_date": local_date.isoformat(),
                    "input_source": (
                        "BROKER_RECONCILED_EQUITY_AND_CASH_CHANGE"
                    ),
                    "baseline_scope": "PRIOR_OBSERVED_SESSION",
                    "current_snapshot_hash_matches": True,
                    "account_fingerprint_count": 1,
                    "account_masked": True,
                    "profit_target_is_guarantee": False,
                    "cash_flow_adjustment_status": cash_flow_status,
                    "cash_flow_detected": cash_flow_detected,
                }
            )
    path = (
        project_root
        / "output"
        / "capital"
        / "daily_profit_target.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return report


def live_prepare(
    project_root: Path,
    *,
    env_file: str | Path,
    con_id: int,
    quantity: Decimal,
    entry_limit_price: Decimal,
    stop_price: Decimal,
    take_profit_price: Decimal,
    fx_rate_to_eur: Decimal,
    reason: str,
    strategy_id: str | None = None,
    target_id: str | None = None,
    asset_class: str = "STOCK",
    liquidity_notional_eur: Decimal | None = None,
) -> dict[str, Any]:
    config, errors = load_live_canary_config(project_root, env_file)
    if config is None or errors:
        report = {
            "schema": "ibkr_live_prepare_v1",
            "status": "NO_GO",
            "prepare_status": "LIVE_CONFIG_BLOCKED",
            "blockers": errors,
            "execution_authority": "NONE",
            "live_place_order_calls": 0,
        }
        _write_live_artifact(project_root, "prepare.json", report)
        return report
    intent, risk = _build_live_intent(
        project_root,
        config,
        con_id=con_id,
        quantity=quantity,
        entry_limit_price=entry_limit_price,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        fx_rate_to_eur=fx_rate_to_eur,
        reason=reason,
        strategy_id=strategy_id,
        target_id=target_id,
        asset_class=asset_class,
        liquidity_notional_eur=liquidity_notional_eur,
    )
    if intent is None:
        report = {
            "schema": "ibkr_live_prepare_v1",
            "status": "NO_GO",
            "prepare_status": "RISK_BLOCKED",
            "risk_status": risk,
            "execution_authority": "NONE",
            "live_place_order_calls": 0,
        }
        _write_live_artifact(project_root, "prepare.json", report)
        return report
    store = LiveExecutionStore.from_project_root(project_root)
    store.initialize()
    register = store.register_intent(intent.jsonable())
    report = {
        "schema": "ibkr_live_prepare_v1",
        "status": (
            "GO"
            if register in {"INTENT_REGISTERED", "INTENT_IDEMPOTENT"}
            else "NO_GO"
        ),
        "prepare_status": "AWAITING_EXACT_LIVE_APPROVAL",
        "intent_id": intent.intent_id,
        "intent_hash": stable_hash(intent.jsonable()),
        "approval_challenge": live_approval_challenge(intent),
        "risk_status": risk,
        "approval_preview": {
            "symbol": intent.symbol,
            "asset_class": intent.asset_class,
            "share_price": str(intent.entry_limit_price),
            "desired_shares": str(intent.desired_qty),
            "normal_allowed_shares": str(intent.normal_allowed_qty),
            "level_one_shares": str(intent.canary_qty),
            "actual_notional_eur": str(intent.estimated_notional_eur),
            "actual_portfolio_weight": str(intent.portfolio_weight),
            "planned_stop": str(intent.stop_price),
            "risk_per_share_eur": str(intent.risk_per_share_eur),
            "total_planned_risk_eur": str(intent.planned_total_risk_eur),
            "estimated_costs_eur": str(intent.estimated_total_cost_eur),
            "expected_net_opportunity_eur": str(
                intent.expected_net_opportunity_eur
            ),
            "cash_after_eur": str(intent.cash_after_eur),
            "sizing_reason": intent.sizing_reason,
        },
        "register_status": register,
        "execution_authority": "NONE",
        "live_place_order_calls": 0,
    }
    _write_live_artifact(
        project_root, "prepare.json", _public_live_prepare(report)
    )
    return report


def live_approve(
    project_root: Path,
    *,
    env_file: str | Path,
    intent_id: str,
    approval: str,
) -> dict[str, Any]:
    config, errors = load_live_canary_config(project_root, env_file)
    store = LiveExecutionStore.from_project_root(project_root)
    store.initialize()
    intent = _load_live_intent(store, intent_id)
    if config is None or errors or intent is None:
        report = {
            "schema": "ibkr_live_approval_v1",
            "status": "NO_GO",
            "approval_status": "APPROVAL_BLOCKED",
            "blockers": errors,
            "execution_authority": "NONE",
            "live_place_order_calls": 0,
        }
    else:
        result = approve_live_intent(
            store,
            intent,
            approval,
            ttl_seconds=config.approval_ttl_seconds,
        )
        report = {
            "schema": "ibkr_live_approval_v1",
            **result,
            "execution_authority": "NONE",
            "live_place_order_calls": 0,
        }
    _write_live_artifact(
        project_root,
        "approval.json",
        {key: value for key, value in report.items() if key != "challenge"},
    )
    return report


def live_submit(
    project_root: Path,
    *,
    env_file: str | Path,
    intent_id: str,
    activation_approval: str,
) -> dict[str, Any]:
    config, errors = load_live_canary_config(project_root, env_file)
    if config is None or errors:
        return _live_submit_blocked(
            project_root, "LIVE_CONFIG_BLOCKED", errors
        )
    if activation_approval != config.manual_activation_phrase:
        return _live_submit_blocked(
            project_root,
            "EXACT_OPERATOR_APPROVAL_REQUIRED",
            ["EXACT_OPERATOR_APPROVAL_REQUIRED"],
        )
    store = LiveExecutionStore.from_project_root(project_root)
    store.initialize()
    intent = _load_live_intent(store, intent_id)
    if intent is None:
        return _live_submit_blocked(
            project_root, "UNKNOWN_LIVE_INTENT", ["UNKNOWN_LIVE_INTENT"]
        )
    if datetime.fromisoformat(intent.expires_at) < datetime.now(UTC):
        return _live_submit_blocked(
            project_root, "LIVE_INTENT_EXPIRED", ["LIVE_INTENT_EXPIRED"]
        )
    binding = _live_intent_binding(project_root, intent)
    if binding["status"] != "GO":
        return _live_submit_blocked(
            project_root,
            "LIVE_INTENT_BINDING_BLOCKED",
            list(binding["blockers"]),
        )
    preflight = live_preflight(
        project_root,
        env_file=env_file,
        strategy_id=intent.strategy_id,
        symbol=intent.symbol,
        approval=activation_approval,
    )
    if preflight["status"] != "GO":
        return _live_submit_blocked(
            project_root, "PREFLIGHT_BLOCKED", preflight["blockers"]
        )
    approval = store.find_unconsumed_approval(
        intent.intent_id, "LIVE_SUBMIT"
    )
    if approval is None:
        return _live_submit_blocked(
            project_root, "APPROVAL_REQUIRED", ["APPROVAL_REQUIRED"]
        )
    return _submit_live_intent_core(
        project_root,
        config=config,
        intent=intent,
        store=store,
        consume_manual_approval=True,
        authority_required=False,
    )


def live_submit_authorized(
    project_root: Path,
    *,
    env_file: str | Path,
    intent_id: str,
) -> dict[str, Any]:
    config, errors = load_live_canary_config(project_root, env_file)
    if config is None or errors:
        return _live_submit_blocked(
            project_root, "LIVE_CONFIG_BLOCKED", errors
        )
    authority = authority_status(project_root)
    current_authority = str(authority.get("execution_authority") or "NONE")
    if current_authority not in {"LIVE_LEVEL_ONE", "AUTONOMOUS_LEVEL_ONE"}:
        return _live_submit_blocked(
            project_root,
            "LIVE_LEVEL_ONE_AUTHORITY_NOT_ACTIVE",
            ["LIVE_LEVEL_ONE_AUTHORITY_NOT_ACTIVE"],
        )
    if authority.get("automatic_order_submission") is not True:
        return _live_submit_blocked(
            project_root,
            "AUTOMATIC_SUBMISSION_NOT_AUTHORIZED",
            ["AUTOMATIC_SUBMISSION_NOT_AUTHORIZED"],
        )
    store = LiveExecutionStore.from_project_root(project_root)
    store.initialize()
    intent = _load_live_intent(store, intent_id)
    if intent is None:
        return _live_submit_blocked(
            project_root, "UNKNOWN_LIVE_INTENT", ["UNKNOWN_LIVE_INTENT"]
        )
    if datetime.fromisoformat(intent.expires_at) < datetime.now(UTC):
        return _live_submit_blocked(
            project_root, "LIVE_INTENT_EXPIRED", ["LIVE_INTENT_EXPIRED"]
        )
    binding = _live_intent_binding(project_root, intent)
    if binding["status"] != "GO":
        return _live_submit_blocked(
            project_root,
            "LIVE_INTENT_BINDING_BLOCKED",
            list(binding["blockers"]),
        )
    preflight = live_preflight(
        project_root,
        env_file=env_file,
        strategy_id=intent.strategy_id,
        symbol=intent.symbol,
    )
    if preflight["status"] != "GO":
        return _live_submit_blocked(
            project_root, "PREFLIGHT_BLOCKED", preflight["blockers"]
        )
    return _submit_live_intent_core(
        project_root,
        config=config,
        intent=intent,
        store=store,
        consume_manual_approval=False,
        authority_required=True,
        required_authority=current_authority,
    )


def _submit_live_intent_core(
    project_root: Path,
    *,
    config: LiveCanaryConfig,
    intent: ManualLiveBracketIntent,
    store: LiveExecutionStore,
    consume_manual_approval: bool,
    authority_required: bool,
    required_authority: str = "LIVE_LEVEL_ONE",
) -> dict[str, Any]:
    try:
        validate_whole_share_intent(intent)
    except ValueError as exc:
        reason = str(exc)
        return _live_submit_blocked(project_root, reason, [reason])
    binding = _live_intent_binding(project_root, intent)
    if binding["status"] != "GO":
        return _live_submit_blocked(
            project_root,
            "LIVE_INTENT_BINDING_CHANGED_BEFORE_TRANSMISSION",
            list(binding["blockers"]),
        )
    app = LiveCanaryApp(CallbackState())
    try:
        connection = app.connect_and_wait(config)
        if connection["status"] != "GO" or app.next_valid_order_id is None:
            return _live_submit_blocked(
                project_root,
                "WRITER_CONNECTION_BLOCKED",
                [str(connection["connection_status"])],
            )
        base = app.next_valid_order_id
        allocated: list[int] = []
        for offset in range(3):
            result = allocate_order_id(
                store,
                broker_next_id=base + offset,
                intent_id=intent.intent_id,
            )
            if result["status"] != "GO":
                return _live_submit_blocked(
                    project_root,
                    str(result["order_id_status"]),
                    [str(result["order_id_status"])],
                )
            allocated.append(base + offset)
        if consume_manual_approval:
            consumed = consume_live_approval(store, intent)
            if consumed["status"] != "GO":
                return _live_submit_blocked(
                    project_root,
                    str(consumed["approval_status"]),
                    [str(consumed["approval_status"])],
                )
        daily_claim = store.claim_daily_submission(
            session_date=intent.session_date,
            intent_id=intent.intent_id,
        )
        if daily_claim != "LIVE_DAILY_SUBMISSION_CLAIMED":
            return _live_submit_blocked(
                project_root,
                daily_claim,
                [daily_claim],
            )
        if (
            authority_required
            and authority_status(project_root).get("execution_authority")
            != required_authority
        ):
            revoked = f"{required_authority}_AUTHORITY_REVOKED_BEFORE_TRANSMISSION"
            return _live_submit_blocked(
                project_root,
                revoked,
                [revoked],
            )
        if _kill_switch_state(project_root).get("active"):
            return _live_submit_blocked(
                project_root,
                "KILL_SWITCH_ACTIVE_BEFORE_TRANSMISSION",
                ["KILL_SWITCH_ACTIVE_BEFORE_TRANSMISSION"],
            )
        orders = build_bracket_orders(intent, parent_order_id=base)
        result = submit_bracket_once(
            app,
            order_ids=(allocated[0], allocated[1], allocated[2]),
            contract=build_stock_contract(intent),
            orders=orders,
            store=store,
            intent_id=intent.intent_id,
        )
        report = {
            "schema": "ibkr_live_submission_v1",
            **result,
            "connection_status": connection["connection_status"],
            "order_ids_masked": True,
            "account_masked": True,
            "execution_authority": (
                required_authority
                if authority_required
                else "SUPERVISED_LIVE_CANARY"
            ),
            "strategy_authority": (
                "FROZEN_PIT_LIVE_ALLOWLIST"
                if authority_required
                else "NONE"
            ),
        }
        _write_live_artifact(project_root, "submission.json", report)
        return report
    finally:
        app.disconnect()


def live_audit(project_root: Path) -> dict[str, Any]:
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="stocks-live-audit-"))
    store = LiveExecutionStore(root / "live.sqlite3")
    store.initialize()
    config = _offline_live_config()
    intent = _offline_live_intent()
    registered = store.register_intent(intent.jsonable())
    challenge = live_approval_challenge(intent)
    mismatch = approve_live_intent(
        store, intent, "WRONG", ttl_seconds=300
    )
    approved = approve_live_intent(
        store, intent, challenge, ttl_seconds=300
    )
    for order_id in (100, 101, 102):
        store.allocate_order_id(order_id, intent.intent_id)
    fake = _FakeLiveApp()
    orders = build_bracket_orders(intent, parent_order_id=100)
    submitted = submit_bracket_once(
        fake,
        order_ids=(100, 101, 102),
        contract=build_stock_contract(intent),
        orders=orders,
        store=store,
        intent_id=intent.intent_id,
    )
    duplicate = submit_bracket_once(
        fake,
        order_ids=(100, 101, 102),
        contract=build_stock_contract(intent),
        orders=orders,
        store=store,
        intent_id=intent.intent_id,
    )
    report = {
        "schema": "ibkr_live_writer_offline_audit_v1",
        "status": (
            "GO"
            if registered == "INTENT_REGISTERED"
            and mismatch["status"] == "NO_GO"
            and approved["status"] == "GO"
            and submitted["status"] == "GO"
            and submitted["live_place_order_calls"] == 3
            and duplicate["status"] == "NO_GO"
            and len(fake.calls) == 3
            and config.max_order_eur == Decimal("250")
            else "NO_GO"
        ),
        "exact_approval_enforced": mismatch["status"] == "NO_GO",
        "single_use_order_ids": duplicate["status"] == "NO_GO",
        "atomic_bracket_transmission": [
            bool(getattr(order, "transmit", False)) for order in orders
        ]
        == [False, False, True],
        "long_only": all(
            getattr(order, "action", "") in {"BUY", "SELL"}
            for order in orders
        ),
        "limit_entry": getattr(orders[0], "orderType", "") == "LMT",
        "broker_side_stop": getattr(orders[1], "orderType", "") == "STP",
        "broker_side_target": getattr(orders[2], "orderType", "") == "LMT",
        "whole_share_enforced": intent.quantity == intent.quantity.to_integral_value(),
        "fractional_quantity_supported": False,
        "live_place_order_calls_in_offline_audit": 0,
        "real_broker_app_used": False,
        "execution_authority": "NONE",
        "strategy_authority": "NONE",
    }
    _write_live_artifact(project_root, "writer-audit.json", report)
    return report


def live_freeze(project_root: Path) -> dict[str, Any]:
    audit = live_audit(project_root)
    if audit["status"] != "GO":
        return {
            "schema": "live_writer_integrity_manifest_v2",
            "status": "NO_GO",
            "freeze_status": "NO_GO",
            "blockers": ["OFFLINE_WRITER_AUDIT_FAILED"],
            "offline_audit_status": audit["status"],
            "execution_authority": "NONE",
            "live_trading_allowed": False,
        }
    report = freeze_manifest(
        project_root,
        LIVE_WRITER_SOURCES,
        operator="LOCAL_OPERATOR",
        reason="INITIAL_OFFLINE_WRITER_FREEZE",
        re_freeze=False,
        confirmed=False,
    )
    report["offline_audit_status"] = audit["status"]
    report["operational_live_canary_proven"] = False
    report["real_live_order_placed"] = False
    report["strategy_authority"] = "NONE"
    return report


def live_writer_integrity_command(
    project_root: Path,
    action: str,
    *,
    operator: str = "",
    reason: str = "",
    confirmed: bool = False,
) -> dict[str, Any]:
    artifact_path = (
        project_root / "output" / "ibkr" / "live" / "freeze-status.json"
    )
    frozen = _json_file(artifact_path)
    if action == "inspect":
        return inspect_manifest(project_root, LIVE_WRITER_SOURCES)
    if action in {"verify", "diff"}:
        result = verify_manifest(project_root, LIVE_WRITER_SOURCES, frozen)
        if action == "diff":
            path_differences_detected = bool(
                result["missing_files"]
                or result["extra_critical_files"]
                or result["changed_files"]
                or result["unauthorized_live_modules"]
            )
            result = {
                **result,
                "schema": "live_writer_integrity_diff_v2",
                "path_differences_detected": path_differences_detected,
                "difference_free": (
                    result["status"] == "GO"
                    and not path_differences_detected
                ),
            }
        _write_live_artifact(
            project_root,
            f"writer-integrity-{action}.json",
            result,
        )
        return result
    if action in {"freeze", "re-freeze"}:
        audit = live_audit(project_root)
        if audit.get("status") != "GO":
            return {
                "schema": "live_writer_integrity_manifest_v2",
                "status": "NO_GO",
                "freeze_status": "NO_GO",
                "blockers": ["OFFLINE_WRITER_AUDIT_FAILED"],
                "execution_authority": "NONE",
                "live_trading_allowed": False,
            }
        return freeze_manifest(
            project_root,
            LIVE_WRITER_SOURCES,
            operator=operator,
            reason=reason,
            re_freeze=action == "re-freeze",
            confirmed=confirmed,
        )
    return {
        "schema": "live_writer_integrity_command_v2",
        "status": "NO_GO",
        "blockers": ["UNKNOWN_INTEGRITY_ACTION"],
        "execution_authority": "NONE",
        "live_trading_allowed": False,
    }


def live_status(project_root: Path) -> dict[str, Any]:
    freeze = _json_file(
        project_root / "output" / "ibkr" / "live" / "freeze-status.json"
    )
    recon = _json_file(
        project_root / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    store = LiveExecutionStore.from_project_root(project_root)
    counts = store.initialize()
    authority = authority_status(project_root)
    machine = _json_file(
        project_root / "output" / "operations" / "machine-status.json"
    )
    heartbeat = _json_file(
        project_root / "runtime" / "heartbeat.json"
    )
    allowlist = _json_file(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "strategy-allowlist.json"
    )
    preflight = _json_file(
        project_root / "output" / "reports" / "live_preflight.json"
    )
    telegram = _json_file(
        project_root
        / "output"
        / "notifications"
        / "telegram_status.json"
    )
    research = _json_file(
        project_root
        / "output"
        / "research"
        / "autopilot"
        / "runtime-status.json"
    )
    active_signals = _json_list(
        project_root / "output" / "signals" / "active_signals.json"
    )
    daily_target = _json_file(
        project_root
        / "output"
        / "capital"
        / "daily_profit_target.json"
    )
    daily_target_verified = (
        daily_target.get("status") == "GO"
        and daily_target.get("input_source")
        == "BROKER_RECONCILED_EQUITY_AND_CASH_CHANGE"
    )
    events = store.list_events()
    real_live_order_placed = any(
        event["event_type"] == "LIVE_BRACKET_PLACE_ORDER_CALLED_ONCE"
        for event in events
    )
    execution_authority = authority.get("execution_authority", "NONE")
    writer_hash_integrity = _writer_frozen(project_root)
    dynamically_recomputed_blockers = {
        "LIVE_EXECUTION_WRITER_NOT_FROZEN",
        "PIT_STRATEGY_ALLOWLIST_REQUIRED",
    }
    open_blockers = {
        str(blocker)
        for blocker in preflight.get("blockers", [])
        if blocker and blocker not in dynamically_recomputed_blockers
    }
    open_blockers.update(
        str(blocker)
        for blocker in recon.get("blockers", [])
        if blocker
    )
    if allowlist.get("status") != "GO":
        open_blockers.add("PIT_STRATEGY_ALLOWLIST_REQUIRED")
    if not writer_hash_integrity:
        open_blockers.add("LIVE_EXECUTION_WRITER_NOT_FROZEN")
    reconciliation_complete = recon.get("status") == "GO"
    position_count = (
        int(recon.get("position_count", 0) or 0)
        if reconciliation_complete
        else "UNAVAILABLE"
    )
    open_order_count = (
        int(recon.get("open_order_count", 0) or 0)
        if reconciliation_complete
        else "UNAVAILABLE"
    )
    report = {
        "schema": "ibkr_live_controlled_runtime_status_v2",
        "status": "GO",
        "runtime_state": heartbeat.get(
            "requested_mode",
            machine.get("mode", "NOT_STARTED"),
        ),
        "runtime_enabled": bool(
            heartbeat.get("enabled", machine.get("enabled", False))
        ),
        "runtime_paused": bool(
            heartbeat.get("paused", machine.get("paused", False))
        ),
        "broker_connection": recon.get(
            "reconciliation_status", "NOT_OBSERVED"
        ),
        "account_reconciliation": recon.get(
            "reconciliation_status", "NOT_OBSERVED"
        ),
        "position_count": position_count,
        "open_order_count": open_order_count,
        "active_strategies": authority.get("active_strategies", []),
        "active_symbols": authority.get("active_symbols", []),
        "active_signal_count": len(active_signals),
        "execution_authority": execution_authority,
        "current_scaling_level": authority.get(
            "current_scaling_level", "LEVEL_0"
        ),
        "daily_pnl_eur": (
            daily_target.get("net_daily_pnl_eur")
            if daily_target_verified
            else "UNAVAILABLE_NO_BROKER_DERIVED_DAILY_EQUITY_CHANGE"
        ),
        "daily_profit_target_status": (
            str(daily_target.get("status"))
            if daily_target_verified
            else "NO_GO_UNVERIFIED_BROKER_SOURCE"
        ),
        "daily_profit_target_eur": (
            daily_target.get("daily_profit_target_eur")
            if daily_target_verified
            else "UNAVAILABLE"
        ),
        "daily_profit_target_progress_ratio": (
            daily_target.get("target_progress_ratio")
            if daily_target_verified
            else "UNAVAILABLE"
        ),
        "daily_profit_target_reached": (
            bool(daily_target.get("target_reached"))
            if daily_target_verified
            else "UNAVAILABLE"
        ),
        "new_entries_allowed_by_daily_target": (
            bool(daily_target.get("new_entries_allowed"))
            if daily_target_verified
            else False
        ),
        "total_pnl_eur": "UNAVAILABLE_NO_RECONCILED_LIVE_FILLS",
        "drawdown": "UNAVAILABLE_NO_LIVE_EQUITY_SERIES",
        "risk_utilisation": (
            "0"
            if reconciliation_complete and position_count == 0
            else "UNAVAILABLE_PRIVATE_FINANCIAL_STATE"
        ),
        "kill_switch_state": (
            "ACTIVE" if authority.get("kill_switch_active") else "CLEAR"
        ),
        "last_heartbeat": heartbeat.get(
            "last_heartbeat", machine.get("last_heartbeat")
        ),
        "last_research_cycle": research.get("last_cycle"),
        "last_telegram_delivery": telegram.get("last_successful_send"),
        "open_blockers": sorted(open_blockers),
        "writer_freeze_status": freeze.get("freeze_status", "NOT_FROZEN"),
        "writer_hash_integrity": writer_hash_integrity,
        "pit_allowlist_status": allowlist.get("status", "NOT_BUILT"),
        "pit_allowlisted_strategy_count": int(
            allowlist.get("strategy_count", 0) or 0
        ),
        "operational_live_canary_proven": False,
        "real_live_order_placed": real_live_order_placed,
        "private_intent_count": counts.get("intent_count", 0),
        "private_approval_count": counts.get("approval_count", 0),
        "private_execution_count": counts.get("execution_count", 0),
        "private_commission_count": counts.get("commission_count", 0),
        "strategy_authority": (
            "FROZEN_PIT_LIVE_ALLOWLIST"
            if execution_authority != "NONE"
            else "NONE"
        ),
    }
    _write_live_artifact(project_root, "status.json", report)
    return report


def live_component_status(
    project_root: Path, component: str
) -> dict[str, Any]:
    component = component.lower()
    reconciliation = _json_file(
        project_root / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    authority = authority_status(project_root)
    store = LiveExecutionStore.from_project_root(project_root)
    counts = store.initialize()
    reconciliation_complete = reconciliation.get("status") == "GO"
    if component == "positions":
        payload = {
            "position_count": (
                int(reconciliation.get("position_count", 0) or 0)
                if reconciliation_complete
                else "UNAVAILABLE"
            ),
            "unknown_position_count": (
                int(reconciliation.get("unknown_positions", 0) or 0)
                if reconciliation_complete
                else "UNAVAILABLE"
            ),
            "reconciliation_status": reconciliation.get(
                "reconciliation_status", "NOT_OBSERVED"
            ),
        }
    elif component == "orders":
        payload = {
            "open_order_count": (
                int(reconciliation.get("open_order_count", 0) or 0)
                if reconciliation_complete
                else "UNAVAILABLE"
            ),
            "same_client_open_order_count": (
                int(
                    reconciliation.get(
                        "same_client_open_order_count", 0
                    )
                    or 0
                )
                if reconciliation_complete
                else "UNAVAILABLE"
            ),
            "all_api_open_order_count": (
                int(
                    reconciliation.get("all_api_open_order_count", 0)
                    or 0
                )
                if reconciliation_complete
                else "UNAVAILABLE"
            ),
            "unknown_order_count": (
                int(reconciliation.get("unknown_orders", 0) or 0)
                if reconciliation_complete
                else "UNAVAILABLE"
            ),
        }
    elif component == "performance":
        payload = {
            "execution_count": counts.get("execution_count", 0),
            "commission_count": counts.get("commission_count", 0),
            "realized_pnl_eur": (
                "UNAVAILABLE_NO_RECONCILED_LIVE_FILLS"
            ),
            "unrealized_pnl_eur": (
                "UNAVAILABLE_PRIVATE_FINANCIAL_STATE"
            ),
        }
    elif component == "strategy-status":
        allowlist = live_strategy_allowlist(project_root)
        payload = {
            "allowlist_status": allowlist.get("status"),
            "active_strategies": authority.get("active_strategies", []),
            "active_symbols": authority.get("active_symbols", []),
            "candidate_count": allowlist.get("candidate_count", 0),
            "allowlisted_strategy_count": allowlist.get(
                "strategy_count", 0
            ),
        }
    elif component == "risk-status":
        payload = {
            "current_scaling_level": authority.get(
                "current_scaling_level", "LEVEL_0"
            ),
            "limits": authority.get("limits", {}),
            "kill_switch_active": authority.get(
                "kill_switch_active", False
            ),
            "qualification_hash_matches": authority.get(
                "qualification_hash_matches", False
            ),
            "allowlist_hash_matches": authority.get(
                "allowlist_hash_matches", False
            ),
        }
    elif component == "runtime-status":
        machine = _json_file(
            project_root / "output" / "operations" / "machine-status.json"
        )
        heartbeat = _json_file(
            project_root / "runtime" / "heartbeat.json"
        )
        payload = {
            "machine_status": machine.get("status", "NOT_STARTED"),
            "mode": machine.get("mode", "NOT_STARTED"),
            "enabled": bool(machine.get("enabled", False)),
            "paused": bool(machine.get("paused", False)),
            "last_cycle_status": machine.get("last_cycle_status"),
            "last_heartbeat": heartbeat.get("last_heartbeat"),
            "single_instance_lock": machine.get("single_instance_lock"),
        }
    else:
        return {
            "schema": "ibkr_live_component_status_v1",
            "status": "NO_GO",
            "component": component,
            "blockers": ["UNKNOWN_LIVE_STATUS_COMPONENT"],
            "execution_authority": "NONE",
        }
    report = {
        "schema": "ibkr_live_component_status_v1",
        "status": "GO",
        "component": component,
        **payload,
        "account_masked": True,
        "financial_values_public": False,
        "execution_authority": authority.get(
            "execution_authority", "NONE"
        ),
        "broker_writes": 0,
    }
    _write_live_artifact(project_root, f"{component}.json", report)
    return report


def live_position_status(project_root: Path) -> dict[str, Any]:
    reconciliation = _json_file(
        project_root / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    report = {
        "schema": "ibkr_live_position_status_v1",
        "status": (
            "GO"
            if reconciliation.get("status") == "GO"
            and reconciliation.get("reconciliation_status")
            == "LIVE_RECONCILED_EMPTY"
            else "NO_GO"
        ),
        "position_count": reconciliation.get("position_count"),
        "open_order_count": reconciliation.get("open_order_count"),
        "account_masked": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    _write_report(project_root, "live_position_status.json", report)
    return report


def live_close_position(
    project_root: Path, *, symbol: str, approval: str
) -> dict[str, Any]:
    return {
        "status": "NO_GO",
        "reason": "LIVE_POSITION_AND_EXIT_WRITER_NOT_RECONCILED_AND_FROZEN",
        "symbol": symbol.upper(),
        "approval_present": bool(approval),
        "kill_switch_recommended": True,
        "order_sent": False,
        "live_place_order_calls": 0,
    }


def live_kill_switch(
    project_root: Path, *, command: str, reason: str | None = None
) -> dict[str, Any]:
    state = _kill_switch_state(project_root)
    if command == "activate":
        if not reason:
            return {"status": "NO_GO", "reason": "KILL_SWITCH_REASON_REQUIRED"}
        state = {
            "active": True,
            "reason": reason,
            "activated_at": datetime.now(UTC).isoformat(),
            "cleared_at": None,
        }
        _save_kill_switch(project_root, state)
    return {
        "schema": "persistent_live_kill_switch_v1",
        "status": "GO",
        **state,
        "automatic_order_generation": False,
        "execution_authority": "NONE" if state["active"] else "CONFIG_CONTROLLED",
    }


def _candidate(
    project_root: Path, strategy_id: str | None
) -> dict[str, Any] | None:
    data = _json_file(
        project_root / "output" / "research" / "recovered_survivors.json"
    )
    return next(
        (
            row
            for row in data.get("survivors", [])
            if row.get("candidate_id") == strategy_id
        ),
        None,
    )


def _strategy_eligibility(
    project_root: Path,
    strategy_id: str | None,
) -> dict[str, Any]:
    if strategy_id is None:
        return {
            "eligible": True,
            "status": "NOT_EVALUATED_NO_STRATEGY",
            "source": "NONE",
            "recommended_strategy_authority": None,
        }

    candidate = _candidate(project_root, strategy_id)
    if candidate is not None and candidate.get("classification") in {
        "PAPER_CANDIDATE",
        "LIVE_CANARY_CANDIDATE",
    }:
        return {
            "eligible": True,
            "status": "LEGACY_LIFECYCLE_ELIGIBLE",
            "source": "RECOVERED_SURVIVOR_LIFECYCLE",
            "recommended_strategy_authority": candidate.get("classification"),
        }

    recommendation = _json_file(
        project_root
        / "output"
        / "research"
        / "evidence_throughput"
        / "strategy-authority-recommendations.json"
    )
    if not _artifact_content_hash_valid(recommendation):
        return {
            "eligible": False,
            "status": "EVIDENCE_RECOMMENDATION_INTEGRITY_BLOCKED",
            "source": "GRADUATED_EVIDENCE_RECOMMENDATION",
            "recommended_strategy_authority": None,
        }

    row = next(
        (
            item
            for item in recommendation.get("candidates", [])
            if item.get("strategy_id") == strategy_id
        ),
        None,
    )
    authority = (
        str(row.get("recommended_strategy_authority"))
        if isinstance(row, dict)
        else None
    )
    paper_gate = (
        row.get("evidence_components", {})
        .get("paper_session", {})
        .get("natural_strategy_session_gate_pass")
        if isinstance(row, dict)
        else False
    )
    eligible = bool(
        isinstance(row, dict)
        and row.get("strategy_canary_eligible") is True
        and paper_gate is True
        and authority in {"CANARY", "LIVE_SMALL", "LIVE_NORMAL"}
    )
    return {
        "eligible": eligible,
        "status": (
            "GRADUATED_EVIDENCE_ELIGIBLE"
            if eligible
            else "GRADUATED_EVIDENCE_NOT_CANARY_ELIGIBLE"
        ),
        "source": "GRADUATED_EVIDENCE_RECOMMENDATION",
        "recommended_strategy_authority": authority,
    }
def _contract(project_root: Path, symbol: str | None) -> dict[str, Any]:
    contract, _ = _fresh_stock_contract(project_root, symbol=symbol)
    return contract


def _fresh_stock_contract(
    project_root: Path,
    *,
    symbol: str | None = None,
    con_id: int | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], str]:
    if symbol is None and con_id is None:
        return {}, "NOT_EVALUATED_NO_SYMBOL"
    path = project_root / "output" / "ibkr" / "contracts" / "stocks.parquet"
    if not path.exists():
        return {}, "CONTRACT_CACHE_MISSING_BLOCKED"
    import pandas as pd

    try:
        frame = pd.read_parquet(path)
    except Exception:
        return {}, "CONTRACT_CACHE_INVALID_BLOCKED"
    required_columns = {
        "con_id",
        "symbol",
        "local_symbol",
        "security_type",
        "currency",
        "exchange",
        "primary_exchange",
        "min_tick",
        "resolved_at",
        "server_version",
        "contract_hash",
    }
    if not required_columns.issubset(frame.columns):
        return {}, "CONTRACT_CACHE_INVALID_BLOCKED"
    matches = frame[frame["security_type"].astype(str).eq("STK")]
    if symbol is not None:
        matches = matches[
            matches["symbol"].astype(str).str.upper().eq(symbol.upper())
        ]
    if con_id is not None:
        if con_id <= 0:
            return {}, "CONTRACT_IDENTITY_INVALID_BLOCKED"
        numeric_con_ids = pd.to_numeric(matches["con_id"], errors="coerce")
        matches = matches[numeric_con_ids.eq(con_id)]
    if len(matches) != 1:
        return {}, "CONTRACT_NOT_FOUND_OR_AMBIGUOUS_BLOCKED"
    row = matches.iloc[0].to_dict()
    try:
        resolved_con_id = int(row["con_id"])
        minimum_tick = Decimal(str(row["min_tick"]))
        server_version = int(row["server_version"])
    except (TypeError, ValueError, ArithmeticError):
        return {}, "CONTRACT_IDENTITY_INVALID_BLOCKED"
    text_fields = {
        key: str(row.get(key, "")).strip()
        for key in (
            "symbol",
            "local_symbol",
            "security_type",
            "currency",
            "exchange",
            "primary_exchange",
            "contract_hash",
        )
    }
    contract_hash = text_fields["contract_hash"].upper()
    if (
        resolved_con_id <= 0
        or minimum_tick <= 0
        or server_version <= 0
        or any(not value for key, value in text_fields.items() if key != "contract_hash")
        or len(contract_hash) != 64
        or any(character not in "0123456789ABCDEF" for character in contract_hash)
    ):
        return {}, "CONTRACT_IDENTITY_INVALID_BLOCKED"
    resolved_at = _cache_datetime(row.get("resolved_at"))
    if resolved_at is None:
        return {}, "CONTRACT_CACHE_TIMESTAMP_INVALID_BLOCKED"
    effective_now = now or datetime.now(UTC)
    expires_at = resolved_at + contract_cache_ttl(IbkrSecurityType.STK)
    if resolved_at > effective_now + timedelta(minutes=5):
        return {}, "CONTRACT_CACHE_TIMESTAMP_INVALID_BLOCKED"
    if effective_now >= expires_at:
        return {}, "CONTRACT_CACHE_STALE_BLOCKED"
    return (
        {
            "con_id": resolved_con_id,
            "symbol": text_fields["symbol"],
            "local_symbol": text_fields["local_symbol"],
            "security_type": text_fields["security_type"],
            "currency": text_fields["currency"],
            "exchange": text_fields["exchange"],
            "primary_exchange": text_fields["primary_exchange"],
            "minimum_tick": str(minimum_tick),
            "resolved_at": resolved_at.isoformat(),
            "cache_expires_at": expires_at.isoformat(),
            "cache_status": "FRESH",
            "contract_source": "PHASE2_EXACT_STK_CACHE",
            "server_version": server_version,
            "contract_hash": contract_hash,
        },
        "FRESH_RESOLVED",
    )


def _cache_datetime(value: Any) -> datetime | None:
    try:
        if hasattr(value, "to_pydatetime"):
            parsed = value.to_pydatetime()
        elif isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _live_reconciliation_empty(project_root: Path) -> bool:
    data = _json_file(
        project_root / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    return bool(
        data.get("status") == "GO"
        and data.get("reconciliation_status") == "LIVE_RECONCILED_EMPTY"
        and data.get("unknown_orders", 1) == 0
        and data.get("unknown_positions", 1) == 0
    )


def _kill_switch_state(project_root: Path) -> dict[str, Any]:
    path = (
        project_root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "kill-switch.json"
    )
    return (
        _json_file(path)
        if path.exists()
        else {
            "active": False,
            "reason": None,
            "activated_at": None,
            "cleared_at": None,
        }
    )


def _save_kill_switch(project_root: Path, state: dict[str, Any]) -> None:
    path = (
        project_root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "kill-switch.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _write_report(project_root, "kill_switch_status.json", state)


def _socket_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _artifact_content_hash_valid(payload: dict[str, Any]) -> bool:
    observed = payload.get("content_hash")
    return isinstance(observed, str) and observed == stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "content_hash"
        }
    )


def _json_list(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []


def _latest_private_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT snapshot_hash, payload_json
                FROM snapshots
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    if row is None:
        return None
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return None
    return {
        "snapshot_hash": str(row["snapshot_hash"]),
        "payload": payload if isinstance(payload, dict) else {},
    }


def _private_snapshots(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                """
                SELECT snapshot_hash, payload_json, created_at
                FROM snapshots
                ORDER BY created_at ASC
                """
            ).fetchall()
    except (sqlite3.Error, OSError):
        return []
    output = []
    for snapshot_hash, payload_json, created_at in rows:
        try:
            parsed = datetime.fromisoformat(
                str(created_at).replace("Z", "+00:00")
            )
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            payload = json.loads(str(payload_json))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        output.append(
            {
                "snapshot_hash": str(snapshot_hash),
                "payload": (
                    payload if isinstance(payload, dict) else {}
                ),
                "created_at": str(created_at),
                "created_at_parsed": parsed,
            }
        )
    return output


def _snapshot_net_liquidation(
    snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    values = (
        snapshot.get("payload", {})
        .get("account", {})
        .get("values", [])
    )
    rows = [
        row
        for row in values
        if row.get("tag") == "NetLiquidation"
        and str(row.get("currency", "")).upper() in {"EUR", "BASE"}
    ]
    fingerprints = {
        str(row.get("account_fingerprint", ""))
        for row in rows
        if row.get("account_fingerprint")
    }
    if len(fingerprints) != 1:
        return None
    parsed = []
    for row in rows:
        try:
            value = Decimal(str(row.get("value")))
        except Exception:
            continue
        if value.is_finite() and value >= 0:
            parsed.append(
                (
                    (
                        0
                        if str(row.get("currency", "")).upper()
                        == "EUR"
                        else 1
                    ),
                    value,
                )
            )
    if not parsed:
        return None
    parsed.sort(key=lambda item: item[0])
    return {
        "net_liquidation": parsed[0][1],
        "account_fingerprint": next(iter(fingerprints)),
        "total_cash_value": _snapshot_account_value(
            values, "TotalCashValue"
        ),
        "gross_position_value": _snapshot_account_value(
            values, "GrossPositionValue"
        ),
    }


def _snapshot_account_value(
    values: list[dict[str, Any]], tag: str
) -> Decimal | None:
    parsed: list[tuple[int, Decimal]] = []
    for row in values:
        if row.get("tag") != tag:
            continue
        currency = str(row.get("currency", "")).upper()
        if currency not in {"EUR", "BASE"}:
            continue
        try:
            value = Decimal(str(row.get("value")))
        except Exception:
            continue
        if value.is_finite():
            parsed.append((0 if currency == "EUR" else 1, value))
    if not parsed:
        return None
    parsed.sort(key=lambda item: item[0])
    return parsed[0][1]


def _active_attestations(path: Path) -> set[str]:
    payload = _json_file(path)
    now = datetime.now(UTC)
    eligible = set()
    for row in payload.get("attestations", []):
        try:
            screened_at = datetime.fromisoformat(
                str(row["screened_at"]).replace("Z", "+00:00")
            )
            expires_at = datetime.fromisoformat(
                str(row["expires_at"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            row.get("status") == "SHARIAH_ELIGIBLE_PIT"
            and screened_at <= now <= expires_at
        ):
            eligible.add(str(row.get("symbol", "")).upper())
    return eligible


def _write_report(project_root: Path, name: str, report: dict[str, Any]) -> None:
    root = project_root / "output" / "reports"
    root.mkdir(parents=True, exist_ok=True)
    payload = {**report, "content_hash": stable_hash(report)}
    (root / name).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def _bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(raw: str | None, default: int) -> int:
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return default


def _decimal(raw: str | None, default: Decimal) -> Decimal:
    try:
        return Decimal(str(raw))
    except Exception:
        return default


def _load_live_observer_config(
    project_root: Path,
    env_file: str | Path,
) -> tuple[Phase8Config | None, list[str]]:
    path = Path(env_file)
    if not path.is_absolute():
        path = project_root / path
    if path.name != ".env.ibkr.live" or not path.exists():
        return None, ["DEDICATED_LIVE_ENV_REQUIRED"]
    values = {
        key: str(value).strip()
        for key, value in dotenv_values(path).items()
        if value is not None
    }
    host = values.get("IBKR_HOST", "")
    port = _int(values.get("IBKR_PORT"), -1)
    writer_id = _int(values.get("IBKR_CLIENT_ID"), -1)
    recon_id = _int(values.get("IBKR_RECON_CLIENT_ID"), -1)
    fingerprint_key = values.get("IBKR_ACCOUNT_FINGERPRINT_KEY", "")
    timeout = float(
        _decimal(
            values.get("IBKR_LIVE_CALLBACK_TIMEOUT_SECONDS"),
            Decimal("15"),
        )
    )
    errors: list[str] = []
    if values.get("IBKR_ENVIRONMENT") != "LIVE" or port not in LIVE_PORTS:
        errors.append("LIVE_ENVIRONMENT_OR_PORT_MISMATCH")
    if host not in {"127.0.0.1", "localhost"}:
        errors.append("LIVE_HOST_LOCAL_ONLY_REQUIRED")
    if writer_id <= 0 or recon_id <= 0 or writer_id == recon_id:
        errors.append("LIVE_CLIENT_ID_CONFIGURATION_BLOCKED")
    if not fingerprint_key:
        errors.append("ACCOUNT_FINGERPRINT_KEY_MISSING")
    if timeout <= 0:
        errors.append("LIVE_TIMEOUT_CONFIG_BLOCKED")
    if errors:
        return None, sorted(set(errors))
    return (
        Phase8Config(
            env_file=path,
            host=host,
            port=port,
            primary_client_id=writer_id,
            recon_client_id=recon_id,
            account_fingerprint_key=fingerprint_key,
            request_timeout_seconds=timeout,
            commission_grace_seconds=1.0,
            snapshot_stability_delay_seconds=1.0,
            read_only=True,
            live_trading_enabled=False,
            allow_order_transmission=False,
            execution_authority="NONE",
        ),
        [],
    )


def _build_live_intent(
    project_root: Path,
    config: LiveCanaryConfig,
    *,
    con_id: int,
    quantity: Decimal,
    entry_limit_price: Decimal,
    stop_price: Decimal,
    take_profit_price: Decimal,
    fx_rate_to_eur: Decimal,
    reason: str,
    strategy_id: str | None,
    target_id: str | None = None,
    asset_class: str = "STOCK",
    liquidity_notional_eur: Decimal | None = None,
) -> tuple[ManualLiveBracketIntent | None, dict[str, Any]]:
    blockers: list[str] = []
    contract = _contract_by_con_id(project_root, con_id)
    if not contract:
        blockers.append("EXACT_RESOLVED_STK_CONTRACT_REQUIRED")
    if not quantity.is_finite() or quantity <= 0:
        blockers.append("LIVE_QUANTITY_BLOCKED")
    elif quantity != quantity.to_integral_value():
        blockers.append("FRACTIONAL_QUANTITY_FORBIDDEN")
    prices = (entry_limit_price, stop_price, take_profit_price)
    if any(not value.is_finite() or value <= 0 for value in prices):
        blockers.append("LIVE_PRICE_BLOCKED")
    elif not (stop_price < entry_limit_price < take_profit_price):
        blockers.append("LONG_BRACKET_PRICE_ORDER_BLOCKED")
    if not fx_rate_to_eur.is_finite() or fx_rate_to_eur <= 0:
        blockers.append("FX_RATE_BLOCKED")
    if not reason.strip():
        blockers.append("OPERATOR_REASON_REQUIRED")

    controlled_portfolio = config.execution_authority == "LIVE_LEVEL_TWO"
    sizing: dict[str, Any] = {}
    if controlled_portfolio and not blockers:
        share_price_eur = entry_limit_price * fx_rate_to_eur
        planned_risk = (
            quantity * (entry_limit_price - stop_price) * fx_rate_to_eur
        )
        sizing = {
            "status": "GO",
            "sizing_reason": "BOUNDED_NORMAL_PORTFOLIO",
            "blocking_reason": None,
            "desired_qty": int(quantity),
            "normal_allowed_qty": int(quantity),
            "canary_qty": int(quantity),
            "downscaled_for_canary": False,
            "risk_per_share_eur": str(
                (entry_limit_price - stop_price) * fx_rate_to_eur
            ),
            "planned_total_risk_eur": str(planned_risk),
            "actual_notional_eur": str(quantity * share_price_eur),
            "actual_portfolio_weight": "0",
            "cash_before_eur": "0",
            "cash_after_eur": "0",
            "estimated_total_cost_eur": "0",
            "expected_net_opportunity_eur": "0",
            "hard_notional_backstop_eur": str(config.max_order_eur),
            "primary_sizing_authority": "NORMAL_RISK_FIRST_WHOLE_SHARE",
            "notional_cap_role": "SECONDARY_EMERGENCY_BACKSTOP",
            "fractional_shares_allowed": False,
        }
    elif not blockers:
        sizing_context = _live_sizing_context(
            project_root,
            symbol=str(contract.get("symbol") or "") if contract else "",
            asset_class=asset_class,
            liquidity_notional_eur=liquidity_notional_eur,
        )
        blockers.extend(sizing_context["blockers"])
    if not controlled_portfolio and not blockers:
        sizing = evaluate_whole_share_canary(
            project_root,
            asset_class=asset_class,
            instrument_currency=str(contract.get("currency") or "EUR"),
            desired_qty=quantity,
            account_equity_eur=sizing_context["account_equity_eur"],
            available_cash_eur=sizing_context["available_cash_eur"],
            reserved_cash_eur=sizing_context["reserved_cash_eur"],
            entry_price_local=entry_limit_price,
            protective_stop_local=stop_price,
            take_profit_local=take_profit_price,
            fx_rate_to_eur=fx_rate_to_eur,
            normal_risk_budget_eur=sizing_context[
                "normal_risk_budget_eur"
            ],
            normal_maximum_position_weight=sizing_context[
                "normal_maximum_position_weight"
            ],
            normal_maximum_portfolio_heat_pct=sizing_context[
                "normal_maximum_portfolio_heat_pct"
            ],
            liquidity_notional_eur=sizing_context[
                "liquidity_notional_eur"
            ],
            existing_position_notional_eur=sizing_context[
                "existing_position_notional_eur"
            ],
            existing_total_exposure_eur=sizing_context[
                "existing_total_exposure_eur"
            ],
            existing_portfolio_risk_eur=sizing_context[
                "existing_portfolio_risk_eur"
            ],
        )
        if sizing.get("status") != "GO":
            blockers.append(
                str(sizing.get("blocking_reason") or "CANARY_SIZING_BLOCKED")
            )

    executable_quantity = Decimal(str(sizing.get("canary_qty", "0")))
    estimated_notional = Decimal(
        str(sizing.get("actual_notional_eur", "0"))
    )
    planned_loss = Decimal(
        str(sizing.get("planned_total_risk_eur", "0"))
    )
    if sizing:
        if estimated_notional > config.max_order_eur:
            blockers.append("CANARY_NOTIONAL_BACKSTOP")
        if estimated_notional > config.max_total_exposure_eur:
            blockers.append("LIVE_TOTAL_EXPOSURE_EXCEEDED")
        if planned_loss <= 0 or planned_loss > config.max_risk_eur:
            blockers.append("LIVE_PLANNED_LOSS_EXCEEDED")

    now = datetime.now(UTC)
    risk = {
        **sizing,
        "status": "GO" if not blockers else "NO_GO",
        "blockers": sorted(set(blockers)),
        "quantity_within_profile_limit": (
            Decimal("0") < executable_quantity <= config.maximum_quantity
        ),
        "minimum_valid_whole_share": executable_quantity >= Decimal("1"),
        "maximum_quantity": str(config.maximum_quantity),
        "whole_share": (
            executable_quantity == executable_quantity.to_integral_value()
        ),
        "estimated_notional_eur": str(estimated_notional),
        "maximum_planned_loss_eur": str(planned_loss),
        "max_order_eur": str(config.max_order_eur),
        "max_risk_eur": str(config.max_risk_eur),
        "long_only": True,
        "margin": False,
        "shorting": False,
    }
    if blockers or not contract:
        return None, risk

    key = economic_order_key(
        strategy_id=strategy_id or "MANUAL_LIVE_CANARY",
        strategy_version=(
            "LIVE_CONTROLLED_PORTFOLIO_V1"
            if controlled_portfolio
            else "LIVE_CANARY_V1"
        ),
        decision_id=stable_hash(
            {
                "con_id": con_id,
                "session_date": now.date().isoformat(),
                "reason": reason.strip(),
                "entry": str(entry_limit_price),
                "stop": str(stop_price),
                "target": str(take_profit_price),
            }
        )[:24],
        con_id=con_id,
        side="BUY",
        target_position=executable_quantity,
        session_date=now.date().isoformat(),
    )
    intent_id = (
        f"LIVE-PORTFOLIO-{key[:24]}"
        if controlled_portfolio
        else f"LIVE-CANARY-{key[:24]}"
    )
    intent = ManualLiveBracketIntent(
        intent_id=intent_id,
        economic_order_key=key,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=10)).isoformat(),
        account_fingerprint=config.approved_account_fingerprint,
        con_id=con_id,
        symbol=str(contract["symbol"]),
        security_type="STK",
        currency=str(contract["currency"]),
        exchange=str(contract["exchange"]),
        quantity=executable_quantity,
        entry_limit_price=entry_limit_price,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        fx_rate_to_eur=fx_rate_to_eur,
        estimated_notional_eur=estimated_notional,
        maximum_planned_loss_eur=planned_loss,
        session_date=now.date().isoformat(),
        operator_reason=reason.strip(),
        contract_hash=str(contract["contract_hash"]),
        strategy_id=strategy_id,
        target_id=target_id,
        asset_class=str(asset_class).upper(),
        desired_qty=Decimal(str(sizing["desired_qty"])),
        normal_allowed_qty=Decimal(str(sizing["normal_allowed_qty"])),
        canary_qty=executable_quantity,
        risk_per_share_eur=Decimal(str(sizing["risk_per_share_eur"])),
        planned_total_risk_eur=planned_loss,
        portfolio_weight=Decimal(str(sizing["actual_portfolio_weight"])),
        cash_before_eur=Decimal(str(sizing["cash_before_eur"])),
        cash_after_eur=Decimal(str(sizing["cash_after_eur"])),
        estimated_total_cost_eur=Decimal(
            str(sizing["estimated_total_cost_eur"])
        ),
        expected_net_opportunity_eur=Decimal(
            str(sizing["expected_net_opportunity_eur"])
        ),
        canary_notional_hard_cap_eur=Decimal(
            str(sizing["hard_notional_backstop_eur"])
        ),
        sizing_reason=str(sizing["sizing_reason"]),
        downscaled_for_canary=bool(sizing["downscaled_for_canary"]),
        fractional_allowed=False,
        capital_level=2 if controlled_portfolio else 1,
    )
    return intent, risk


def _live_sizing_context(
    project_root: Path,
    *,
    symbol: str,
    asset_class: str,
    liquidity_notional_eur: Decimal | None,
) -> dict[str, Any]:
    payload = _json_file(
        project_root / "data/portfolio/private/current-state.json"
    )
    account = payload.get("account_state", {})
    blockers: list[str] = []
    if not isinstance(account, dict) or account.get("status") != "GO":
        blockers.append("CURRENT_RECONCILED_ACCOUNT_STATE_REQUIRED")
        account = {}
    equity = _decimal(
        str(account.get("net_liquidation_eur") or "0"), Decimal("0")
    )
    cash_candidates = [
        _decimal(str(account.get(key) or "0"), Decimal("0"))
        for key in (
            "eur_available_for_new_longs",
            "available_funds_eur",
            "total_cash_value_eur",
        )
    ]
    positive_cash = [value for value in cash_candidates if value > 0]
    available_cash = min(positive_cash) if positive_cash else Decimal("0")
    if equity <= 0:
        blockers.append("POSITIVE_ACCOUNT_EQUITY_REQUIRED")
    if available_cash <= 0:
        blockers.append("AVAILABLE_CASH_REQUIRED")

    whole_share = payload.get("whole_share_sizing", {})
    if not isinstance(whole_share, dict):
        whole_share = {}
    positions = whole_share.get("positions", [])
    matching = next(
        (
            row
            for row in positions
            if isinstance(row, dict)
            and str(row.get("ticker", "")).upper() == symbol.upper()
        ),
        {},
    )
    normal_risk_budget = _decimal(
        str(matching.get("risk_budget_eur") or equity * Decimal("0.01")),
        Decimal("0"),
    )
    policy = _json_file(
        project_root / "config/portfolio/active_manager_v1.json"
    ).get("small_account_whole_share", {})
    if not isinstance(policy, dict):
        policy = {}
    pooled = str(asset_class).upper() in {"ETF", "COMMODITY_VEHICLE"}
    normal_weight = _decimal(
        str(
            policy.get(
                "maximum_etf_weight" if pooled else "maximum_stock_weight",
                "0.40" if pooled else "0.35",
            )
        ),
        Decimal("0"),
    )
    normal_heat = _decimal(
        str(policy.get("maximum_portfolio_heat", "0.06")), Decimal("0")
    )
    if normal_weight <= 0 or normal_heat <= 0:
        blockers.append("NORMAL_PORTFOLIO_RISK_POLICY_REQUIRED")

    capacity = liquidity_notional_eur
    if capacity is None:
        report = _json_file(project_root / "output/capital/capacity_report.json")
        capacity = next(
            (
                _decimal(
                    str(row.get("maximum_order_value_eur") or "0"),
                    Decimal("0"),
                )
                for row in report.get("instruments", [])
                if isinstance(row, dict)
                and str(row.get("symbol", "")).upper() == symbol.upper()
            ),
            Decimal("0"),
        )
    if capacity <= 0:
        blockers.append("LIQUIDITY_LIMIT")

    current_exposure = (
        equity
        * _decimal(
            str(whole_share.get("current_gross_exposure_pct") or "0"),
            Decimal("0"),
        )
    )
    current_risk = (
        equity
        * _decimal(
            str(whole_share.get("current_portfolio_heat") or "0"),
            Decimal("0"),
        )
    )
    existing_position_notional = (
        _decimal(
            str(matching.get("current_quantity") or "0"), Decimal("0")
        )
        * _decimal(
            str(matching.get("unit_notional_eur") or "0"), Decimal("0")
        )
    )
    return {
        "blockers": sorted(set(blockers)),
        "account_equity_eur": equity,
        "available_cash_eur": available_cash,
        "reserved_cash_eur": Decimal("0"),
        "normal_risk_budget_eur": normal_risk_budget,
        "normal_maximum_position_weight": normal_weight,
        "normal_maximum_portfolio_heat_pct": normal_heat,
        "liquidity_notional_eur": capacity,
        "existing_position_notional_eur": existing_position_notional,
        "existing_total_exposure_eur": current_exposure,
        "existing_portfolio_risk_eur": current_risk,
        "account_state_source": "RECONCILED_PRIVATE_PORTFOLIO_STATE",
        "cash_reservation_source": "NET_AVAILABLE_FOR_NEW_LONGS",
    }


def _contract_by_con_id(
    project_root: Path,
    con_id: int,
) -> dict[str, Any]:
    contract, _ = _fresh_stock_contract(project_root, con_id=con_id)
    return contract


def _live_intent_binding(
    project_root: Path,
    intent: ManualLiveBracketIntent,
) -> dict[str, Any]:
    blockers: list[str] = []
    strategy_id = str(intent.strategy_id or "").strip()
    symbol = str(intent.symbol).strip().upper()
    identity_symbols = {symbol}
    if not strategy_id:
        blockers.append("LIVE_STRATEGY_ID_REQUIRED")

    allowlist = live_strategy_allowlist(project_root)
    matching_strategy = next(
        (
            row
            for row in allowlist.get("strategies", [])
            if str(row.get("strategy_id")) == strategy_id
        ),
        None,
    )
    if allowlist.get("status") != "GO":
        blockers.append("PIT_STRATEGY_ALLOWLIST_REQUIRED")
    elif matching_strategy is None:
        blockers.append("STRATEGY_NOT_PIT_LIVE_ALLOWLISTED")
    else:
        allowed_symbols = {
            str(item).strip().upper()
            for item in matching_strategy.get("allowed_symbols", [])
            if item
        }
        for portfolio_symbol, metadata in broad_asset_metadata(
            project_root
        ).items():
            if not isinstance(metadata, dict):
                continue
            if (
                str(metadata.get("broker_symbol") or "").strip().upper()
                == symbol
            ):
                identity_symbols.add(str(portfolio_symbol).strip().upper())
        if identity_symbols.isdisjoint(allowed_symbols):
            blockers.append("SYMBOL_NOT_ALLOWED_FOR_LIVE_STRATEGY")

    contract = _contract_by_con_id(project_root, intent.con_id)
    if not contract:
        blockers.append("EXACT_RESOLVED_STK_CONTRACT_REQUIRED")
    else:
        if str(contract.get("symbol", "")).strip().upper() != symbol:
            blockers.append("LIVE_INTENT_CONTRACT_SYMBOL_MISMATCH")
        if str(contract.get("security_type", "")).strip().upper() != "STK":
            blockers.append("LIVE_INTENT_CONTRACT_SECURITY_TYPE_MISMATCH")
        if (
            str(contract.get("currency", "")).strip().upper()
            != str(intent.currency).strip().upper()
        ):
            blockers.append("LIVE_INTENT_CONTRACT_CURRENCY_MISMATCH")
        if str(contract.get("contract_hash", "")) != intent.contract_hash:
            blockers.append("LIVE_INTENT_CONTRACT_HASH_MISMATCH")

    return {
        "status": "GO" if not blockers else "NO_GO",
        "blockers": sorted(set(blockers)),
        "strategy_id": strategy_id or None,
        "symbol": symbol,
        "strategy_identity_symbols": sorted(identity_symbols),
        "allowlist_status": allowlist.get("status", "NO_GO"),
        "contract_bound": bool(contract) and not any(
            blocker.startswith("LIVE_INTENT_CONTRACT_")
            for blocker in blockers
        ),
        "execution_authority": "NONE",
        "live_place_order_calls": 0,
    }


def _load_live_intent(
    store: LiveExecutionStore,
    intent_id: str,
) -> ManualLiveBracketIntent | None:
    payload = store.get_intent(intent_id)
    if payload is None:
        return None
    decimal_fields = (
        "quantity",
        "entry_limit_price",
        "stop_price",
        "take_profit_price",
        "fx_rate_to_eur",
        "estimated_notional_eur",
        "maximum_planned_loss_eur",
        "desired_qty",
        "normal_allowed_qty",
        "canary_qty",
        "risk_per_share_eur",
        "planned_total_risk_eur",
        "portfolio_weight",
        "cash_before_eur",
        "cash_after_eur",
        "estimated_total_cost_eur",
        "expected_net_opportunity_eur",
        "canary_notional_hard_cap_eur",
    )
    for field in decimal_fields:
        if field in payload:
            payload[field] = Decimal(str(payload[field]))
    return ManualLiveBracketIntent(**payload)


def _public_live_prepare(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"approval_challenge", "risk_status"}
    } | {
        "approval_challenge_private": True,
        "financial_values_private": True,
    }


def _live_submit_blocked(
    project_root: Path,
    reason: str,
    blockers: list[str],
) -> dict[str, Any]:
    report = {
        "schema": "ibkr_live_submission_v1",
        "status": "NO_GO",
        "submission_status": reason,
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "strategy_authority": "NONE",
        "live_place_order_calls": 0,
    }
    _write_live_artifact(project_root, "submission.json", report)
    return report


def _write_live_artifact(
    project_root: Path,
    name: str,
    report: dict[str, Any],
) -> None:
    root = project_root / "output" / "ibkr" / "live"
    root.mkdir(parents=True, exist_ok=True)
    public = {
        **report,
        "account_masked": True,
        "credentials_logged": False,
    }
    public["content_hash"] = stable_hash(public)
    (root / name).write_text(
        json.dumps(public, indent=2, default=str),
        encoding="utf-8",
    )


def _writer_frozen(project_root: Path) -> bool:
    artifact = _json_file(
        project_root / "output" / "ibkr" / "live" / "freeze-status.json"
    )
    return bool(
        verify_manifest(
            project_root,
            LIVE_WRITER_SOURCES,
            artifact,
        ).get("writer_hash_integrity")
    )


def _offline_live_config() -> LiveCanaryConfig:
    return LiveCanaryConfig(
        host="127.0.0.1",
        port=7496,
        writer_client_id=91,
        recon_client_id=92,
        quote_client_id=93,
        account_fingerprint_key="offline-test-key",
        approved_account_fingerprint="FINGERPRINT",
        manual_activation_phrase="OFFLINE-ACTIVATION",
        writer_enabled=True,
        max_order_eur=Decimal("250"),
        max_total_exposure_eur=Decimal("250"),
        max_risk_eur=Decimal("10"),
        max_open_positions=1,
        max_new_orders_per_day=1,
        approval_ttl_seconds=300,
        callback_timeout_seconds=1.0,
        fractional_shares_enabled=False,
    )


def _offline_live_intent() -> ManualLiveBracketIntent:
    now = datetime.now(UTC)
    return ManualLiveBracketIntent(
        intent_id="LIVE-CANARY-OFFLINE-AUDIT",
        economic_order_key=stable_hash("LIVE-CANARY-OFFLINE-AUDIT"),
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=10)).isoformat(),
        account_fingerprint="FINGERPRINT",
        con_id=265598,
        symbol="AAPL",
        security_type="STK",
        currency="USD",
        exchange="SMART",
        quantity=Decimal("1"),
        entry_limit_price=Decimal("70"),
        stop_price=Decimal("65"),
        take_profit_price=Decimal("80"),
        fx_rate_to_eur=Decimal("0.85"),
        estimated_notional_eur=Decimal("59.5"),
        maximum_planned_loss_eur=Decimal("4.25"),
        session_date=now.date().isoformat(),
        operator_reason="offline audit",
        contract_hash=stable_hash("AAPL-265598"),
        strategy_id="OFFLINE-AUDIT",
        target_id="OFFLINE-TARGET",
        desired_qty=Decimal("2"),
        normal_allowed_qty=Decimal("2"),
        canary_qty=Decimal("1"),
        risk_per_share_eur=Decimal("4.25"),
        planned_total_risk_eur=Decimal("4.25"),
        portfolio_weight=Decimal("0.0318"),
        cash_before_eur=Decimal("1870"),
        cash_after_eur=Decimal("1809.7"),
        estimated_total_cost_eur=Decimal("0.8"),
        expected_net_opportunity_eur=Decimal("7.7"),
        canary_notional_hard_cap_eur=Decimal("250"),
        sizing_reason="CANARY_DOWNSCALED_TO_ONE_SHARE",
        downscaled_for_canary=True,
    )


class _FakeLiveApp:
    def __init__(self) -> None:
        self.calls: list[tuple[int, object, object]] = []

    def _record(
        self,
        order_id: int,
        contract: object,
        order: object,
    ) -> None:
        self.calls.append((order_id, contract, order))

    def __getattr__(self, name: str) -> Any:
        if name == "place" + "Order":
            return self._record
        raise AttributeError(name)
