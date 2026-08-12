from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from filelock import FileLock, Timeout

from stocks.operations.background_jobs import (
    background_job_status,
    launch_background_job,
)
from stocks.application.phase_gates import phase1_freeze_status
from stocks.ibkr.paper_execution import (
    PHASE9_MARKER,
    phase9_reconcile,
    phase9_status,
)
from stocks.ibkr.contract_cache import (
    ContractCacheLayout,
    read_contract_cache_rows,
)
from stocks.live import (
    AUTONOMOUS_LEVEL_ONE,
    AUTONOMOUS_PROFILE,
    LIVE_LEVEL_ONE,
    activate_live_capability,
    automatic_cycle,
    authority_status,
    create_live_capability,
    kill_level_one,
    live_capability_status,
    live_close_position,
    live_kill_switch,
    live_position_status,
    live_preflight,
    live_reconcile,
    pause_level_one,
    resume_level_one,
)
from stocks.market.context import DEFAULT_CONTEXT_SYMBOLS
from stocks.operations.paper_runtime import (
    paper_runtime_status,
    plan_paper_cycle,
)
from stocks.operations.manual_positions import (
    audit_manual_position_broker_match,
)
from stocks.operations.paper_safety import paper_kill_switch_drill
from stocks.operations.paper_session import audit_paper_session
from stocks.operations.paper_writer import (
    AUTO_PAPER_AUTHORITY,
    automatic_paper_preflight,
    build_paper_strategy_allowlist,
    execute_automatic_paper_cycle,
)
from stocks.signals.storage import SignalStore
from stocks.signals.active_swing import active_swing_context_gaps


MACHINE_MODES = {
    "SIGNALS_ONLY",
    "PAPER_AUTOMATIC",
    "LIVE_CANARY_AUTOMATIC",
    "CONTROLLED_LIVE",
}
PAPER_ACTIVATION_PHRASE = "ACTIVATE BOUNDED AUTOMATIC PAPER"
PAPER_LIMITS = {
    "max_order_eur": 500,
    "max_total_exposure_eur": 1500,
    "max_open_positions": 3,
    "max_new_orders_per_day": 3,
    "max_risk_per_trade_pct": 0.5,
    "max_daily_loss_pct": 2,
    "max_drawdown_pct": 8,
}
LIVE_LIMITS = {
    "max_order_eur": 250,
    "max_total_exposure_eur": 250,
    "max_open_positions": 1,
    "max_new_orders_per_day": 1,
    "max_risk_eur": 10,
    "max_daily_loss_eur": 5,
    "max_drawdown_eur": 10,
}
INTRADAY_CORE_SYMBOLS = (
    "AAPL",
    "AMZN",
    "ASML",
    "GOOGL",
    "INTC",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "ON",
    "XOM",
    "EEM",
    "EFA",
    "IWM",
    "QQQ",
    "SPY",
    "TLT",
    "DBC",
    "GLD",
    "SLV",
)
INTRADAY_CHALLENGER_BATCH_SIZE = 10
INTRADAY_CHALLENGER_POOL_LIMIT = 150
INTRADAY_PRIORITY_SYMBOL_LIMIT = 25
INTRADAY_COLLECTION_SYMBOL_LIMIT = 50
MARKET_CONTEXT_SYMBOL_LIMIT = 30
OPERATIONS_PROFILE_PATH = Path("config/operations/autonomous_multi_asset_v1.json")


def execution_command(
    project_root: Path,
    command: str,
    *,
    environment: str = "paper",
    approval: str | None = None,
    env_file: str | Path = ".env.ibkr",
    session_date: str | None = None,
    confirmed: bool = False,
    profile: str = AUTONOMOUS_PROFILE,
) -> dict[str, Any]:
    state = _state(project_root)
    if command == "status":
        return _execution_status(project_root, state)
    if command == "preflight":
        return _execution_preflight(project_root, environment, env_file)
    if (
        command
        in {
            "create-live-capability",
            "activate-live-capability",
            "activate-live-canary",
        }
        and not confirmed
    ):
        return {
            "schema": "controlled_live_operator_confirmation_v1",
            "status": "NO_GO",
            "blockers": ["EXPLICIT_YES_CONFIRMATION_REQUIRED"],
            "execution_authority": "NONE",
            "automatic_orders_allowed": False,
            "broker_writes": 0,
        }
    if (
        command
        in {
            "create-live-capability",
            "activate-live-capability",
            "activate-live-canary",
        }
        and profile != AUTONOMOUS_PROFILE
    ):
        return {
            "schema": "controlled_live_profile_validation_v1",
            "status": "NO_GO",
            "blockers": ["UNKNOWN_LIVE_PROFILE"],
            "profile": profile,
            "execution_authority": "NONE",
            "automatic_orders_allowed": False,
            "broker_writes": 0,
        }
    if command == "activate-paper":
        preflight = _execution_preflight(project_root, "paper", env_file)
        blockers = list(preflight["blockers"])
        if approval != PAPER_ACTIVATION_PHRASE:
            blockers.append("EXACT_PAPER_ACTIVATION_PHRASE_REQUIRED")
        if blockers:
            return _publish(
                project_root,
                "execution-activation.json",
                {
                    "schema": "bounded_execution_activation_v1",
                    "status": "NO_GO",
                    "environment": "paper",
                    "blockers": sorted(set(blockers)),
                    "expected_approval": PAPER_ACTIVATION_PHRASE,
                    "execution_authority": "NONE",
                    "automatic_orders_allowed": False,
                },
            )
        state.update(
            mode="PAPER_AUTOMATIC",
            paper_enabled=True,
            live_enabled=False,
            activated_at=_now(),
        )
        _save_state(project_root, state)
        return _execution_status(project_root, state)
    if command == "deactivate-paper":
        state.update(mode="SIGNALS_ONLY", paper_enabled=False)
        _save_state(project_root, state)
        return _execution_status(project_root, state)
    if command == "live-capability-status":
        return live_capability_status(project_root)
    if command == "create-live-capability":
        reconciliation = live_reconcile(project_root, env_file=env_file)
        report = live_preflight(
            project_root,
            env_file=env_file,
        )
        if report["status"] != "GO":
            return {
                **report,
                "reconciliation_status": reconciliation.get("reconciliation_status"),
                "execution_authority": "NONE",
                "automatic_orders_allowed": False,
            }
        return create_live_capability(
            project_root,
            preflight=report,
            confirmed=confirmed,
            profile=profile,
        )
    if command in {
        "activate-live-capability",
        "activate-live-canary",
    }:
        reconciliation = live_reconcile(
            project_root,
            env_file=env_file,
        )
        report = live_preflight(
            project_root,
            env_file=env_file,
            approval=approval,
        )
        if report["status"] != "GO":
            return {
                **report,
                "execution_authority": "NONE",
                "automatic_orders_allowed": False,
            }
        if command == "activate-live-canary":
            capability = create_live_capability(
                project_root,
                preflight=report,
                confirmed=confirmed,
                profile=profile,
            )
            if capability.get("status") != "GO":
                return {
                    **capability,
                    "reconciliation_status": (
                        reconciliation.get("reconciliation_status") if reconciliation else None
                    ),
                    "automatic_orders_allowed": False,
                }
        transition = activate_live_capability(
            project_root,
            preflight=report,
            confirmed=confirmed,
        )
        if transition.get("execution_authority") != LIVE_LEVEL_ONE:
            return {
                **transition,
                "automatic_orders_allowed": False,
            }
        state.update(
            mode="CONTROLLED_LIVE",
            requested_mode="CONTROLLED_LIVE",
            paper_enabled=False,
            live_enabled=True,
            activated_at=_now(),
        )
        _save_state(project_root, state)
        return {
            **_execution_status(project_root, state),
            "transition_status": transition.get("transition_status"),
        }
    if command == "deactivate-live":
        pause_level_one(
            project_root,
            reason="OPERATOR_DEACTIVATE_LIVE",
        )
        state.update(mode="SIGNALS_ONLY", live_enabled=False)
        state["requested_mode"] = "SIGNALS_ONLY"
        _save_state(project_root, state)
        return _execution_status(project_root, state)
    if command == "pause-live":
        transition = pause_level_one(
            project_root,
            reason="OPERATOR_PAUSE",
        )
        state.update(
            mode="SIGNALS_ONLY",
            requested_mode="SIGNALS_ONLY",
            live_enabled=False,
        )
        _save_state(project_root, state)
        return transition
    if command == "resume-live":
        report = live_preflight(
            project_root,
            env_file=env_file,
        )
        transition = resume_level_one(
            project_root,
            preflight=report,
        )
        if transition.get("execution_authority") in {
            LIVE_LEVEL_ONE,
            AUTONOMOUS_LEVEL_ONE,
        }:
            state.update(
                mode="CONTROLLED_LIVE",
                requested_mode="CONTROLLED_LIVE",
                paper_enabled=False,
                live_enabled=True,
            )
            _save_state(project_root, state)
        return transition
    if command == "kill-live":
        reason = str(approval or "").strip()
        transition = kill_level_one(project_root, reason=reason)
        if transition.get("transition_status") == "LIVE_LEVEL_ONE_KILLED":
            live_kill_switch(
                project_root,
                command="activate",
                reason=reason,
            )
            state.update(
                mode="SIGNALS_ONLY",
                requested_mode="SIGNALS_ONLY",
                paper_enabled=False,
                live_enabled=False,
            )
            _save_state(project_root, state)
        return transition
    if command == "paper-fill-close-canary":
        status = phase9_status(project_root)
        readiness_path = project_root / "output" / "ibkr" / "phase9" / "canary-b-readiness.json"
        readiness = _read_json(readiness_path)
        complete = bool(status.get("checks", {}).get("fill_canary"))
        return _publish(
            project_root,
            "paper-fill-close-canary.json",
            {
                "schema": "paper_fill_close_canary_operator_gate_v1",
                "status": "GO" if complete else "OPERATOR_ACTION_REQUIRED",
                "canary_complete": complete,
                "offline_readiness": readiness.get("status"),
                "operator_required": not complete,
                "paper_only": True,
                "required_sequence": [
                    "prepare and approve one bounded marketable BUY limit",
                    "submit and observe broker fill plus commission",
                    "reconcile one paper position",
                    "prepare and approve one bounded marketable SELL limit",
                    "submit and observe close fill plus commission",
                    "reconcile empty and run restart recovery audit",
                ],
                "automatic_canary_submission": False,
                "reason": (
                    None if complete else "CURRENT_PRICE_AND_EXACT_PER_INTENT_APPROVAL_REQUIRED"
                ),
                "execution_authority": "NONE",
                "paper_place_order_calls": 0,
                "live_place_order_calls": 0,
            },
        )
    if command == "paper-runtime-status":
        return paper_runtime_status(project_root)
    if command == "paper-allowlist":
        return build_paper_strategy_allowlist(project_root)
    if command == "paper-writer-preflight":
        return automatic_paper_preflight(project_root, env_file)
    if command == "paper-writer-cycle":
        authority = (
            AUTO_PAPER_AUTHORITY
            if state.get("mode") == "PAPER_AUTOMATIC" and state.get("paper_enabled")
            else "NONE"
        )
        return execute_automatic_paper_cycle(
            project_root,
            execution_authority=authority,
            env_file=env_file,
        )
    if command == "paper-kill-switch-drill":
        return paper_kill_switch_drill(project_root)
    if command == "paper-session-audit":
        return audit_paper_session(
            project_root,
            session_date=session_date,
        )
    raise ValueError(f"Unknown execution command: {command}")


def machine_command(
    project_root: Path,
    command: str,
    *,
    mode: str = "SIGNALS_ONLY",
    max_cycles: int = 1,
    interval_seconds: int = 300,
) -> dict[str, Any]:
    state = _state(project_root)
    if command == "status":
        return _machine_status(project_root, state)
    if command == "start":
        paper_enabled, live_enabled = _requested_mode_enablement(
            project_root,
            state,
            mode,
        )
        state.update(
            enabled=True,
            paused=False,
            mode=mode,
            requested_mode=mode,
            paper_enabled=paper_enabled,
            live_enabled=live_enabled,
            activated_at=state.get("activated_at") or _now(),
        )
        _save_state(project_root, state)
        _runtime_heartbeat(project_root, "READY", state=state)
        return _machine_status(project_root, state)
    if command == "stop":
        state.update(
            enabled=False,
            paused=False,
            mode="SIGNALS_ONLY",
            requested_mode="SIGNALS_ONLY",
            paper_enabled=False,
            live_enabled=False,
        )
        _save_state(project_root, state)
        _runtime_heartbeat(project_root, "STOPPED", state=state)
        return _machine_status(project_root, state)
    if command == "pause":
        state.update(paused=True)
        _save_state(project_root, state)
        return _machine_status(project_root, state)
    if command == "resume":
        state.update(enabled=True, paused=False)
        _save_state(project_root, state)
        return _machine_status(project_root, state)
    if command == "run-once":
        return _run_cycles(
            project_root,
            state,
            mode=mode,
            max_cycles=1,
            interval_seconds=interval_seconds,
        )
    if command == "run":
        # The mode supplied to an explicit bounded run is authoritative for
        # its first cycle.  Persisting an older live requested_mode here can
        # otherwise turn a diagnostic SIGNALS_ONLY invocation into a live
        # cycle before the operator has a chance to inspect the result.
        paper_enabled, live_enabled = _requested_mode_enablement(
            project_root,
            state,
            mode,
        )
        state.update(
            enabled=True,
            paused=False,
            mode=mode,
            requested_mode=mode,
            paper_enabled=paper_enabled,
            live_enabled=live_enabled,
            activated_at=state.get("activated_at") or _now(),
        )
        _save_state(project_root, state)
        return _run_cycles(
            project_root,
            state,
            mode=mode,
            max_cycles=max_cycles,
            interval_seconds=interval_seconds,
        )
    raise ValueError(f"Unknown machine command: {command}")


def _requested_mode_enablement(
    project_root: Path,
    state: dict[str, Any],
    mode: str,
) -> tuple[bool, bool]:
    if mode == "PAPER_AUTOMATIC":
        return bool(state.get("paper_enabled")), False
    if mode in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"}:
        authority = str(authority_status(project_root).get("execution_authority") or "NONE")
        return False, authority in {
            LIVE_LEVEL_ONE,
            AUTONOMOUS_LEVEL_ONE,
        }
    return False, False


def positions_command(
    project_root: Path,
    command: str,
    *,
    environment: str = "paper",
    symbol: str | None = None,
    approval: str | None = None,
    signal_id: str | None = None,
    quantity: Decimal | None = None,
    fill_price: Decimal | None = None,
    position_id: str | None = None,
    ownership_mode: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    if command == "status":
        paper = phase9_reconcile(project_root)
        live = live_position_status(project_root)
        with SignalStore(project_root) as store:
            manual = store.manual_positions()
        ownership_counts = _count_values(
            manual,
            "ownership_status",
        )
        return _publish(
            project_root,
            "positions-status.json",
            {
                "schema": "managed_positions_status_v1",
                "status": "GO",
                "paper_reconciliation_status": paper.get("reconciliation_status"),
                "paper_position_count": paper.get("broker_position_count", 0),
                "paper_open_order_count": paper.get("broker_open_order_count", 0),
                "live_status": live.get("status"),
                "manual_position_count": len(manual),
                "manual_ownership_counts": ownership_counts,
                "manual_broker_unverified_count": sum(
                    row.get("broker_match_status") != "MATCHED" for row in manual
                ),
                "manual_auto_execution_eligible_count": sum(
                    bool(row.get("automatic_execution_eligible")) for row in manual
                ),
                "manual_financial_values_public": False,
                "execution_authority": "NONE",
            },
        )
    if command == "reconcile":
        if environment == "paper":
            return phase9_reconcile(project_root)
        return live_position_status(project_root)
    if command == "close":
        if not symbol:
            return _blocked("SYMBOL_REQUIRED")
        if environment == "live":
            return live_close_position(
                project_root,
                symbol=symbol,
                approval=approval or "",
            )
        return _publish(
            project_root,
            "paper-position-close.json",
            {
                "schema": "paper_position_close_operator_gate_v1",
                "status": "OPERATOR_ACTION_REQUIRED",
                "symbol": symbol.upper(),
                "reason": (
                    "USE_PHASE9_PREPARE_SELL_APPROVE_AND_SUBMIT_AFTER_RECONCILED_POSITION_MATCH"
                ),
                "automatic_close_submission": False,
                "execution_authority": "NONE",
                "paper_place_order_calls": 0,
            },
        )
    if command == "register-manual":
        if (
            not signal_id
            or quantity is None
            or fill_price is None
            or quantity <= 0
            or fill_price <= 0
        ):
            return _blocked("INVALID_MANUAL_POSITION_REGISTRATION")
        with SignalStore(project_root) as store:
            signal = store.signal(signal_id)
            if signal is None:
                return _blocked("SIGNAL_NOT_FOUND")
            if str(signal.get("action", "")).upper() not in {
                "BUY",
                "LONG",
            }:
                return _blocked("MANUAL_LONG_ENTRY_SIGNAL_REQUIRED")
            preferred_entry = Decimal(str(signal["preferred_entry"]))
            private_payload = {
                "quantity": str(quantity),
                "fill_price": str(fill_price),
                "preferred_entry": str(preferred_entry),
                "entry_slippage_value": str((fill_price - preferred_entry) * quantity),
                "initial_stop": signal.get("initial_stop"),
                "take_profit_1": signal.get("take_profit_1"),
                "take_profit_2": signal.get("take_profit_2"),
                "strategy_id": signal.get("strategy_id"),
            }
            contract = _manual_contract_identity(
                project_root,
                signal,
            )
            private_payload["contract_resolution_status"] = contract["status"]
            try:
                position = store.register_manual_position(
                    signal_id=signal_id,
                    environment=environment,
                    con_id=contract.get("con_id"),
                    contract_hash=contract.get("contract_hash"),
                    quantity=str(quantity),
                    fill_price=str(fill_price),
                    payload=private_payload,
                )
            except ValueError as exc:
                return _blocked(str(exc))
        return _publish(
            project_root,
            "manual-position-registration.json",
            _public_manual_position_result(
                position,
                status="GO",
            ),
        )
    if command == "broker-match":
        if not position_id:
            return _blocked("POSITION_ID_REQUIRED")
        with SignalStore(project_root) as store:
            matched_position = store.manual_position(position_id)
            if matched_position is None:
                return _blocked("MANUAL_POSITION_NOT_FOUND")
            audit = audit_manual_position_broker_match(
                project_root,
                position=matched_position,
                environment=environment,
            )
            stored = None
            snapshot_hash = audit.get("snapshot_hash")
            if snapshot_hash:
                stored = store.set_manual_position_broker_match(
                    position_id=position_id,
                    broker_match_status=str(audit["broker_match_status"]),
                    snapshot_hash=str(snapshot_hash),
                    detail=dict(audit.get("private_detail", {})),
                )
        public_position = stored or matched_position
        public_position["broker_match_status"] = audit["broker_match_status"]
        return _publish(
            project_root,
            "manual-position-broker-match.json",
            {
                **_public_manual_position_result(
                    public_position,
                    status=str(audit["status"]),
                ),
                "snapshot_age_seconds": audit.get("snapshot_age_seconds"),
                "broker_position_count": audit.get("broker_position_count"),
                "identity_match_count": audit.get("identity_match_count"),
                "quantity_match": audit.get("quantity_match", False),
                "private_match_values_public": False,
            },
        )
    if command in {"claim", "unclaim"}:
        if not confirmed:
            return _blocked("EXPLICIT_YES_CONFIRMATION_REQUIRED")
        if not position_id:
            return _blocked("POSITION_ID_REQUIRED")
        if command == "claim" and ownership_mode != "bot-managed":
            return _blocked("BOT_MANAGED_MODE_REQUIRED")
        to_ownership = "BOT_MANAGED" if command == "claim" else "MANUAL_TRACKED"
        event_type = (
            "MANUAL_POSITION_CLAIMED" if command == "claim" else "MANUAL_POSITION_UNCLAIMED"
        )
        with SignalStore(project_root) as store:
            try:
                transitioned_position = store.transition_manual_position(
                    position_id=position_id,
                    to_ownership=to_ownership,
                    event_type=event_type,
                )
            except ValueError as exc:
                return _blocked(str(exc))
        if transitioned_position is None:
            return _blocked("MANUAL_POSITION_NOT_FOUND")
        return _publish(
            project_root,
            f"manual-position-{command}.json",
            _public_manual_position_result(
                transitioned_position,
                status="GO",
            ),
        )
    raise ValueError(f"Unknown positions command: {command}")


def _run_cycles(
    project_root: Path,
    state: dict[str, Any],
    *,
    mode: str,
    max_cycles: int,
    interval_seconds: int,
) -> dict[str, Any]:
    if mode not in MACHINE_MODES:
        return _blocked("UNKNOWN_MACHINE_MODE")
    if not state["enabled"]:
        return _blocked("MACHINE_DISABLED")
    if state["paused"]:
        return _blocked("MACHINE_PAUSED")
    cycles = max(1, min(int(max_cycles), 1_440))
    interval = max(60, min(int(interval_seconds), 86_400))
    lock_path = _private_root(project_root) / "machine.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            records: list[dict[str, Any]] = []
            for index in range(cycles):
                current = _state(project_root)
                if not current["enabled"] or current["paused"]:
                    break
                cycle_mode = str(current.get("requested_mode") or mode)
                records.append(_cycle(project_root, current, cycle_mode))
                if index + 1 < cycles:
                    time.sleep(interval)
            run_status = (
                "GO"
                if records and all(record.get("status") == "GO" for record in records)
                else "DEGRADED"
                if records
                else "NO_GO"
            )
            return {
                "schema": "stocks_machine_bounded_run_v1",
                "status": run_status,
                "mode": mode,
                "bounded": True,
                "cycles_requested": cycles,
                "cycles_completed": len(records),
                "interval_seconds": interval,
                "cycles": records,
                "execution_authority": records[-1]["execution_authority"] if records else "NONE",
            }
    except Timeout:
        return _blocked("MACHINE_SINGLE_INSTANCE_LOCKED")


def _cycle(project_root: Path, state: dict[str, Any], requested_mode: str) -> dict[str, Any]:
    started = datetime.now(UTC)
    cycle_id = f"CYCLE-{started.strftime('%Y%m%dT%H%M%S%fZ')}"
    _cycle_progress(project_root, cycle_id, "PHASE1_FREEZE", "STARTED")
    phase1 = phase1_freeze_status(project_root)
    _cycle_progress(
        project_root,
        cycle_id,
        "PHASE1_FREEZE",
        "COMPLETED",
        component_status=phase1.status,
    )
    mode = _effective_mode(state, requested_mode)
    intraday_refresh_plan = _intraday_refresh_plan(project_root, state)
    tactical_cadence = _cadence_hours(project_root, "tactical_15m_bars", 0.25)
    primary_cadence = _cadence_hours(project_root, "primary_1h_scan", 1.0)
    portfolio_cadence = _cadence_hours(project_root, "portfolio_fast_path", 0.25)
    tactical_due = _refresh_due(
        state.get("last_tactical_data_refresh"),
        hours=tactical_cadence,
    )
    primary_refresh_report = _read_json(project_root / "output/operations/primary-refresh.json")
    primary_due = _primary_step_due(
        primary_refresh_report,
        "MARKET_DATA",
        hours=primary_cadence,
    )
    observed_positions_present = _has_observed_positions(project_root)
    fast_path_due = (
        tactical_due
        or observed_positions_present
        or _refresh_due(
            state.get("last_portfolio_refresh"),
            hours=portfolio_cadence,
        )
    )
    reconciliation_arguments = _reconciliation_arguments(project_root, mode)
    reconciliation = _observed_command_step(
        project_root,
        cycle_id,
        "RECONCILIATION",
        reconciliation_arguments,
        timeout_seconds=60,
    )
    priority_portfolio = (
        _observed_command_step(
            project_root,
            cycle_id,
            "POSITION_RISK_MANAGER",
            ("portfolio", "plan"),
            timeout_seconds=120,
        )
        if observed_positions_present
        else {"status": "NOT_APPLICABLE_NO_OBSERVED_POSITIONS"}
    )
    tactical_data_refresh = (
        _observed_command_step(
            project_root,
            cycle_id,
            "MARKET_DATA_15M",
            (
                "data",
                "multitimeframe",
                "collect",
                "--symbols",
                ",".join(intraday_refresh_plan["symbols"]),
                "--intervals",
                "15m",
                "--providers",
                "local,datascraper,yfinance",
                "--lookback-days",
                "30",
            ),
            timeout_seconds=240,
        )
        if tactical_due
        else {"status": "NOT_DUE"}
    )
    active_swing_context_gap_audit = (
        active_swing_context_gaps(
            project_root,
            intraday_refresh_plan["symbols"],
        )
        if tactical_due
        else {
            "schema": "active_swing_context_gap_audit_v1",
            "status": "NOT_DUE",
            "gap_symbol_count": 0,
            "gap_symbols": [],
        }
    )
    context_gap_symbols = active_swing_context_gap_audit.get("gap_symbols", [])
    context_gap_symbols = (
        [str(symbol).upper() for symbol in context_gap_symbols]
        if isinstance(context_gap_symbols, list)
        else []
    )
    if not tactical_due:
        active_swing_context_bootstrap = {"status": "NOT_DUE"}
    elif not _is_success_status(tactical_data_refresh):
        active_swing_context_bootstrap = {
            "status": "DATA_BLOCKED",
            "blockers": ["TACTICAL_MARKET_DATA_REFRESH_REQUIRED"],
        }
    elif not context_gap_symbols:
        active_swing_context_bootstrap = {
            "schema": "active_swing_context_bootstrap_v1",
            "status": "GO",
            "refresh_required": False,
            "gap_symbol_count": 0,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "broker_writes": 0,
            "orders_generated": 0,
        }
    else:
        active_swing_context_bootstrap = _observed_command_step(
            project_root,
            cycle_id,
            "ACTIVE_SWING_CONTEXT_BOOTSTRAP",
            (
                "data",
                "multitimeframe",
                "collect",
                "--symbols",
                ",".join(context_gap_symbols),
                "--intervals",
                "1h,4h",
                "--providers",
                "local,datascraper,yfinance",
                "--lookback-days",
                "30",
            ),
            timeout_seconds=180,
        )
    active_swing_candidates = (
        _observed_command_step(
            project_root,
            cycle_id,
            "ACTIVE_SWING_15M_CANDIDATES",
            (
                "signals",
                "active-swing-refresh",
                "--symbols",
                ",".join(intraday_refresh_plan["symbols"]),
            ),
            timeout_seconds=60,
        )
        if _is_success_status(tactical_data_refresh)
        else {
            "status": "DATA_BLOCKED" if tactical_due else "NOT_DUE",
            "blockers": (["TACTICAL_MARKET_DATA_REFRESH_REQUIRED"] if tactical_due else []),
        }
    )
    primary_background_current = background_job_status(project_root, "primary_refresh")
    primary_background_due = _background_refresh_due(
        primary_background_current,
        hours=tactical_cadence,
    )
    if primary_background_due:
        _cycle_progress(
            project_root,
            cycle_id,
            "PRIMARY_BACKGROUND",
            "STARTED",
        )
        primary_background = launch_background_job(
            project_root,
            "primary_refresh",
            ("primary-refresh",),
            timeout_seconds=7_200,
        )
        _cycle_progress(
            project_root,
            cycle_id,
            "PRIMARY_BACKGROUND",
            "COMPLETED",
            component_status=_normalized_status(primary_background),
        )
    else:
        primary_background = primary_background_current
    data_refresh = _delegated_primary(primary_background, "MARKET_DATA")
    daily = _delegated_primary(primary_background, "DAILY")
    hmm_regime = _delegated_primary(primary_background, "HMM_REGIME")
    dynamic = {
        **_delegated_primary(primary_background, "DYNAMIC"),
        "signals": _read_json(project_root / "output/dynamic/current_signals.json"),
    }
    lifecycle = (
        _signal_lifecycle(project_root, dynamic)
        if fast_path_due
        else {
            "status": "NOT_DUE",
            "fresh_entry_count": 0,
            "exit_count": 0,
        }
    )
    multitimeframe_watchlist = _delegated_primary(primary_background, "MULTITIMEFRAME_WATCHLIST")
    multitimeframe_pit_observation = (
        _observed_command_step(
            project_root,
            cycle_id,
            "MTF_PIT_OBSERVATION",
            ("research", "phase11-10", "pit-observe"),
            timeout_seconds=180,
        )
        if _refresh_due(
            state.get("last_multitimeframe_pit_observation_refresh"),
            hours=tactical_cadence,
        )
        else {"status": "NOT_DUE"}
    )
    multitimeframe_pit_notification = (
        _observed_command_step(
            project_root,
            cycle_id,
            "MTF_PIT_NOTIFICATION",
            ("telegram", "send-pit-mtf-signals"),
            timeout_seconds=60,
        )
        if _is_success_status(multitimeframe_pit_observation)
        else {"status": "NOT_DUE"}
    )
    fast_track_observation = _delegated_primary(primary_background, "FAST_TRACK_OBSERVATION")
    broad_shadow_observation = _delegated_primary(primary_background, "BROAD_SHADOW_OBSERVATION")
    survivor_shadow_observation = _delegated_primary(
        primary_background, "SURVIVOR_SHADOW_OBSERVATION"
    )
    broad_shadow_notification = _delegated_primary(primary_background, "BROAD_SHADOW_NOTIFICATION")
    macro = _delegated_primary(primary_background, "MACRO")
    news_digest = _delegated_primary(primary_background, "NEWS_DIGEST")
    market_context = _delegated_primary(primary_background, "MARKET_CONTEXT")
    cot_context = _delegated_primary(primary_background, "COT_CONTEXT")
    asset_context = _delegated_primary(primary_background, "ASSET_CONTEXT")
    entry_observer = (
        _observed_command_step(
            project_root,
            cycle_id,
            "ENTRY_OBSERVER",
            (
                "market",
                "context",
                "observe",
                "--max-symbols",
                "20",
                "--depth-symbols",
                "5",
            ),
            timeout_seconds=60,
        )
        if fast_path_due
        else {"status": "NOT_DUE"}
    )
    episode_outcomes = (
        _observed_command_step(
            project_root,
            cycle_id,
            "ENTRY_EPISODE_OUTCOMES",
            ("market", "context", "settle-episodes"),
            timeout_seconds=120,
        )
        if fast_path_due
        else {"status": "NOT_DUE"}
    )
    active_swing_sprints = _delegated_primary(primary_background, "ACTIVE_SWING_SPRINTS")
    role_leaderboards = _delegated_primary(primary_background, "ROLE_LEADERBOARDS")
    if observed_positions_present:
        portfolio = priority_portfolio
    elif fast_path_due:
        portfolio = _observed_command_step(
            project_root,
            cycle_id,
            "PORTFOLIO_MANAGER",
            ("portfolio", "plan"),
            timeout_seconds=120,
        )
    else:
        portfolio = {"status": "NOT_DUE"}
    ai_decision_intelligence = (
        _observed_command_step(
            project_root,
            cycle_id,
            "AI_DECISION_INTELLIGENCE",
            ("ai", "enqueue-refresh"),
            timeout_seconds=30,
        )
        if _is_success_status(portfolio)
        else {"status": "NOT_DUE"}
    )
    exit_notification = _observed_command_step(
        project_root,
        cycle_id,
        "EXIT_NOTIFICATION",
        ("telegram", "send-exit-signals"),
        timeout_seconds=60,
    )
    research_due = _research_due(project_root)
    if research_due:
        _cycle_progress(
            project_root,
            cycle_id,
            "RESEARCH_BACKGROUND",
            "STARTED",
        )
        research = launch_background_job(
            project_root,
            "research",
            ("autopilot", "run-once"),
            timeout_seconds=7_200,
        )
        _cycle_progress(
            project_root,
            cycle_id,
            "RESEARCH_BACKGROUND",
            "COMPLETED",
            component_status=_normalized_status(research),
        )
    else:
        research = {"status": "NOT_DUE"}
    research_background = (
        research if research_due else background_job_status(project_root, "research")
    )
    p3_evidence = _delegated_primary(primary_background, "P3_EVIDENCE")
    rl_shadow = _delegated_primary(primary_background, "RL_SHADOW")
    p4_evidence = _delegated_primary(primary_background, "P4_EVIDENCE")
    hmm_notification = _delegated_primary(primary_background, "HMM_NOTIFICATION")
    telegram = _observed_command_step(
        project_root,
        cycle_id,
        "TELEGRAM",
        ("telegram", "retry-failed"),
        timeout_seconds=60,
    )
    blockers: list[str] = []
    operational_steps = {
        "RECONCILIATION": reconciliation,
        "MARKET_DATA_15M": tactical_data_refresh,
        "ACTIVE_SWING_CONTEXT_BOOTSTRAP": active_swing_context_bootstrap,
        "ACTIVE_SWING_15M_CANDIDATES": active_swing_candidates,
        "MARKET_DATA": data_refresh,
        "DAILY": daily,
        "HMM_REGIME": hmm_regime,
        "DYNAMIC": dynamic,
        "SIGNAL_LIFECYCLE": lifecycle,
        "MULTITIMEFRAME_WATCHLIST": multitimeframe_watchlist,
        "MTF_PIT_OBSERVATION": multitimeframe_pit_observation,
        "FAST_TRACK_OBSERVATION": fast_track_observation,
        "BROAD_SHADOW_OBSERVATION": broad_shadow_observation,
        "SURVIVOR_SHADOW_OBSERVATION": survivor_shadow_observation,
        "BROAD_SHADOW_NOTIFICATION": broad_shadow_notification,
        "MACRO": macro,
        "NEWS_DIGEST": news_digest,
        "MARKET_CONTEXT": market_context,
        "COT_CONTEXT": cot_context,
        "ASSET_CONTEXT": asset_context,
        "ENTRY_OBSERVER": entry_observer,
        "ENTRY_EPISODE_OUTCOMES": episode_outcomes,
        "ACTIVE_SWING_SPRINTS": active_swing_sprints,
        "ROLE_LEADERBOARDS": role_leaderboards,
        "PORTFOLIO_MANAGER": portfolio,
        "AI_DECISION_INTELLIGENCE": ai_decision_intelligence,
        "EXIT_NOTIFICATION": exit_notification,
        "RESEARCH": research,
        "P3_EVIDENCE": p3_evidence,
        "RL_SHADOW": rl_shadow,
        "P4_EVIDENCE": p4_evidence,
        "HMM_NOTIFICATION": hmm_notification,
        "TELEGRAM": telegram,
    }
    for step_name, payload in operational_steps.items():
        blockers.extend(_step_blockers(step_name, payload))
    authority = "NONE"
    orders_generated = 0
    live_writer: dict[str, Any] = {"status": "NOT_APPLICABLE"}
    if not phase1.frozen:
        blockers.append("PHASE1_FREEZE_INTEGRITY_BLOCKED")
    if mode == "PAPER_AUTOMATIC":
        preflight = automatic_paper_preflight(project_root, ".env.ibkr")
        blockers.extend(preflight["blockers"])
        if not blockers:
            authority = AUTO_PAPER_AUTHORITY
    elif mode in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"}:
        live_gate = live_preflight(
            project_root,
            env_file=".env.ibkr.live",
        )
        live_authority = authority_status(project_root)
        blockers.extend(live_gate.get("blockers", []))
        active_live_authority = str(live_authority.get("execution_authority") or "NONE")
        if active_live_authority not in {
            LIVE_LEVEL_ONE,
            AUTONOMOUS_LEVEL_ONE,
        }:
            blockers.append("LIVE_LEVEL_ONE_AUTHORITY_NOT_ACTIVE")
        if not blockers:
            authority = active_live_authority
            live_writer = _step(
                lambda: automatic_cycle(
                    project_root,
                    env_file=".env.ibkr.live",
                    preflight_report=live_gate,
                )
            )
            if live_writer.get("status") != "GO":
                blockers.append("AUTOMATIC_LIVE_WRITER_CYCLE_BLOCKED")
    _cycle_progress(project_root, cycle_id, "PAPER_PLANS", "STARTED")
    paper_plans: dict[str, Any]
    paper_writer: dict[str, Any]
    if mode in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"}:
        paper_plans = {"status": "NOT_APPLICABLE_LIVE_MODE"}
        paper_writer = {
            "status": "NOT_APPLICABLE_LIVE_MODE",
            "paper_place_order_calls": 0,
            "live_place_order_calls": 0,
        }
    else:
        paper_plans = _step(
            lambda: plan_paper_cycle(
                project_root,
                dynamic=dynamic,
                lifecycle=lifecycle,
                execution_authority=authority,
            )
        )
        paper_writer = _step(
            lambda: execute_automatic_paper_cycle(
                project_root,
                execution_authority=authority,
                env_file=".env.ibkr",
                preflight=(
                    preflight
                    if mode == "PAPER_AUTOMATIC"
                    else {
                        "status": "NO_GO",
                        "blockers": ["PAPER_MODE_NOT_ACTIVE"],
                    }
                ),
            )
        )
    if mode == "PAPER_AUTOMATIC" and authority != "NONE":
        writer_blockers = paper_writer.get("preflight_blockers", [])
        if isinstance(writer_blockers, list):
            blockers.extend(str(item) for item in writer_blockers)
        else:
            blockers.append("AUTOMATIC_PAPER_WRITER_BLOCKERS_INVALID")
        if paper_writer.get("status") != "GO":
            blockers.append("AUTOMATIC_PAPER_WRITER_CYCLE_BLOCKED")
    orders_generated = (
        int(live_writer.get("live_place_order_calls", 0))
        if mode in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"}
        else int(paper_writer.get("paper_place_order_calls", 0))
    )
    _cycle_progress(
        project_root,
        cycle_id,
        "PAPER_PLANS",
        "COMPLETED",
        component_status=_normalized_status(paper_plans),
    )
    completed = datetime.now(UTC)
    daily_performance = _record_daily_performance(project_root)
    cycle_status = "GO" if not blockers else "DEGRADED"
    record = {
        "schema": "stocks_machine_cycle_v1",
        "status": cycle_status,
        "cycle_id": cycle_id,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_seconds": (completed - started).total_seconds(),
        "process_id": os.getpid(),
        "requested_mode": requested_mode,
        "effective_mode": mode,
        "phase1_freeze_integrity": phase1.status,
        "intraday_refresh_plan": intraday_refresh_plan,
        "scheduling": {
            "schema": "active_swing_cadence_decision_v1",
            "tactical_15m_due": tactical_due,
            "primary_1h_due": primary_due,
            "fast_path_due": fast_path_due,
            "observed_positions_present": observed_positions_present,
            "tactical_cadence_hours": tactical_cadence,
            "primary_cadence_hours": primary_cadence,
            "portfolio_cadence_hours": portfolio_cadence,
            "heavy_research_due": research_due,
            "heavy_research_execution": "DETACHED_BACKGROUND_JOB",
            "heavy_research_resource_priority": "BELOW_NORMAL",
            "heavy_research_can_block_money_loop": False,
            "primary_context_execution": "DETACHED_BACKGROUND_JOB",
            "primary_context_resource_priority": "BELOW_NORMAL",
            "primary_context_can_block_money_loop": False,
            "reconciliation_executed_before_market_data": True,
            "position_risk_executed_before_discovery": bool(observed_positions_present),
            "tactical_candidate_path": "DEDICATED_15M_OBSERVER_ARTIFACT",
            "tactical_context_bootstrap": ("MISSING_OR_STALE_1H_4H_GAPS_ONLY"),
            "tactical_context_gap_count": len(context_gap_symbols),
            "full_dynamic_scan_tactical_path_allowed": False,
            "full_dynamic_scan_primary_path_due": bool(primary_due),
            "full_dynamic_scan_foreground_allowed": False,
            "position_management_priority": 1,
            "new_entry_evaluation_priority": 4,
            "heavy_research_priority": 5,
        },
        "tactical_market_data": _summary(tactical_data_refresh),
        "active_swing_context_gap_audit": active_swing_context_gap_audit,
        "active_swing_context_bootstrap": _summary(active_swing_context_bootstrap),
        "active_swing_15m_candidates": _summary(active_swing_candidates),
        "market_data": _summary(data_refresh),
        "daily": _summary(daily),
        "hmm_regime": _summary(hmm_regime),
        "dynamic": _summary(dynamic),
        "signal_lifecycle": {
            "status": lifecycle["status"],
            "fresh_entry_count": lifecycle["fresh_entry_count"],
            "exit_count": lifecycle["exit_count"],
        },
        "multitimeframe_watchlist": _summary(multitimeframe_watchlist),
        "multitimeframe_pit_observation": _summary(multitimeframe_pit_observation),
        "multitimeframe_pit_notification": _summary(multitimeframe_pit_notification),
        "fast_track_observation": _summary(fast_track_observation),
        "broad_shadow_observation": _summary(broad_shadow_observation),
        "survivor_shadow_observation": _summary(survivor_shadow_observation),
        "broad_shadow_notification": _summary(broad_shadow_notification),
        "macro": _summary(macro),
        "news_digest": _summary(news_digest),
        "market_context": _summary(market_context),
        "cot_context": _summary(cot_context),
        "asset_context": _summary(asset_context),
        "entry_observer": _summary(entry_observer),
        "entry_observer_state_counts": entry_observer.get("state_counts", {}),
        "entry_episode_outcomes": _summary(episode_outcomes),
        "entry_episode_completion_ratio": episode_outcomes.get("completion_ratio"),
        "active_swing_sprints": _summary(active_swing_sprints),
        "role_leaderboards": _summary(role_leaderboards),
        "market_context_policy": ("CONTEXT_ONLY_NO_STANDALONE_ENTRY_AUTHORITY"),
        "macro_context_policy": (
            "OPTIONAL_PARTIAL_CONTEXT_NON_BLOCKING"
            if _macro_context_usable(macro) and _normalized_status(macro) == "DATA_INCOMPLETE"
            else "STANDARD"
        ),
        "reconciliation": _summary(reconciliation),
        "reconciliation_environment": (
            "LIVE_READ_ONLY"
            if reconciliation_arguments == ("live", "reconcile")
            else "PAPER_READ_ONLY"
        ),
        "reconciliation_status": reconciliation.get("reconciliation_status"),
        "reconciliation_operationally_usable": (_reconciliation_read_only_usable(reconciliation)),
        "reconciliation_operational_broker_state": reconciliation.get(
            "operational_broker_state_status"
        ),
        "reconciliation_historical_execution_evidence": reconciliation.get(
            "canonical_execution_evidence_status"
        ),
        "reconciliation_historical_quarantine_status": reconciliation.get(
            "historical_orphan_quarantine_status"
        ),
        "reconciliation_historical_execution_blocks_trading": (
            reconciliation.get("canonical_execution_evidence_status")
            == "INCOMPLETE_HISTORICAL_EXECUTION_CHAIN"
        ),
        "portfolio_manager": _summary(portfolio),
        "portfolio_priority_path": (
            "POSITION_RISK_FIRST" if observed_positions_present else "DISCOVERY_AFTER_CONTEXT"
        ),
        "ai_decision_intelligence": _summary(ai_decision_intelligence),
        "exit_notification": _summary(exit_notification),
        "daily_performance": _summary(daily_performance),
        "research": _summary(research),
        "research_background": _summary(research_background),
        "primary_background": _summary(primary_background),
        "p3_evidence": _summary(p3_evidence),
        "rl_shadow": _summary(rl_shadow),
        "p4_evidence": _summary(p4_evidence),
        "research_data_policy": (
            "PIT_ELIGIBILITY_UNAVAILABLE_FAIL_CLOSED_NON_OPERATIONAL"
            if _research_data_blocked_expected(research)
            else "STANDARD"
        ),
        "hmm_notification": _summary(hmm_notification),
        "telegram": _summary(telegram),
        "paper_plans": _summary(paper_plans),
        "paper_writer": _summary(paper_writer),
        "live_writer": _summary(live_writer),
        "live_writer_cycle_status": live_writer.get("cycle_status"),
        "paper_writer_blocked_reasons": sorted(
            {
                str(item.get("reason"))
                for item in paper_writer.get("blocked_plans", [])
                if isinstance(item, dict) and item.get("reason")
            }
        ),
        "blockers": sorted(set(blockers)),
        "execution_authority": authority,
        "orders_generated": orders_generated,
        "broker_writes": orders_generated,
        "paper_place_order_calls": (
            0 if mode in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"} else orders_generated
        ),
        "live_place_order_calls": (
            orders_generated
            if mode in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"}
            else int(paper_writer.get("live_place_order_calls", 0))
        ),
    }
    state.update(
        last_heartbeat=completed.isoformat(),
        last_cycle_id=record["cycle_id"],
        last_cycle_status=cycle_status,
        last_cycle_blockers=record["blockers"],
        cycle_count=int(state.get("cycle_count", 0)) + 1,
    )
    if _is_success_status(data_refresh):
        state["last_data_refresh"] = completed.isoformat()
        state["intraday_refresh_cursor"] = intraday_refresh_plan["next_cursor"]
    if _is_success_status(tactical_data_refresh):
        state["last_tactical_data_refresh"] = completed.isoformat()
        state["intraday_refresh_cursor"] = intraday_refresh_plan["next_cursor"]
    if context_gap_symbols and _is_success_status(active_swing_context_bootstrap):
        state["last_active_swing_context_bootstrap"] = completed.isoformat()
    if _is_success_status(active_swing_candidates):
        state["last_active_swing_candidate_refresh"] = completed.isoformat()
    if _is_success_status(daily):
        state["last_daily_refresh"] = completed.isoformat()
    if _is_success_status(hmm_regime):
        state["last_hmm_regime_refresh"] = completed.isoformat()
    if _is_success_status(dynamic):
        state["last_dynamic_refresh"] = completed.isoformat()
    if _is_success_status(multitimeframe_watchlist):
        state["last_multitimeframe_watchlist_refresh"] = completed.isoformat()
    if _is_success_status(multitimeframe_pit_observation):
        state["last_multitimeframe_pit_observation_refresh"] = completed.isoformat()
    if _is_success_status(fast_track_observation):
        state["last_fast_track_observation_refresh"] = completed.isoformat()
    if _is_success_status(broad_shadow_observation):
        state["last_broad_shadow_observation_refresh"] = completed.isoformat()
    if _is_success_status(survivor_shadow_observation):
        state["last_survivor_shadow_observation_refresh"] = completed.isoformat()
    if _macro_context_usable(macro):
        state["last_macro_refresh"] = completed.isoformat()
    if _news_context_usable(news_digest):
        state["last_news_refresh"] = completed.isoformat()
    if _is_success_status(market_context):
        state["last_market_context_refresh"] = completed.isoformat()
    if _is_success_status(cot_context):
        state["last_cot_context_refresh"] = completed.isoformat()
    if _is_success_status(asset_context):
        state["last_asset_context_refresh"] = completed.isoformat()
    if _is_success_status(entry_observer):
        state["last_entry_observer_refresh"] = completed.isoformat()
    if _is_success_status(episode_outcomes):
        state["last_entry_episode_outcomes_refresh"] = completed.isoformat()
    if _is_success_status(active_swing_sprints):
        state["last_active_swing_sprints_refresh"] = completed.isoformat()
    if _is_success_status(portfolio):
        state["last_portfolio_refresh"] = completed.isoformat()
    if _is_success_status(role_leaderboards):
        state["last_role_leaderboards_refresh"] = completed.isoformat()
    if _is_success_status(p3_evidence):
        state["last_p3_evidence_refresh"] = completed.isoformat()
    if _rl_shadow_usable(rl_shadow):
        state["last_rl_shadow_refresh"] = completed.isoformat()
    if _is_success_status(p4_evidence):
        state["last_p4_evidence_refresh"] = completed.isoformat()
    if _normalized_status(primary_background) in {
        "COMPLETED",
        "ENQUEUED",
        "RUNNING",
        "SKIPPED_BUSY",
    }:
        state["last_primary_background_refresh"] = completed.isoformat()
    if _is_success_status(reconciliation) or _reconciliation_read_only_usable(reconciliation):
        state["last_account_reconciliation"] = completed.isoformat()
        state["last_position_check"] = completed.isoformat()
        state["last_ibkr_status"] = (
            str(reconciliation.get("operational_broker_state_status"))
            if _reconciliation_read_only_usable(reconciliation)
            else _normalized_status(reconciliation)
        )
        state["last_open_positions"] = reconciliation.get(
            "broker_position_count",
            reconciliation.get("position_count"),
        )
        state["last_open_orders"] = reconciliation.get(
            "broker_open_order_count",
            reconciliation.get("open_order_count"),
        )
    _save_state(project_root, state)
    _append_cycle(project_root, record)
    _publish(project_root, "last-cycle.json", record)
    _machine_status(project_root, state)
    _cycle_progress(
        project_root,
        cycle_id,
        "CYCLE",
        "COMPLETED",
        component_status=cycle_status,
    )
    return record


def _execution_preflight(
    project_root: Path, environment: str, env_file: str | Path
) -> dict[str, Any]:
    if environment == "live":
        report = live_preflight(
            project_root,
            env_file=env_file,
            probe_socket=False,
        )
        return {
            "schema": "execution_preflight_v1",
            "status": report["status"],
            "environment": "live",
            "blockers": report["blockers"],
            "execution_authority": "NONE",
            "automatic_orders_allowed": False,
        }
    report = automatic_paper_preflight(project_root, env_file)
    return {
        **report,
        "schema": "execution_preflight_v1",
        "environment": "paper",
        "activation_phrase": PAPER_ACTIVATION_PHRASE,
        "execution_authority": "NONE",
    }


def _execution_status(project_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    phase9 = phase9_status(project_root)
    paper_ready = phase9.get("status") == PHASE9_MARKER and all(
        bool(phase9.get("checks", {}).get(check))
        for check in (
            "submit_cancel_canary",
            "fill_canary",
            "closing_sell_canary",
        )
    )
    mode = str(state.get("mode", "SIGNALS_ONLY"))
    live_authority = authority_status(project_root)
    authority = (
        "AUTOMATIC_BOUNDED_PAPER"
        if mode == "PAPER_AUTOMATIC" and paper_ready
        else str(live_authority.get("execution_authority"))
        if mode in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"}
        and live_authority.get("execution_authority") in {LIVE_LEVEL_ONE, AUTONOMOUS_LEVEL_ONE}
        else "NONE"
    )
    return _publish(
        project_root,
        "execution-status.json",
        {
            "schema": "bounded_execution_status_v1",
            "status": "GO",
            "mode": mode,
            "paper_enabled": bool(state.get("paper_enabled")),
            "live_enabled": bool(state.get("live_enabled")),
            "paper_fill_close_canary_go": paper_ready,
            "paper_reconciliation_go": phase9.get("checks", {}).get("reconciliation", False),
            "execution_authority": authority,
            "strategy_authority": ("FROZEN_DYNAMIC_ALLOWLIST" if authority != "NONE" else "NONE"),
            "paper_limits": PAPER_LIMITS,
            "live_limits": LIVE_LIMITS,
            "margin_enabled": False,
            "shorting_enabled": False,
            "options_enabled": False,
            "futures_live_enabled": False,
            "live_autoscale_enabled": False,
            "live_authority": live_authority,
            "real_live_order_placed": False,
        },
    )


def _effective_mode(state: dict[str, Any], requested: str) -> str:
    if requested == "PAPER_AUTOMATIC" and not state.get("paper_enabled"):
        return "SIGNALS_ONLY"
    if requested in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"} and not state.get("live_enabled"):
        return "SIGNALS_ONLY"
    return requested


def _machine_status(project_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    last_cycle_status = str(state.get("last_cycle_status", "NEVER_RUN"))
    return _publish(
        project_root,
        "machine-status.json",
        {
            "schema": "stocks_machine_status_v1",
            "status": ("DEGRADED" if last_cycle_status == "DEGRADED" else "GO"),
            **state,
            "single_instance_lock": str(_private_root(project_root) / "machine.lock"),
            "append_only_cycle_log": str(_private_root(project_root) / "cycles.jsonl"),
        },
    )


def _state(project_root: Path) -> dict[str, Any]:
    path = _private_root(project_root) / "state.json"
    current = _read_json(path)
    return {
        "enabled": bool(current.get("enabled", False)),
        "paused": bool(current.get("paused", False)),
        "mode": str(current.get("mode", "SIGNALS_ONLY")),
        "requested_mode": str(current.get("requested_mode", "SIGNALS_ONLY")),
        "paper_enabled": bool(current.get("paper_enabled", False)),
        "live_enabled": bool(current.get("live_enabled", False)),
        "activated_at": current.get("activated_at"),
        "last_heartbeat": current.get("last_heartbeat"),
        "last_cycle_id": current.get("last_cycle_id"),
        "last_cycle_status": current.get("last_cycle_status", "NEVER_RUN"),
        "last_cycle_blockers": list(current.get("last_cycle_blockers", [])),
        "cycle_count": int(current.get("cycle_count", 0)),
        "last_data_refresh": current.get("last_data_refresh"),
        "last_tactical_data_refresh": current.get("last_tactical_data_refresh"),
        "last_active_swing_candidate_refresh": current.get("last_active_swing_candidate_refresh"),
        "intraday_refresh_cursor": int(current.get("intraday_refresh_cursor", 0)),
        "last_daily_refresh": current.get("last_daily_refresh"),
        "last_dynamic_refresh": current.get("last_dynamic_refresh"),
        "last_multitimeframe_watchlist_refresh": current.get(
            "last_multitimeframe_watchlist_refresh"
        ),
        "last_multitimeframe_pit_observation_refresh": current.get(
            "last_multitimeframe_pit_observation_refresh"
        ),
        "last_fast_track_observation_refresh": current.get("last_fast_track_observation_refresh"),
        "last_broad_shadow_observation_refresh": current.get(
            "last_broad_shadow_observation_refresh"
        ),
        "last_survivor_shadow_observation_refresh": current.get(
            "last_survivor_shadow_observation_refresh"
        ),
        "last_macro_refresh": current.get("last_macro_refresh"),
        "last_news_refresh": current.get("last_news_refresh"),
        "last_market_context_refresh": current.get("last_market_context_refresh"),
        "last_cot_context_refresh": current.get("last_cot_context_refresh"),
        "last_asset_context_refresh": current.get("last_asset_context_refresh"),
        "last_entry_observer_refresh": current.get("last_entry_observer_refresh"),
        "last_entry_episode_outcomes_refresh": current.get("last_entry_episode_outcomes_refresh"),
        "last_active_swing_sprints_refresh": current.get("last_active_swing_sprints_refresh"),
        "last_portfolio_refresh": current.get("last_portfolio_refresh"),
        "last_role_leaderboards_refresh": current.get("last_role_leaderboards_refresh"),
        "last_p3_evidence_refresh": current.get("last_p3_evidence_refresh"),
        "last_hmm_regime_refresh": current.get("last_hmm_regime_refresh"),
        "last_account_reconciliation": current.get("last_account_reconciliation"),
        "last_position_check": current.get("last_position_check"),
        "last_ibkr_status": current.get("last_ibkr_status"),
        "last_open_positions": current.get("last_open_positions"),
        "last_open_orders": current.get("last_open_orders"),
    }


def _save_state(project_root: Path, state: dict[str, Any]) -> None:
    path = _private_root(project_root) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, state)


def _cadence_hours(
    project_root: Path,
    name: str,
    fallback: float,
) -> float:
    profile = _read_json(project_root / OPERATIONS_PROFILE_PATH)
    value = profile.get("runtime", {}).get("cadence_hours", {}).get(name, fallback)
    try:
        cadence = float(value)
    except (TypeError, ValueError):
        cadence = fallback
    return max(1.0 / 60.0, cadence)


def _has_observed_positions(project_root: Path) -> bool:
    allocation = _read_json(project_root / "output" / "portfolio" / "current_allocation.json")
    positions = allocation.get("positions", [])
    return bool(isinstance(positions, list) and positions)


def _intraday_refresh_plan(
    project_root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    ranking = _read_json(project_root / "output" / "portfolio" / "opportunity_ranking.json")
    opportunities = ranking.get("opportunities", [])
    core = list(INTRADAY_CORE_SYMBOLS)
    current_allocation = _read_json(
        project_root / "output" / "portfolio" / "current_allocation.json"
    )
    position_symbols: list[str] = []
    for row in current_allocation.get("positions", []):
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        if symbol and len(symbol) <= 20 and symbol not in position_symbols:
            position_symbols.append(symbol)
    mandatory_symbols = list(dict.fromkeys([*position_symbols, *core]))
    mandatory_set = set(mandatory_symbols)
    ranked_symbols: list[str] = []
    for row in opportunities:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("ticker", "")).strip().upper()
        if not symbol or symbol in mandatory_set or symbol in ranked_symbols or len(symbol) > 20:
            continue
        ranked_symbols.append(symbol)
        if len(ranked_symbols) >= INTRADAY_CHALLENGER_POOL_LIMIT:
            break
    opportunity_pool_count = len(ranked_symbols)

    signal_path = project_root / "output" / "signals" / "latest_signals.json"
    try:
        signal_payload = json.loads(signal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        signal_payload = {}
    signal_rows = (
        signal_payload
        if isinstance(signal_payload, list)
        else signal_payload.get("signals", [])
        if isinstance(signal_payload, dict)
        else []
    )
    now = datetime.now(UTC)
    signal_candidates: dict[str, float] = {}
    stale_signal_symbols: set[str] = set()
    for row in signal_rows:
        if not isinstance(row, dict):
            continue
        freshness = str(row.get("data_freshness", "")).upper()
        original_action = str(row.get("original_action") or row.get("action") or "").upper()
        refreshable_stale = (
            freshness == "STALE"
            and original_action in {"BUY", "STRONG_BUY", "WATCHLIST"}
            and str(row.get("price_validity_status", ""))
            == "CURRENT_MARKET_REFERENCE_UNAVAILABLE_OR_STALE"
        )
        if freshness != "FRESH" and not refreshable_stale:
            continue
        symbol = str(row.get("ticker") or row.get("asset") or "").strip().upper()
        if not symbol or symbol in mandatory_set or len(symbol) > 20:
            continue
        expiration = _utc_timestamp(row.get("expiration_timestamp"))
        if expiration is None or expiration < now:
            continue
        try:
            score = float(row.get("confidence_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        signal_candidates[symbol] = max(score, signal_candidates.get(symbol, float("-inf")))
        if refreshable_stale:
            stale_signal_symbols.add(symbol)
    ranked_signal_symbols = [
        symbol
        for symbol, _score in sorted(
            signal_candidates.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    priority: list[str] = []
    for symbol in [*ranked_symbols, *ranked_signal_symbols]:
        if symbol in priority:
            continue
        priority.append(symbol)
        if len(priority) >= INTRADAY_PRIORITY_SYMBOL_LIMIT:
            break
    priority_set = set(priority)
    pool: list[str] = []
    for symbol in [*ranked_symbols, *ranked_signal_symbols]:
        if symbol in priority_set or symbol in pool:
            continue
        pool.append(symbol)
        if len(pool) >= INTRADAY_CHALLENGER_POOL_LIMIT:
            break
    if not priority and not pool:
        return {
            "schema": "intraday_rotating_refresh_plan_v1",
            "symbols": list(dict.fromkeys([*core, *position_symbols])),
            "core_symbol_count": len(core),
            "position_symbols": position_symbols,
            "position_symbol_count": len(position_symbols),
            "priority_symbols": [],
            "priority_symbol_count": 0,
            "challenger_symbols": [],
            "challenger_pool_count": 0,
            "opportunity_pool_count": 0,
            "signal_pool_count": 0,
            "cursor": 0,
            "next_cursor": 0,
            "rotation_status": "CORE_ONLY_NO_CHALLENGER_POOL",
            "execution_authority": "NONE",
            "broker_calls": 0,
        }
    cursor = int(state.get("intraday_refresh_cursor", 0)) % len(pool) if pool else 0
    batch_size = min(INTRADAY_CHALLENGER_BATCH_SIZE, len(pool))
    challengers = [pool[(cursor + index) % len(pool)] for index in range(batch_size)]
    next_cursor = (cursor + batch_size) % len(pool) if pool else 0
    remaining = max(0, INTRADAY_COLLECTION_SYMBOL_LIMIT - len(mandatory_symbols))
    selected_priority = priority[:remaining]
    remaining -= len(selected_priority)
    selected_challengers = challengers[:remaining]
    selected_symbols = [
        *mandatory_symbols,
        *selected_priority,
        *selected_challengers,
    ][:INTRADAY_COLLECTION_SYMBOL_LIMIT]
    return {
        "schema": "intraday_rotating_refresh_plan_v1",
        "symbols": selected_symbols,
        "collection_symbol_limit": INTRADAY_COLLECTION_SYMBOL_LIMIT,
        "core_symbol_count": len(core),
        "position_symbols": position_symbols,
        "position_symbol_count": len(position_symbols),
        "priority_symbols": selected_priority,
        "priority_symbol_count": len(selected_priority),
        "challenger_symbols": selected_challengers,
        "challenger_pool_count": len(pool),
        "opportunity_pool_count": opportunity_pool_count,
        "signal_pool_count": len(ranked_signal_symbols),
        "stale_signal_refresh_symbol_count": len(stale_signal_symbols),
        "cursor": cursor,
        "next_cursor": next_cursor,
        "rotation_status": (
            "PRIORITY_AND_ROTATING_CHALLENGER_BATCH_GO" if challengers else "PRIORITY_REFRESH_GO"
        ),
        "execution_authority": "NONE",
        "broker_calls": 0,
    }


def _market_context_symbols(
    intraday_symbols: list[str],
) -> list[str]:
    symbols: list[str] = []
    for value in [*intraday_symbols, *DEFAULT_CONTEXT_SYMBOLS]:
        symbol = str(value).strip().upper()
        if not symbol or symbol in symbols or len(symbol) > 20:
            continue
        symbols.append(symbol)
        if len(symbols) >= MARKET_CONTEXT_SYMBOL_LIMIT:
            break
    return symbols


def _record_daily_performance(project_root: Path) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    paper = _read_json(project_root / "output" / "ibkr" / "phase9" / "position-ledger-audit.json")
    projection = paper.get("partial_close_projection", {})
    _add_performance_candidate(
        candidates,
        environment="PAPER",
        source="PAPER_LEDGER_AUDIT",
        observed_at=projection.get("last_updated_at"),
        realized=projection.get("realized_pnl_eur"),
        unrealized=None,
        evidence="HISTORICAL_PAPER_PROJECTION",
    )

    live = _read_json(project_root / "output" / "ibkr" / "live" / "performance.json")
    _add_performance_candidate(
        candidates,
        environment="LIVE",
        source="RECONCILED_LIVE_PERFORMANCE",
        observed_at=live.get("generated_at"),
        realized=live.get("realized_pnl_eur"),
        unrealized=live.get("unrealized_pnl_eur"),
        evidence="BROKER_DERIVED",
    )

    path = project_root / "data" / "performance" / "private" / "daily-pnl.jsonl"
    existing_hashes: set[str] = set()
    existing_count = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing_count += 1
            existing_hashes.add(str(row.get("content_hash", "")))
    appended = [row for row in candidates if row["content_hash"] not in existing_hashes]
    if appended:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in appended:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    status = {
        "schema": "daily_performance_capture_status_v1",
        "status": "GO" if candidates else "NO_VERIFIED_PNL_AVAILABLE",
        "verified_candidate_count": len(candidates),
        "records_appended": len(appended),
        "record_count": existing_count + len(appended),
        "operator_supplied_values_counted": 0,
        "unavailable_values_counted_as_zero": 0,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    output = project_root / "output" / "performance" / "status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, status)
    return status


def _add_performance_candidate(
    target: list[dict[str, Any]],
    *,
    environment: str,
    source: str,
    observed_at: Any,
    realized: Any,
    unrealized: Any,
    evidence: str,
) -> None:
    timestamp = _utc_timestamp(observed_at)
    realized_value = _numeric_or_none(realized)
    unrealized_value = _numeric_or_none(unrealized)
    if timestamp is None or (realized_value is None and unrealized_value is None):
        return
    payload = {
        "schema": "daily_performance_record_v1",
        "session_date": timestamp.astimezone(ZoneInfo("Europe/Amsterdam")).date().isoformat(),
        "observed_at": timestamp.isoformat(),
        "environment": environment,
        "source": source,
        "evidence": evidence,
        "realized_pnl_eur": realized_value,
        "unrealized_pnl_eur": unrealized_value,
        "net_pnl_eur": (realized_value or 0.0) + (unrealized_value or 0.0),
        "account_identifier_stored": False,
        "execution_authority": "NONE",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["content_hash"] = hashlib.sha256(canonical.encode()).hexdigest().upper()
    target.append(payload)


def _numeric_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _append_cycle(project_root: Path, record: dict[str, Any]) -> None:
    path = _private_root(project_root) / "cycles.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _publish(project_root: Path, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = project_root / "output" / "operations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        last_error: PermissionError | None = None
        for delay in (0.0, 0.01, 0.05, 0.1, 0.25):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temp, path)
                return
            except PermissionError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _step(call: Any) -> dict[str, Any]:
    try:
        result = call()
        return result if isinstance(result, dict) else {"status": "GO"}
    except Exception as exc:
        return {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "detail": str(exc)[:300],
        }


def _command_step(
    project_root: Path,
    arguments: tuple[str, ...],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    python = project_root / ".venv-ibkr" / "Scripts" / "python.exe"
    if not python.exists():
        return {
            "status": "ERROR",
            "error_type": "PYTHON_RUNTIME_NOT_FOUND",
        }
    command = [str(python), str(project_root / "main.py"), *arguments]
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=max(1, timeout_seconds))
    except subprocess.TimeoutExpired:
        process_tree_terminated = _terminate_process_tree(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            process_tree_terminated = False
        return {
            "status": "TIMEOUT",
            "command": list(arguments),
            "timeout_seconds": timeout_seconds,
            "process_tree_terminated": process_tree_terminated,
            "stdout_tail": stdout[-300:],
            "stderr_tail": stderr[-300:],
        }
    completed_return_code = int(process.returncode or 0)
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "status": "ERROR",
            "error_type": "INVALID_STEP_OUTPUT",
            "command": list(arguments),
            "process_return_code": completed_return_code,
            "stderr": stderr[-300:],
        }
    if not isinstance(payload, dict):
        return {
            "status": "ERROR",
            "error_type": "INVALID_STEP_PAYLOAD",
            "command": list(arguments),
        }
    payload["process_return_code"] = completed_return_code
    if completed_return_code != 0 and _is_success_status(payload):
        payload["status"] = "ERROR"
        payload["error_type"] = "NON_ZERO_PROCESS_EXIT"
    return payload


def _observed_command_step(
    project_root: Path,
    cycle_id: str,
    step_name: str,
    arguments: tuple[str, ...],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    _cycle_progress(project_root, cycle_id, step_name, "STARTED")
    result = _command_step(
        project_root,
        arguments,
        timeout_seconds=timeout_seconds,
    )
    result["duration_seconds"] = round(time.monotonic() - started, 6)
    _cycle_progress(
        project_root,
        cycle_id,
        step_name,
        "COMPLETED",
        component_status=_normalized_status(result),
    )
    return result


def _cycle_progress(
    project_root: Path,
    cycle_id: str,
    step_name: str,
    step_status: str,
    *,
    component_status: str | None = None,
) -> None:
    updated_at = _now()
    _publish(
        project_root,
        "cycle-progress.json",
        {
            "schema": "stocks_machine_cycle_progress_v1",
            "cycle_id": cycle_id,
            "updated_at": updated_at,
            "step": step_name,
            "step_status": step_status,
            "component_status": component_status,
            "execution_authority": "NONE",
            "broker_writes": 0,
        },
    )
    _runtime_heartbeat(
        project_root,
        "RUNNING"
        if step_status != "COMPLETED" or step_name != "CYCLE"
        else (component_status or "COMPLETED"),
        cycle_id=cycle_id,
        step=step_name,
        step_status=step_status,
        component_status=component_status,
        updated_at=updated_at,
    )


def _runtime_heartbeat(
    project_root: Path,
    runtime_status: str,
    *,
    state: dict[str, Any] | None = None,
    cycle_id: str | None = None,
    step: str | None = None,
    step_status: str | None = None,
    component_status: str | None = None,
    updated_at: str | None = None,
) -> None:
    current = state or _state(project_root)
    path = project_root / "runtime" / "heartbeat.json"
    _atomic_json(
        path,
        {
            "schema": "stocks_runtime_heartbeat_v1",
            "process_id": os.getpid(),
            "started_at": current.get("activated_at") or updated_at or _now(),
            "last_heartbeat": updated_at or _now(),
            "updated_at": updated_at or _now(),
            "runtime_status": runtime_status,
            "runtime_state": (
                "FULL_STOP"
                if runtime_status == "STOPPED"
                else "ENTRY_BLOCKED"
                if str(current.get("requested_mode")) != "SIGNALS_ONLY"
                else "NORMAL"
            ),
            "enabled": bool(current.get("enabled", False)),
            "paused": bool(current.get("paused", False)),
            "requested_mode": str(current.get("requested_mode", "SIGNALS_ONLY")),
            "cycle_id": cycle_id or current.get("last_cycle_id"),
            "step": step,
            "step_status": step_status,
            "component_status": component_status,
            "IBKR_status": current.get("last_ibkr_status") or "NOT_YET_OBSERVED",
            "last_account_reconciliation": current.get("last_account_reconciliation"),
            "last_market_data": current.get("last_data_refresh"),
            "last_order_update": None,
            "last_position_check": current.get("last_position_check"),
            "market_state": "CALENDAR_GATE_DELEGATED",
            "open_positions": current.get("last_open_positions"),
            "open_orders": current.get("last_open_orders"),
            "kill_switch": "UNKNOWN_USE_LIVE_KILL_SWITCH_STATUS",
            "execution_authority": "NONE",
            "broker_writes": 0,
        },
    )


def _terminate_process_tree(process_id: int) -> bool:
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(process_id), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "status": _normalized_status(payload),
        "schema": payload.get("schema"),
    }
    if "duration_seconds" in payload:
        summary["duration_seconds"] = payload["duration_seconds"]
    for field in (
        "background_status",
        "money_loop_blocked",
        "resource_priority",
    ):
        if field in payload:
            summary[field] = payload[field]
    return summary


def _delegated_primary(background: dict[str, Any], step_name: str) -> dict[str, Any]:
    return {
        "schema": "stocks_primary_step_delegation_v1",
        "status": "BACKGROUND_DELEGATED",
        "step_name": step_name,
        "background_status": _normalized_status(background),
        "worker_pid": background.get("worker_pid"),
        "money_loop_blocked": False,
        "resource_priority": "BELOW_NORMAL",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "orders_generated": 0,
    }


def _background_refresh_due(payload: dict[str, Any], *, hours: float) -> bool:
    status = _normalized_status(payload)
    if status in {"ENQUEUED", "RUNNING"}:
        return False
    return _refresh_due(
        payload.get("completed_at") or payload.get("started_at"),
        hours=hours,
    )


def _primary_step_due(report: dict[str, Any], step_name: str, *, hours: float) -> bool:
    steps = report.get("steps")
    steps = steps if isinstance(steps, dict) else {}
    step = steps.get(step_name)
    step = step if isinstance(step, dict) else {}
    return _refresh_due(
        step.get("last_attempt_at") or report.get("completed_at"),
        hours=hours,
    )


def _normalized_status(payload: dict[str, Any]) -> str:
    status = payload.get("status", "UNKNOWN")
    if isinstance(status, dict):
        status = status.get("status", "UNKNOWN")
    return str(status).upper()


def _step_blockers(step_name: str, payload: dict[str, Any]) -> list[str]:
    status = _normalized_status(payload)
    if _primary_delegation_usable(payload):
        return []
    if step_name == "MACRO" and _macro_context_usable(payload):
        return []
    if step_name == "NEWS_DIGEST" and _news_context_usable(payload):
        return []
    if step_name == "RECONCILIATION" and _reconciliation_read_only_usable(payload):
        return []
    if step_name == "RESEARCH" and _research_data_blocked_expected(payload):
        return []
    if step_name == "RESEARCH" and _research_background_usable(payload):
        return []
    if step_name == "RL_SHADOW" and _rl_shadow_usable(payload):
        return []
    if step_name == "AI_DECISION_INTELLIGENCE" and _ai_shadow_usable(payload):
        return []
    if _is_success_status(payload) or status == "NOT_DUE":
        return []
    safe_status = "".join(character if character.isalnum() else "_" for character in status).strip(
        "_"
    )
    return [f"OPERATIONAL_STEP_{step_name}_{safe_status or 'UNKNOWN'}"]


def _is_success_status(payload: dict[str, Any]) -> bool:
    status = _normalized_status(payload)
    return status == "GO" or (status.endswith("_GO") and not status.endswith("NO_GO"))


def _rl_shadow_usable(payload: dict[str, Any]) -> bool:
    return (
        _normalized_status(payload) == "SHADOW_ONLY"
        and str(payload.get("rl_mode", "")).upper() == "SHADOW_ONLY"
        and payload.get("rl_live_enabled") is False
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("broker_calls", 0) or 0) == 0
        and int(payload.get("broker_writes", 0) or 0) == 0
        and int(payload.get("orders_generated", 0) or 0) == 0
    )


def _ai_shadow_usable(payload: dict[str, Any]) -> bool:
    return (
        _normalized_status(payload) in {"ENQUEUED", "COMPLETED", "SKIPPED_FRESH", "SKIPPED_BUSY"}
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("broker_writes", 0) or 0) == 0
    )


def _research_background_usable(payload: dict[str, Any]) -> bool:
    return (
        payload.get("schema") == "stocks_background_job_v1"
        and _normalized_status(payload)
        in {
            "ENQUEUED",
            "RUNNING",
            "COMPLETED",
            "SKIPPED_BUSY",
            "RETRY_BACKOFF",
        }
        and payload.get("money_loop_blocked") is False
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("broker_writes", 0) or 0) == 0
        and int(payload.get("orders_generated", 0) or 0) == 0
    )


def _primary_delegation_usable(payload: dict[str, Any]) -> bool:
    return (
        _normalized_status(payload) == "BACKGROUND_DELEGATED"
        and payload.get("money_loop_blocked") is False
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("broker_writes", 0) or 0) == 0
        and int(payload.get("orders_generated", 0) or 0) == 0
    )


def _reconciliation_read_only_usable(payload: dict[str, Any]) -> bool:
    if payload.get("schema") != "phase9_reconciliation_audit_v1":
        return False
    read_counters = payload.get("read_only_request_counters")
    write_counters = payload.get("broker_write_counters")
    if not isinstance(read_counters, dict) or not isinstance(write_counters, dict):
        return False
    required_reads = {
        "read_only_account_summary_requests",
        "read_only_all_api_open_order_requests",
        "read_only_execution_requests",
        "read_only_position_requests",
        "read_only_same_client_open_order_requests",
    }
    return (
        str(payload.get("broker_observation_status", "")).upper() == "GO"
        and str(payload.get("broker_snapshot_status", "")).upper() == "BROKER_SNAPSHOT_OBSERVED"
        and str(payload.get("operational_broker_state_status", "")).upper()
        == "CURRENT_BROKER_FLAT_READ_ONLY"
        and int(payload.get("broker_position_count", 0) or 0) == 0
        and int(payload.get("broker_open_order_count", 0) or 0) == 0
        and all(int(read_counters.get(name, 0) or 0) >= 1 for name in required_reads)
        and all(int(value or 0) == 0 for value in write_counters.values())
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("paper_place_order_calls", 0) or 0) == 0
        and int(payload.get("live_place_order_calls", 0) or 0) == 0
        and payload.get("automatic_submission") is False
    )


def _downstream_refresh_due(
    upstream: dict[str, Any],
    last_refresh: Any,
    *,
    hours: int,
) -> bool:
    return _is_success_status(upstream) or _refresh_due(
        last_refresh,
        hours=hours,
    )


def _macro_context_usable(payload: dict[str, Any]) -> bool:
    return (
        payload.get("schema") == "macro_update_v2"
        and _normalized_status(payload) in {"GO", "DATA_INCOMPLETE", "PARTIAL"}
        and str(payload.get("collection_status", "")).upper() == "GO"
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("broker_calls", 0)) == 0
        and int(payload.get("order_calls", 0)) == 0
    )


def _news_context_usable(payload: dict[str, Any]) -> bool:
    if _is_success_status(payload):
        return (
            str(payload.get("execution_authority", "NONE")).upper() == "NONE"
            and int(payload.get("broker_calls", 0)) == 0
            and int(payload.get("order_calls", 0)) == 0
            and int(payload.get("orders_generated", 0)) == 0
            and not bool(payload.get("automatic_execution", False))
        )
    digest = payload.get("digest")
    if not isinstance(digest, dict):
        return False
    sources = digest.get("news_source_status")
    return (
        payload.get("schema") == "telegram_market_digest_preview_v1"
        and _normalized_status(payload) in {"GO", "PARTIAL"}
        and isinstance(sources, dict)
        and str(sources.get("status", "")).upper() == "GO"
        and str(digest.get("news_freshness_status", "")).startswith("CURRENT_")
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("broker_calls", 0)) == 0
        and int(payload.get("orders_generated", 0)) == 0
        and int(digest.get("order_calls", 0)) == 0
        and not bool(digest.get("automatic_execution", False))
    )


def _research_data_blocked_expected(payload: dict[str, Any]) -> bool:
    campaign = payload.get("campaign")
    recovery = payload.get("survivor_recovery")
    if not isinstance(campaign, dict) or not isinstance(recovery, dict):
        return False
    eligibility = campaign.get("eligibility")
    return (
        payload.get("schema") == "bounded_research_autopilot_cycle_v1"
        and _normalized_status(payload) == "DATA_BLOCKED"
        and isinstance(eligibility, dict)
        and eligibility.get("status") == "PIT_ELIGIBILITY_UNAVAILABLE"
        and int(campaign.get("complete_trial_count", 0)) == 0
        and str(recovery.get("status", "")).upper() == "GO"
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("broker_calls", 0)) == 0
        and int(payload.get("orders_generated", 0)) == 0
    )


def _signal_lifecycle(project_root: Path, dynamic: dict[str, Any]) -> dict[str, Any]:
    path = _private_root(project_root) / "signal-states.json"
    previous = _read_json(path).get("states", {})
    current: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    signals = dynamic.get("signals", {})
    for signal in signals.get("signals", []) if isinstance(signals, dict) else []:
        symbol = str(signal.get("ticker", "")).upper()
        strategy = str(signal.get("strategy_id", ""))
        if not symbol or not strategy:
            continue
        key = f"{strategy}:{symbol}"
        action = str(signal.get("action", "NO_SIGNAL")).upper()
        prior = str(previous.get(key, "NO_SIGNAL")).upper()
        if action in {"BUY", "STRONG_BUY"}:
            lifecycle = (
                "FRESH_ENTRY" if prior not in {"BUY", "STRONG_BUY"} else "ACTIVE_STATE_NO_NEW_ENTRY"
            )
        elif action in {"SELL", "EXIT"}:
            lifecycle = "EXIT" if prior in {"BUY", "STRONG_BUY"} else "NO_SIGNAL"
        elif action == "REDUCE":
            lifecycle = "REDUCE"
        elif action in {"WATCHLIST", "PENDING"}:
            lifecycle = "BREAKOUT_TRIGGER_PENDING"
        elif action in {"HOLD", "POSITION_HOLD"}:
            lifecycle = "POSITION_HOLD"
        elif action == "AVOID":
            lifecycle = "AVOID"
        else:
            lifecycle = "NO_SIGNAL"
        current[key] = action
        rows.append(
            {
                "strategy_id": strategy,
                "ticker": symbol,
                "previous_action": prior,
                "current_action": action,
                "lifecycle_status": lifecycle,
                "new_entry_eligible": lifecycle == "FRESH_ENTRY",
            }
        )
    _atomic_json(path, {"updated_at": _now(), "states": current})
    report = {
        "schema": "signal_lifecycle_transition_v1",
        "status": "GO",
        "generated_at": _now(),
        "fresh_entry_count": sum(row["lifecycle_status"] == "FRESH_ENTRY" for row in rows),
        "exit_count": sum(row["lifecycle_status"] == "EXIT" for row in rows),
        "rows": rows,
        "execution_authority": "NONE",
    }
    return _publish(project_root, "signal-lifecycle.json", report)


def _refresh_due(value: Any, *, hours: int = 6) -> bool:
    if not value:
        return True
    try:
        previous = datetime.fromisoformat(str(value))
    except ValueError:
        return True
    if previous.tzinfo is None:
        previous = previous.replace(tzinfo=UTC)
    return (datetime.now(UTC) - previous).total_seconds() >= hours * 3600


def _reconciliation_arguments(project_root: Path, mode: str) -> tuple[str, ...]:
    if mode in {"LIVE_CANARY_AUTOMATIC", "CONTROLLED_LIVE"}:
        return ("live", "reconcile")
    if mode == "SIGNALS_ONLY":
        live = _read_json(project_root / "output" / "ibkr" / "live" / "status.json")
        if str(live.get("account_reconciliation", "")).startswith("LIVE_RECONCILED"):
            return ("live", "reconcile")
    return ("ibkr", "phase9", "reconcile")


def _research_due(project_root: Path) -> bool:
    state = _read_json(
        project_root / "data" / "research" / "autopilot" / "private" / "runtime-state.json"
    )
    if not state.get("enabled", False) or state.get("paused", False):
        return False
    next_cycle = state.get("next_cycle_at")
    if not next_cycle:
        return True
    try:
        due = datetime.fromisoformat(str(next_cycle))
    except ValueError:
        return True
    if due.tzinfo is None:
        due = due.replace(tzinfo=UTC)
    return due <= datetime.now(UTC)


def _count_values(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "UNKNOWN"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _public_manual_position_result(
    position: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    ownership = str(position.get("ownership_status", "UNKNOWN"))
    broker_match = str(position.get("broker_match_status", "UNVERIFIED"))
    auto_eligible = bool(position.get("automatic_execution_eligible", False))
    return {
        "schema": "manual_tracked_position_transition_v1",
        "status": status,
        "position_id": position.get("position_id"),
        "signal_id": position.get("signal_id"),
        "ticker": position.get("ticker"),
        "ownership_status": ownership,
        "lifecycle_status": position.get("lifecycle_status", "OPEN"),
        "transition_status": position.get(
            "transition_status",
            position.get("registration_status"),
        ),
        "broker_match_status": broker_match,
        "management_status": (
            "BOT_MANAGED_PENDING_BROKER_MATCH"
            if ownership == "BOT_MANAGED" and broker_match != "MATCHED"
            else (
                "BOT_MANAGED_BROKER_MATCHED_AUTHORITY_REQUIRED"
                if ownership == "BOT_MANAGED"
                else ownership
            )
        ),
        "automatic_execution_eligible": auto_eligible,
        "automatic_position_adoption": False,
        "financial_values_public": False,
        "execution_authority": "NONE",
        "broker_write_calls": 0,
        "orders_generated": 0,
    }


def _manual_contract_identity(
    project_root: Path,
    signal: dict[str, Any],
) -> dict[str, Any]:
    ticker = str(signal.get("ticker", "")).strip().upper()
    if not ticker:
        return {"status": "MISSING_SIGNAL_TICKER"}
    identity = signal.get("contract_identity")
    requested_con_id: int | None = None
    if isinstance(identity, dict):
        raw_con_id = identity.get("con_id", identity.get("conId"))
        if raw_con_id not in {None, ""}:
            try:
                requested_con_id = int(str(raw_con_id))
            except (TypeError, ValueError):
                return {"status": "INVALID_SIGNAL_CONTRACT_IDENTITY"}
            if requested_con_id <= 0:
                return {"status": "INVALID_SIGNAL_CONTRACT_IDENTITY"}
    try:
        rows = read_contract_cache_rows(ContractCacheLayout.from_project_root(project_root))
    except (OSError, ValueError):
        return {"status": "CONTRACT_CACHE_INVALID"}
    matches = [
        row
        for row in rows
        if row.contract.security_type.value == "STK"
        and row.contract.symbol.upper() == ticker
        and (requested_con_id is None or row.contract.con_id == requested_con_id)
    ]
    if not matches:
        return {
            "status": (
                "CONTRACT_IDENTITY_NOT_IN_CACHE"
                if requested_con_id is not None
                else "MISSING_CONTRACT_IDENTITY"
            )
        }
    if len(matches) != 1:
        return {"status": "AMBIGUOUS_CONTRACT_IDENTITY"}
    row = matches[0]
    return {
        "status": "RESOLVED",
        "con_id": row.contract.con_id,
        "contract_hash": row.contract_hash,
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "execution_authority": "NONE",
        "orders_generated": 0,
    }


def _private_root(project_root: Path) -> Path:
    return project_root / "data" / "operations" / "private"


def _now() -> str:
    return datetime.now(UTC).isoformat()
