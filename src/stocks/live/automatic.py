from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from stocks.data.phase5_common import sha256_file
from stocks.live.authority import (
    AUTONOMOUS_LEVEL_ONE,
    LIVE_LEVEL_ONE,
    authority_status,
)
from stocks.live.autonomous_policy import autonomous_final_policy_check
from stocks.live.config import load_live_canary_config
from stocks.live.quote import capture_live_quote, regular_session_open
from stocks.live.service import (
    live_prepare,
    live_preflight,
    live_submit_authorized,
)
AUTOMATIC_CYCLE_FREEZE_MARKER = (
    "LIVE_LEVEL_ONE_AUTOMATIC_CYCLE_FROZEN_GO"
)
AUTOMATIC_CYCLE_SOURCES = (
    "src/stocks/live/automatic.py",
    "src/stocks/live/authority.py",
    "src/stocks/live/quote.py",
    "src/stocks/live/service.py",
    "src/stocks/capital/canary.py",
    "src/stocks/research/phase11_14.py",
    "tests/test_live_automatic_cycle.py",
    "tests/test_whole_share_canary_policy.py",
    "tests/test_phase11_14.py",
)
QuoteProvider = Callable[[Any, dict[str, Any]], dict[str, Any]]
Submitter = Callable[[Path, str | Path, str], dict[str, Any]]


def automatic_cycle(
    project_root: Path,
    *,
    env_file: str | Path = ".env.ibkr.live",
    quote_provider: QuoteProvider | None = None,
    submitter: Submitter | None = None,
    now: datetime | None = None,
    preflight_report: dict[str, Any] | None = None,
    session_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    authority = authority_status(project_root)
    current_authority = str(authority.get("execution_authority") or "NONE")
    if current_authority not in {LIVE_LEVEL_ONE, AUTONOMOUS_LEVEL_ONE}:
        return _cycle_result(
            project_root,
            "AUTHORITY_NOT_GRANTED",
            ["LIVE_LEVEL_ONE_AUTHORITY_NOT_ACTIVE"],
        )
    if authority.get("automatic_order_submission") is not True:
        return _cycle_result(
            project_root,
            "AUTOMATIC_SUBMISSION_NOT_AUTHORIZED",
            ["AUTOMATIC_SUBMISSION_NOT_AUTHORIZED"],
        )
    preflight = preflight_report or live_preflight(
        project_root,
        env_file=env_file,
    )
    if preflight.get("status") != "GO":
        return _cycle_result(
            project_root,
            "PREFLIGHT_BLOCKED",
            list(preflight.get("blockers", [])),
        )
    config, errors = load_live_canary_config(project_root, env_file)
    if config is None or errors:
        return _cycle_result(
            project_root,
            "LIVE_CONFIG_BLOCKED",
            errors,
        )
    current_time = now or datetime.now(UTC)
    allowlist = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "strategy-allowlist.json"
    )
    observation = _merge_observations(
        _read_json(
            project_root
            / "output"
            / "research"
            / "phase11_13"
            / "latest-forward-observation.json"
        ),
        _read_json(
            project_root
            / "output"
            / "research"
            / "phase11_10"
            / "latest-pit-forward-observation.json"
        ),
        _read_json(
            project_root
            / "output"
            / "research"
            / "phase11_14"
            / "latest-forward-observation.json"
        ),
    )
    survivor_observation = _read_json(
        project_root
        / "output"
        / "research"
        / "phase11_14"
        / "latest-forward-observation.json"
    )
    signals = [
        *_read_list(
            project_root / "output" / "signals" / "active_signals.json"
        ),
        *_read_list(
            project_root / "output" / "signals" / "pit_mtf_signals.json"
        ),
        *_observation_signals(survivor_observation),
    ]
    candidate, selection_blockers = select_candidate(
        allowlist,
        observation,
        signals,
    )
    if candidate is None:
        return _cycle_result(
            project_root,
            "NO_ACTIONABLE_ALLOWLISTED_SIGNAL",
            selection_blockers,
            status="GO",
        )
    daily_target = _daily_profit_target_gate(
        project_root,
        current_time,
    )
    if daily_target["status"] != "GO":
        return _cycle_result(
            project_root,
            "LIVE_DAILY_PROFIT_POLICY_NOT_PROVEN",
            list(daily_target["blockers"]),
        )
    if daily_target["target_reached"]:
        return _cycle_result(
            project_root,
            "DAILY_PROFIT_TARGET_REACHED",
            [],
            status="GO",
            daily_profit_target_reached=True,
        )
    contract = _contract(project_root, candidate["symbol"])
    if not contract:
        return _cycle_result(
            project_root,
            "EXACT_RESOLVED_CONTRACT_REQUIRED",
            ["EXACT_RESOLVED_CONTRACT_REQUIRED"],
        )
    session = session_report or regular_session_open(
        str(contract.get("primary_exchange", "")),
        current_time,
    )
    if session.get("status") != "GO":
        return _cycle_result(
            project_root,
            str(session.get("session_status", "MARKET_CLOSED")),
            [str(session.get("session_status", "MARKET_CLOSED"))],
            status="GO",
        )
    provider = quote_provider or capture_live_quote
    quote = provider(config, contract)
    quote_blockers = list(quote.get("quote_blockers", []))
    if quote.get("status") != "GO":
        quote_blockers.append("LIVE_QUOTE_CAPTURE_BLOCKED")
    if quote.get("quote_validation_status") != "GO":
        quote_blockers.append("LIVE_QUOTE_VALIDATION_BLOCKED")
    if not _fresh_quote(quote, current_time):
        quote_blockers.append("STALE_LIVE_QUOTE")
    if quote_blockers:
        return _cycle_result(
            project_root,
            "LIVE_QUOTE_BLOCKED",
            quote_blockers,
            market_data_calls=int(quote.get("market_data_calls", 0)),
        )
    capital = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "capital-safety.json"
    )
    if (
        capital.get("status") != "GO"
        or capital.get("buying_power_sufficient") is not True
    ):
        return _cycle_result(
            project_root,
            "LIVE_BUYING_POWER_NOT_PROVEN",
            ["LIVE_BUYING_POWER_NOT_PROVEN"],
            market_data_calls=int(quote.get("market_data_calls", 0)),
        )
    entry = _round_up(
        Decimal(str(quote["ask"]))
        + Decimal(str(contract.get("minimum_tick") or "0.01")) * 2,
        Decimal(str(contract.get("minimum_tick") or "0.01")),
    )
    stop = Decimal(str(candidate["stop_price"]))
    target = Decimal(str(candidate["take_profit_price"]))
    if not stop < entry < target:
        return _cycle_result(
            project_root,
            "LIVE_BRACKET_INVALID_AT_CURRENT_QUOTE",
            ["LIVE_BRACKET_INVALID_AT_CURRENT_QUOTE"],
            market_data_calls=int(quote.get("market_data_calls", 0)),
        )
    private_state = _read_json(
        project_root / "data/portfolio/private/current-state.json"
    )
    account = private_state.get("account_state", {})
    try:
        equity = Decimal(str(account["net_liquidation_eur"]))
        target_weight = Decimal(str(candidate["target_weight"]))
        fx_rate = Decimal(str(quote["fx_rate_to_eur"]))
        share_price_eur = entry * fx_rate
    except (KeyError, TypeError, ValueError):
        equity = Decimal("0")
        target_weight = Decimal("0")
        fx_rate = Decimal("0")
        share_price_eur = Decimal("0")
    if account.get("status") != "GO" or min(
        equity, target_weight, fx_rate, share_price_eur
    ) <= 0:
        return _cycle_result(
            project_root,
            "CURRENT_RECONCILED_ACCOUNT_STATE_REQUIRED",
            ["CURRENT_RECONCILED_ACCOUNT_STATE_REQUIRED"],
            market_data_calls=int(quote.get("market_data_calls", 0)),
        )
    desired_notional = equity * target_weight
    desired_quantity = int(
        (desired_notional / share_price_eur).to_integral_value(
            rounding=ROUND_DOWN
        )
    )
    if desired_quantity == 0:
        desired_quantity = 1
    prepared = live_prepare(
        project_root,
        env_file=env_file,
        con_id=int(contract["con_id"]),
        quantity=Decimal(desired_quantity),
        entry_limit_price=entry,
        stop_price=stop,
        take_profit_price=target,
        fx_rate_to_eur=fx_rate,
        reason="frozen PIT automatic Level-1 signal",
        strategy_id=str(candidate["strategy_id"]),
        target_id=str(candidate["signal_id"]),
        asset_class=str(candidate["asset_class"]),
    )
    if prepared.get("status") != "GO":
        return _cycle_result(
            project_root,
            "LIVE_INTENT_PREPARATION_BLOCKED",
            list(prepared.get("risk_status", {}).get("blockers", [])),
            market_data_calls=int(quote.get("market_data_calls", 0)),
        )
    autonomous_decision: dict[str, Any] | None = None
    if current_authority == AUTONOMOUS_LEVEL_ONE:
        intent = _prepared_intent(project_root, str(prepared["intent_id"]))
        if intent is None:
            return _cycle_result(
                project_root,
                "AUTONOMOUS_PREPARED_INTENT_UNAVAILABLE",
                ["AUTONOMOUS_PREPARED_INTENT_UNAVAILABLE"],
                market_data_calls=int(quote.get("market_data_calls", 0)),
            )
        policy_gates = _portfolio_policy_gates(
            project_root,
            symbol=str(candidate["symbol"]),
            strategy_id=str(candidate["strategy_id"]),
        )
        autonomous_decision = autonomous_final_policy_check(
            project_root,
            intent,
            policy_gates=policy_gates,
            candidate={
                **candidate,
                "strategy_ids": [str(candidate["strategy_id"])],
            },
            source_hashes={
                "preflight": preflight.get("content_hash"),
                "quote": quote.get("content_hash"),
                "prepared_intent": prepared.get("content_hash"),
            },
        )
        if autonomous_decision.get("approved") is not True:
            return _cycle_result(
                project_root,
                "AUTONOMOUS_FINAL_POLICY_BLOCKED",
                list(autonomous_decision.get("blockers", [])),
                market_data_calls=int(quote.get("market_data_calls", 0)),
                autonomous_decision_id=autonomous_decision.get("decision_id"),
                autonomous_decision_state=autonomous_decision.get("final_state"),
            )
    execute = submitter or _production_submitter
    submission = execute(
        project_root,
        env_file,
        str(prepared["intent_id"]),
    )
    return _cycle_result(
        project_root,
        (
            "LIVE_LEVEL_ONE_ORDER_SUBMITTED"
            if submission.get("status") == "GO"
            else "LIVE_LEVEL_ONE_SUBMISSION_BLOCKED"
        ),
        list(submission.get("blockers", [])),
        status="GO" if submission.get("status") == "GO" else "NO_GO",
        market_data_calls=int(quote.get("market_data_calls", 0)),
        live_place_order_calls=int(
            submission.get("live_place_order_calls", 0)
        ),
        intent_id=str(prepared["intent_id"]),
        strategy_id=str(candidate["strategy_id"]),
        symbol=str(candidate["symbol"]),
        desired_qty=prepared.get("risk_status", {}).get("desired_qty"),
        normal_allowed_qty=prepared.get("risk_status", {}).get(
            "normal_allowed_qty"
        ),
        canary_qty=prepared.get("risk_status", {}).get("canary_qty"),
        autonomous_decision_id=(
            autonomous_decision.get("decision_id")
            if autonomous_decision
            else None
        ),
        autonomous_decision_state=(
            autonomous_decision.get("final_state")
            if autonomous_decision
            else None
        ),
    )


def select_candidate(
    allowlist: dict[str, Any],
    observation: dict[str, Any],
    signals: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if allowlist.get("status") != "GO":
        return None, ["PIT_STRATEGY_ALLOWLIST_REQUIRED"]
    allowed = {
        str(row.get("strategy_id")): {
            str(symbol).upper()
            for symbol in row.get("allowed_symbols", [])
        }
        for row in allowlist.get("strategies", [])
    }
    training_end = {
        str(row.get("strategy_id")): _timestamp(row.get("training_data_end"))
        for row in allowlist.get("strategies", [])
    }
    observed = {
        str(row.get("strategy_id")): row
        for row in observation.get("observations", [])
        if row.get("independent_forward_session")
        and (
            row.get("observation_status")
            in {"OBSERVATION_COMPLETE", "PIT_OBSERVATION_COMPLETE"}
            or (
                row.get("data_freshness") == "FRESH_CLOSED_BAR"
                and row.get("portfolio_action")
                == "OBSERVE_HYPOTHETICAL_NEXT_BAR_TARGET"
            )
        )
    }
    candidates: list[dict[str, Any]] = []
    reasons: set[str] = set()
    for signal in signals:
        strategy_id = str(signal.get("strategy_id", ""))
        symbol = str(signal.get("ticker", "")).upper()
        if strategy_id not in allowed:
            reasons.add("UNAUTHORIZED_STRATEGY_REJECTED")
            continue
        if symbol not in allowed[strategy_id]:
            reasons.add("UNAUTHORIZED_SYMBOL_REJECTED")
            continue
        strategy_observation = observed.get(strategy_id)
        if strategy_observation is None:
            reasons.add("INDEPENDENT_FORWARD_SESSION_REQUIRED")
            continue
        targets = strategy_observation.get(
            "current_attested_target_weights", {}
        )
        if symbol not in targets:
            reasons.add("SYMBOL_NOT_IN_CURRENT_FROZEN_TARGET")
            continue
        if str(signal.get("action", "")).upper() not in {
            "BUY",
            "STRONG_BUY",
        }:
            continue
        if str(signal.get("data_freshness", "")).upper() != "FRESH":
            reasons.add("STALE_SIGNAL_REJECTED")
            continue
        data_timestamp = _timestamp(signal.get("data_timestamp"))
        training_cutoff = training_end.get(strategy_id)
        if (
            data_timestamp is None
            or training_cutoff is None
            or data_timestamp <= training_cutoff
        ):
            reasons.add("SAME_OR_PRIOR_BAR_REJECTED")
            continue
        try:
            stop = Decimal(str(signal["stop_loss"]))
            target = Decimal(str(signal["take_profit_1"]))
        except Exception:
            reasons.add("COMPLETE_EXIT_STRUCTURE_REQUIRED")
            continue
        candidates.append(
            {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "stop_price": stop,
                "take_profit_price": target,
                "confidence": Decimal(
                    str(signal.get("confidence_score", "0"))
                ),
                "signal_id": str(signal.get("signal_id", "")),
                "target_weight": Decimal(str(targets[symbol])),
                "asset_class": str(
                    signal.get("asset_class") or "STOCK"
                ).upper(),
            }
        )
    if not candidates:
        return None, sorted(reasons or {"NO_CURRENT_BUY_SIGNAL"})
    candidates.sort(
        key=lambda row: (
            -row["confidence"],
            row["strategy_id"],
            row["symbol"],
        )
    )
    return candidates[0], []


def automatic_cycle_audit(project_root: Path) -> dict[str, Any]:
    allowlist = {
        "status": "GO",
        "strategies": [
            {
                "strategy_id": "STRATEGY",
                "allowed_symbols": ["TEST"],
                "training_data_end": "2026-07-20T00:00:00+00:00",
            }
        ],
    }
    observation = {
        "observations": [
            {
                "strategy_id": "STRATEGY",
                "independent_forward_session": True,
                "observation_status": "OBSERVATION_COMPLETE",
                "current_attested_target_weights": {"TEST": 1.0},
            }
        ]
    }
    signal = {
        "signal_id": "SIGNAL",
        "strategy_id": "STRATEGY",
        "ticker": "TEST",
        "action": "BUY",
        "data_freshness": "FRESH",
        "data_timestamp": "2026-07-21T00:00:00+00:00",
        "stop_loss": "9",
        "take_profit_1": "12",
        "confidence_score": "0.8",
    }
    selected, blockers = select_candidate(
        allowlist, observation, [signal]
    )
    unauthorized, unauthorized_blockers = select_candidate(
        allowlist,
        observation,
        [{**signal, "strategy_id": "UNAUTHORIZED"}],
    )
    same_bar, same_bar_blockers = select_candidate(
        allowlist,
        observation,
        [{**signal, "data_timestamp": "2026-07-20T00:00:00+00:00"}],
    )
    report = {
        "schema": "live_level_one_automatic_cycle_audit_v1",
        "status": (
            "GO"
            if selected is not None
            and not blockers
            and unauthorized is None
            and "UNAUTHORIZED_STRATEGY_REJECTED"
            in unauthorized_blockers
            and same_bar is None
            and "SAME_OR_PRIOR_BAR_REJECTED" in same_bar_blockers
            else "NO_GO"
        ),
        "exact_allowlist_enforced": unauthorized is None,
        "same_bar_execution_blocked": same_bar is None,
        "closed_bar_required": True,
        "whole_share_quantity": 1,
        "market_order_supported": False,
        "margin_enabled": False,
        "shorting_enabled": False,
        "automatic_capital_promotion": False,
        "broker_calls": 0,
        "live_place_order_calls": 0,
    }
    _write(project_root, "automatic-cycle-audit.json", report)
    return report


def automatic_cycle_freeze(project_root: Path) -> dict[str, Any]:
    audit = automatic_cycle_audit(project_root)
    hashes = {
        relative: sha256_file(project_root / relative)
        for relative in AUTOMATIC_CYCLE_SOURCES
        if (project_root / relative).exists()
    }
    report = {
        "schema": "live_level_one_automatic_cycle_freeze_v1",
        "status": "GO" if audit["status"] == "GO" else "NO_GO",
        "freeze_status": (
            AUTOMATIC_CYCLE_FREEZE_MARKER
            if audit["status"] == "GO"
            else "NO_GO"
        ),
        "source_hashes": hashes,
        "audit_status": audit["status"],
        "operational_live_order_proven": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
    }
    _write(project_root, "automatic-cycle-freeze.json", report)
    return report


def automatic_cycle_status(project_root: Path) -> dict[str, Any]:
    freeze = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "automatic-cycle-freeze.json"
    )
    expected = freeze.get("source_hashes", {})
    integrity = bool(expected) and all(
        (project_root / relative).exists()
        and sha256_file(project_root / relative) == digest
        for relative, digest in expected.items()
    )
    return {
        "schema": "live_level_one_automatic_cycle_status_v1",
        "status": "GO" if integrity else "NO_GO",
        "freeze_status": freeze.get("freeze_status", "NOT_FROZEN"),
        "hash_integrity": integrity,
        "operational_live_order_proven": False,
        "execution_authority": authority_status(project_root).get(
            "execution_authority", "NONE"
        ),
    }


def _production_submitter(
    project_root: Path,
    env_file: str | Path,
    intent_id: str,
) -> dict[str, Any]:
    return live_submit_authorized(
        project_root,
        env_file=env_file,
        intent_id=intent_id,
    )


def _prepared_intent(project_root: Path, intent_id: str) -> Any | None:
    from stocks.live.service import _load_live_intent
    from stocks.live.store import LiveExecutionStore

    store = LiveExecutionStore.from_project_root(project_root)
    store.initialize()
    return _load_live_intent(store, intent_id)


def _portfolio_policy_gates(
    project_root: Path,
    *,
    symbol: str,
    strategy_id: str,
) -> dict[str, bool]:
    cycle = _read_json(
        project_root / "data/portfolio/private/orchestrator/current-cycle.json"
    )
    proposals = cycle.get("execution_bridge", {}).get("proposals", [])
    match = next(
        (
            row
            for row in proposals
            if str(row.get("symbol") or "").upper() == symbol.upper()
            and str(row.get("strategy_id") or "") == strategy_id
        ),
        None,
    )
    if not isinstance(match, dict):
        return {}
    return {
        str(name): bool(value)
        for name, value in match.get("gates", {}).items()
    }


def _cycle_result(
    project_root: Path,
    cycle_status: str,
    blockers: list[str],
    *,
    status: str = "NO_GO",
    market_data_calls: int = 0,
    live_place_order_calls: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    report = {
        "schema": "live_level_one_automatic_cycle_v1",
        "status": status,
        "cycle_status": cycle_status,
        "blockers": sorted(set(blockers)),
        "generated_at": datetime.now(UTC).isoformat(),
        "market_data_calls": market_data_calls,
        "live_place_order_calls": live_place_order_calls,
        "broker_write_calls": live_place_order_calls,
        **extra,
    }
    _write(project_root, "automatic-cycle.json", report)
    return report


def _contract(project_root: Path, symbol: str) -> dict[str, Any]:
    path = project_root / "output" / "ibkr" / "contracts" / "stocks.parquet"
    if not path.exists():
        return {}
    import pandas as pd

    frame = pd.read_parquet(path)
    matches = frame.loc[
        frame["symbol"].astype(str).str.upper().eq(symbol.upper())
        & frame["security_type"].astype(str).eq("STK")
    ]
    if len(matches) != 1:
        return {}
    return matches.iloc[0].to_dict()


def _fresh_quote(quote: dict[str, Any], now: datetime) -> bool:
    timestamp = _timestamp(quote.get("captured_at"))
    if timestamp is None:
        return False
    current = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return abs((current - timestamp).total_seconds()) <= 30


def _daily_profit_target_gate(
    project_root: Path,
    now: datetime,
) -> dict[str, Any]:
    report = _read_json(
        project_root
        / "output"
        / "capital"
        / "daily_profit_target.json"
    )
    current = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    session_date = current.astimezone(
        ZoneInfo("Europe/Amsterdam")
    ).date().isoformat()
    blockers = []
    if report.get("status") != "GO":
        blockers.append("BROKER_DERIVED_DAILY_PROFIT_TARGET_REQUIRED")
    if (
        report.get("input_source")
        != "BROKER_RECONCILED_EQUITY_AND_CASH_CHANGE"
    ):
        blockers.append("BROKER_DERIVED_DAILY_PNL_REQUIRED")
    if report.get("session_date") != session_date:
        blockers.append("CURRENT_SESSION_DAILY_PROFIT_TARGET_REQUIRED")
    if report.get("enforcement_active") is not True:
        blockers.append("DAILY_PROFIT_TARGET_ENFORCEMENT_REQUIRED")
    if report.get("risk_chasing_allowed") is not False:
        blockers.append("DAILY_PROFIT_TARGET_RISK_CHASING_BLOCKED")
    if report.get("risk_reducing_exits_allowed") is not True:
        blockers.append("RISK_REDUCING_EXITS_MUST_REMAIN_ALLOWED")
    return {
        "status": "GO" if not blockers else "NO_GO",
        "blockers": sorted(set(blockers)),
        "target_reached": (
            bool(report.get("target_reached"))
            or report.get("new_entries_allowed") is False
        ),
    }


def _round_up(value: Decimal, tick: Decimal) -> Decimal:
    return (value / tick).to_integral_value(rounding=ROUND_UP) * tick


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00").replace(" ", "T")
        )
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _merge_observations(
    *payloads: dict[str, Any],
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    source_schemas: list[str] = []
    for payload in payloads:
        schema = payload.get("schema")
        if schema:
            source_schemas.append(str(schema))
        observations.extend(
            row
            for row in payload.get("observations", [])
            if isinstance(row, dict)
        )
    return {
        "schema": "merged_live_forward_observations_v1",
        "source_schemas": source_schemas,
        "observations": observations,
    }


def _observation_signals(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for observation in payload.get("observations", []):
        if not isinstance(observation, dict):
            continue
        if not observation.get("independent_forward_session"):
            continue
        if observation.get("data_freshness") != "FRESH_CLOSED_BAR":
            continue
        strategy_id = str(observation.get("strategy_id", ""))
        targets = {
            str(symbol).upper()
            for symbol in observation.get(
                "current_attested_target_weights",
                {},
            )
        }
        for row in observation.get("raw_active_signals", []):
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).upper()
            if symbol not in targets:
                continue
            if row.get("execution_envelope_status") != "GO":
                continue
            signals.append(
                {
                    "signal_id": row.get("signal_id"),
                    "strategy_id": strategy_id,
                    "ticker": symbol,
                    "action": row.get("action", "BUY"),
                    "data_freshness": row.get(
                        "data_freshness",
                        "FRESH",
                    ),
                    "data_timestamp": row.get(
                        "data_timestamp",
                        observation.get("closed_bar_timestamp"),
                    ),
                    "stop_loss": row.get("stop_loss"),
                    "take_profit_1": row.get("take_profit_1"),
                    "take_profit_2": row.get("take_profit_2"),
                    "confidence_score": row.get(
                        "confidence_score",
                        0,
                    ),
                    "execution_envelope_status": "GO",
                    "source_observation_schema": payload.get("schema"),
                }
            )
    return signals


def _read_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            [row for row in value if isinstance(row, dict)]
            if isinstance(value, list)
            else []
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []


def _write(project_root: Path, name: str, payload: dict[str, Any]) -> None:
    path = project_root / "output" / "ibkr" / "live" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
