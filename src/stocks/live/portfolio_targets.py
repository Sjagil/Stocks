from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.capital.service import capital_level_limits
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.p0_readiness import inspect_p0_execution_readiness_gate
from stocks.live.authority import authority_status
from stocks.live.approvals import approval_challenge, approve
from stocks.live.config import load_live_portfolio_config
from stocks.live.store import LiveExecutionStore
from stocks.universe import broad_asset_metadata


PRIVATE_TARGETS = Path(
    "data/portfolio/private/desired-portfolio-targets.json"
)
PRIVATE_PORTFOLIO_STATE = Path("data/portfolio/private/current-state.json")
PRIVATE_PURCHASE_PLAN = Path(
    "data/execution/live/private/controlled-purchase-plan.json"
)
PUBLIC_PURCHASE_PLAN = Path(
    "output/ibkr/live/controlled-purchase-plan.json"
)
CAPITAL_STATUS = Path("output/capital/current_level.json")
PORTFOLIO_PLAN = Path("output/portfolio/active_portfolio_plan.json")
CONTRACTS = Path("output/ibkr/contracts/stocks.parquet")
CONTROLLED_LEVEL = 2
CONTROLLED_AUTHORITY = "LIVE_LEVEL_TWO"


def publish_controlled_purchase_plan(project_root: Path) -> dict[str, Any]:
    """Bind desired whole-share deltas to controlled-live prerequisites.

    This function is deliberately advisory: it creates no broker order, intent,
    approval, capability, or authority state.
    """
    targets = _read_json(project_root / PRIVATE_TARGETS)
    portfolio_state = _read_json(project_root / PRIVATE_PORTFOLIO_STATE)
    capital = _read_json(project_root / CAPITAL_STATUS)
    portfolio = _read_json(project_root / PORTFOLIO_PLAN)
    account = portfolio_state.get("account_state", {})
    equity = _decimal(account.get("net_liquidation_eur"))
    if equity <= 0:
        equity = Decimal("0")
    try:
        limits = (
            capital_level_limits(
                project_root,
                level=CONTROLLED_LEVEL,
                account_equity_eur=equity,
            )
            if equity > 0
            else {}
        )
    except (KeyError, TypeError, ValueError):
        limits = {}
    current_level = int(capital.get("CURRENT_CAPITAL_LEVEL", 0) or 0)
    authority = authority_status(project_root)
    execution_authority = str(
        authority.get("execution_authority") or "NONE"
    )
    p0_gate = inspect_p0_execution_readiness_gate(project_root)
    opportunities = {
        str(row.get("ticker") or "").upper(): row
        for row in portfolio.get("opportunities", {}).get(
            "opportunities", []
        )
    }
    sizing = {
        str(row.get("ticker") or "").upper(): row
        for row in portfolio_state.get("whole_share_sizing", {}).get(
            "positions", []
        )
    }
    contracts = _contract_map(
        project_root / CONTRACTS,
        asset_metadata=broad_asset_metadata(project_root),
    )
    rows: list[dict[str, Any]] = []
    for target in targets.get("targets", []):
        if str(target.get("action")) != "BUY_DELTA":
            continue
        symbol = str(target.get("symbol") or "").upper()
        if not symbol:
            continue
        rows.append(
            evaluate_controlled_purchase_target(
                target=target,
                sizing=sizing.get(symbol, {}),
                opportunity=opportunities.get(symbol, {}),
                contract=contracts.get(symbol, {}),
                limits=limits,
                current_capital_level=current_level,
                execution_authority=execution_authority,
                p0_gate_go=p0_gate.get("status") == "GO",
            )
        )
    private = {
        "schema": "controlled_purchase_plan_private_v1",
        "status": "GO" if rows else "NO_ACTION",
        "generated_at": datetime.now(UTC).isoformat(),
        "required_capital_level": CONTROLLED_LEVEL,
        "required_execution_authority": CONTROLLED_AUTHORITY,
        "current_capital_level": current_level,
        "current_execution_authority": execution_authority,
        "p0_execution_infrastructure_ready": p0_gate.get("status") == "GO",
        "resolved_level_limits": limits,
        "purchase_target_count": len(rows),
        "technically_executable_target_count": sum(
            bool(row["target_technically_executable"]) for row in rows
        ),
        "live_ready_target_count": sum(bool(row["live_ready"]) for row in rows),
        "targets": rows,
        "submits_orders": False,
        "creates_order_intents": False,
        "changes_authority": False,
        "orders_generated": 0,
        "broker_calls": 0,
        "live_place_order_calls": 0,
    }
    private["content_hash"] = stable_hash(private)
    public_rows = [
        {
            "symbol": row["symbol"],
            "action": row["action"],
            "asset_class": row["asset_class"],
            "contract_resolved": row["contract_resolved"],
            "contract_identity_hash": row["contract_identity_hash"],
            "whole_share": row["whole_share"],
            "target_technically_executable": row[
                "target_technically_executable"
            ],
            "live_ready": row["live_ready"],
            "blockers": row["blockers"],
            "quantities_public": False,
            "financial_values_public": False,
        }
        for row in rows
    ]
    public = {
        **{
            key: value
            for key, value in private.items()
            if key
            not in {
                "targets",
                "content_hash",
                "resolved_level_limits",
            }
        },
        "schema": "controlled_purchase_plan_public_v1",
        "targets": public_rows,
        "private_plan_reference": PRIVATE_PURCHASE_PLAN.as_posix(),
    }
    public["content_hash"] = stable_hash(public)
    _write_json(project_root / PRIVATE_PURCHASE_PLAN, private)
    _write_json(project_root / PUBLIC_PURCHASE_PLAN, public)
    return public


def evaluate_controlled_purchase_target(
    *,
    target: dict[str, Any],
    sizing: dict[str, Any],
    opportunity: dict[str, Any],
    contract: dict[str, Any],
    limits: dict[str, Any],
    current_capital_level: int,
    execution_authority: str,
    p0_gate_go: bool,
) -> dict[str, Any]:
    symbol = str(target.get("symbol") or "").upper()
    asset_class = str(target.get("asset_class") or "UNKNOWN").upper()
    quantity = _decimal(target.get("quantity_delta"))
    notional = _decimal(sizing.get("planned_notional_eur"))
    risk = _decimal(sizing.get("actual_risk_eur"))
    remaining_cash = _decimal(sizing.get("remaining_cash_eur"))
    entry = _decimal(sizing.get("reference_price"))
    stop = _decimal(sizing.get("stop_price"))
    take_profit = _decimal(sizing.get("take_profit_price"))
    portfolio_heat = _decimal(risk)
    pooled_vehicle_classes = {
        "ETF",
        "COMMODITY_EXPOSURE",
        "COMMODITY_EQUITY_ETF",
        "COMMODITY_CLOSED_END_TRUST",
    }
    order_cap_key = (
        "maximum_etf_order_eur"
        if asset_class in pooled_vehicle_classes
        else "maximum_stock_order_eur"
    )
    order_cap = _decimal(limits.get(order_cap_key))
    risk_cap = _decimal(limits.get("maximum_risk_per_trade_eur"))
    heat_cap = _decimal(limits.get("maximum_portfolio_heat_eur"))
    exposure_cap = _decimal(limits.get("maximum_total_exposure_eur"))
    whole_share = quantity > 0 and quantity == quantity.to_integral_value()
    broker_symbol = str(
        contract.get("broker_symbol")
        or contract.get("symbol")
        or ""
    ).upper()
    portfolio_symbol = str(
        contract.get("portfolio_symbol") or symbol
    ).upper()
    contract_resolved = bool(
        int(contract.get("con_id", 0) or 0) > 0
        and str(contract.get("security_type") or "") == "STK"
        and str(contract.get("symbol") or "").upper() == broker_symbol
        and portfolio_symbol == symbol
        and contract.get("contract_hash")
    )
    technical_blockers: list[str] = []
    if not whole_share:
        technical_blockers.append("POSITIVE_WHOLE_SHARE_DELTA_REQUIRED")
    if not contract_resolved:
        technical_blockers.append("EXACT_RESOLVED_STK_CONTRACT_REQUIRED")
    if not sizing:
        technical_blockers.append("WHOLE_SHARE_SIZING_REQUIRED")
    if notional <= 0 or order_cap <= 0 or notional > order_cap:
        technical_blockers.append("CONTROLLED_ORDER_NOTIONAL_EXCEEDED")
    if risk <= 0 or risk_cap <= 0 or risk > risk_cap:
        technical_blockers.append("CONTROLLED_TRADE_RISK_EXCEEDED")
    if heat_cap <= 0 or portfolio_heat > heat_cap:
        technical_blockers.append("CONTROLLED_PORTFOLIO_HEAT_EXCEEDED")
    if exposure_cap <= 0 or notional > exposure_cap:
        technical_blockers.append("CONTROLLED_TOTAL_EXPOSURE_EXCEEDED")
    if remaining_cash < 0:
        technical_blockers.append("NEGATIVE_REMAINING_CASH")
    if str(sizing.get("execution_candidate_status")) != (
        "EXECUTABLE_WHOLE_SHARE"
    ):
        technical_blockers.append("WHOLE_SHARE_TARGET_NOT_EXECUTABLE")
    if not (Decimal("0") < stop < entry < take_profit):
        technical_blockers.append("VALID_LONG_BRACKET_REQUIRED")

    blockers = list(technical_blockers)
    if not p0_gate_go:
        blockers.append("P0_EXECUTION_INFRASTRUCTURE_READY_REQUIRED")
    if current_capital_level < CONTROLLED_LEVEL:
        blockers.append("CAPITAL_LEVEL_2_REQUIRED")
    if execution_authority != CONTROLLED_AUTHORITY:
        blockers.append("LIVE_LEVEL_TWO_AUTHORITY_REQUIRED")
    blockers.extend(
        str(value)
        for value in opportunity.get("deployment_blockers", [])
        if str(value) != "EXECUTION_AUTHORITY_NONE"
    )
    technically_executable = not technical_blockers
    live_ready = technically_executable and not blockers
    con_id = int(contract.get("con_id", 0) or 0)
    contract_identity = {
        "con_id": con_id,
        "symbol": broker_symbol,
        "portfolio_symbol": symbol,
        "currency": contract.get("currency"),
        "exchange": contract.get("exchange"),
        "contract_hash": contract.get("contract_hash"),
    }
    return {
        "schema": "controlled_purchase_target_private_v1",
        "symbol": symbol,
        "asset_class": asset_class,
        "action": "BUY",
        "quantity": str(quantity),
        "whole_share": whole_share,
        "estimated_notional_eur": str(notional),
        "maximum_planned_loss_eur": str(risk),
        "remaining_cash_eur": str(remaining_cash),
        "entry_reference": str(entry),
        "stop_price": str(stop),
        "take_profit_price": str(take_profit),
        "currency": sizing.get("currency"),
        "fx_rate_to_eur": sizing.get("fx_to_eur"),
        "strategy_source": target.get("strategy_source"),
        "contract_resolved": contract_resolved,
        "contract": contract_identity,
        "contract_identity_hash": stable_hash(contract_identity)[:24],
        "within_order_notional_cap": notional > 0 and notional <= order_cap,
        "within_trade_risk_cap": risk > 0 and risk <= risk_cap,
        "within_portfolio_heat_cap": portfolio_heat <= heat_cap,
        "within_total_exposure_cap": notional <= exposure_cap,
        "target_technically_executable": technically_executable,
        "live_ready": live_ready,
        "technical_blockers": sorted(set(technical_blockers)),
        "blockers": sorted(set(blockers)),
        "submits_orders": False,
        "execution_authority": "NONE",
    }


def controlled_live_preflight(
    project_root: Path,
    *,
    symbol: str,
    env_file: str | Path = ".env.ibkr.portfolio.live",
    require_authority: bool = True,
) -> dict[str, Any]:
    publish_controlled_purchase_plan(project_root)
    private = _read_json(project_root / PRIVATE_PURCHASE_PLAN)
    target = next(
        (
            row
            for row in private.get("targets", [])
            if str(row.get("symbol") or "").upper() == symbol.upper()
        ),
        None,
    )
    config, config_errors = load_live_portfolio_config(project_root, env_file)
    blockers = list(config_errors)
    from stocks.live.service import live_writer_integrity_command

    integrity = live_writer_integrity_command(project_root, "verify")
    if integrity.get("status") != "GO":
        blockers.append("CONTROLLED_WRITER_FREEZE_REQUIRED")
    if target is None:
        blockers.append("CONTROLLED_PURCHASE_TARGET_REQUIRED")
    else:
        blockers.extend(
            str(value)
            for value in target.get("blockers", [])
            if require_authority
            or str(value) != "LIVE_LEVEL_TWO_AUTHORITY_REQUIRED"
        )
    report = {
        "schema": "controlled_live_preflight_v1",
        "status": "GO" if not blockers and config is not None else "NO_GO",
        "symbol": symbol.upper(),
        "target_technically_executable": bool(
            target and target.get("target_technically_executable")
        ),
        "required_execution_authority": CONTROLLED_AUTHORITY,
        "authority_required_now": require_authority,
        "safe_config": config.safe_dict() if config is not None else {},
        "writer_integrity_status": integrity.get("status", "NO_GO"),
        "blockers": sorted(set(blockers)),
        "submits_orders": False,
        "broker_calls": 0,
        "live_place_order_calls": 0,
    }
    _write_json(
        project_root / "output/ibkr/live/controlled-preflight.json",
        report,
    )
    return report


def controlled_live_prepare(
    project_root: Path,
    *,
    symbol: str,
    strategy_id: str,
    env_file: str | Path = ".env.ibkr.portfolio.live",
) -> dict[str, Any]:
    """Create an immutable multi-share intent without contacting IBKR."""
    preflight = controlled_live_preflight(
        project_root,
        symbol=symbol,
        env_file=env_file,
        require_authority=False,
    )
    if preflight["status"] != "GO" or not strategy_id.strip():
        blockers = list(preflight["blockers"])
        if not strategy_id.strip():
            blockers.append("EXACT_LIVE_STRATEGY_ID_REQUIRED")
        return _controlled_result(
            project_root,
            "prepare",
            status="NO_GO",
            blockers=blockers,
        )
    config, errors = load_live_portfolio_config(project_root, env_file)
    private = _read_json(project_root / PRIVATE_PURCHASE_PLAN)
    target = next(
        row
        for row in private.get("targets", [])
        if str(row.get("symbol") or "").upper() == symbol.upper()
    )
    if config is None or errors:
        return _controlled_result(
            project_root,
            "prepare",
            status="NO_GO",
            blockers=errors,
        )
    from stocks.live.service import _build_live_intent

    intent, risk = _build_live_intent(
        project_root,
        config,
        con_id=int(target["contract"]["con_id"]),
        quantity=_decimal(target["quantity"]),
        entry_limit_price=_decimal(target["entry_reference"]),
        stop_price=_decimal(target["stop_price"]),
        take_profit_price=_decimal(target["take_profit_price"]),
        fx_rate_to_eur=_decimal(target["fx_rate_to_eur"]),
        reason="controlled desired-portfolio whole-share delta",
        strategy_id=strategy_id.strip(),
        target_id=str(target.get("target_id") or symbol.upper()),
        asset_class=str(target.get("asset_class") or "STOCK"),
        liquidity_notional_eur=None,
    )
    if intent is None:
        return _controlled_result(
            project_root,
            "prepare",
            status="NO_GO",
            blockers=list(risk.get("blockers", [])),
        )
    store = LiveExecutionStore.from_project_root(project_root)
    store.initialize()
    registration = store.register_intent(intent.jsonable())
    status = (
        "GO"
        if registration in {"INTENT_REGISTERED", "INTENT_IDEMPOTENT"}
        else "NO_GO"
    )
    private_report = {
        "schema": "controlled_live_prepare_private_v1",
        "status": status,
        "intent_id": intent.intent_id,
        "intent_hash": stable_hash(intent.jsonable()),
        "symbol": intent.symbol,
        "quantity": str(intent.quantity),
        "approval_challenge": approval_challenge(intent),
        "register_status": registration,
        "risk": risk,
        "execution_authority": "NONE",
        "submits_orders": False,
        "broker_calls": 0,
        "live_place_order_calls": 0,
    }
    _write_json(
        project_root
        / "data/execution/live/private/controlled-prepare.json",
        private_report,
    )
    return _controlled_result(
        project_root,
        "prepare",
        status=status,
        blockers=[],
        intent_id=intent.intent_id,
        intent_hash=private_report["intent_hash"],
        register_status=registration,
        approval_challenge_private=True,
    )


def controlled_live_approve(
    project_root: Path,
    *,
    intent_id: str,
    approval_text: str,
    env_file: str | Path = ".env.ibkr.portfolio.live",
) -> dict[str, Any]:
    config, errors = load_live_portfolio_config(project_root, env_file)
    store = LiveExecutionStore.from_project_root(project_root)
    store.initialize()
    from stocks.live.service import _load_live_intent

    intent = _load_live_intent(store, intent_id)
    if config is None or errors or intent is None:
        blockers = list(errors)
        if intent is None:
            blockers.append("UNKNOWN_CONTROLLED_LIVE_INTENT")
        return _controlled_result(
            project_root,
            "approval",
            status="NO_GO",
            blockers=blockers,
        )
    result = approve(
        store,
        intent,
        approval_text,
        ttl_seconds=config.approval_ttl_seconds,
    )
    return _controlled_result(
        project_root,
        "approval",
        status=str(result["status"]),
        blockers=(
            []
            if result["status"] == "GO"
            else [str(result["approval_status"])]
        ),
        approval_status=result["approval_status"],
    )


def controlled_live_submit(
    project_root: Path,
    *,
    intent_id: str,
    activation_approval: str,
    env_file: str | Path = ".env.ibkr.portfolio.live",
) -> dict[str, Any]:
    """Submit only after every controlled-live gate and approval is current."""
    config, errors = load_live_portfolio_config(project_root, env_file)
    if config is None or errors:
        return _controlled_result(
            project_root, "submission", status="NO_GO", blockers=errors
        )
    if activation_approval != config.manual_activation_phrase:
        return _controlled_result(
            project_root,
            "submission",
            status="NO_GO",
            blockers=["EXACT_OPERATOR_APPROVAL_REQUIRED"],
        )
    store = LiveExecutionStore.from_project_root(project_root)
    store.initialize()
    from stocks.live.service import (
        _live_intent_binding,
        _load_live_intent,
        _submit_live_intent_core,
    )

    intent = _load_live_intent(store, intent_id)
    if intent is None:
        return _controlled_result(
            project_root,
            "submission",
            status="NO_GO",
            blockers=["UNKNOWN_CONTROLLED_LIVE_INTENT"],
        )
    preflight = controlled_live_preflight(
        project_root,
        symbol=intent.symbol,
        env_file=env_file,
        require_authority=True,
    )
    binding = _live_intent_binding(project_root, intent)
    blockers = [*preflight["blockers"], *binding["blockers"]]
    if blockers:
        return _controlled_result(
            project_root,
            "submission",
            status="NO_GO",
            blockers=blockers,
        )
    approval_record = store.find_unconsumed_approval(
        intent.intent_id, "LIVE_SUBMIT"
    )
    if approval_record is None:
        return _controlled_result(
            project_root,
            "submission",
            status="NO_GO",
            blockers=["APPROVAL_REQUIRED"],
        )
    return _submit_live_intent_core(
        project_root,
        config=config,
        intent=intent,
        store=store,
        consume_manual_approval=True,
        authority_required=True,
        required_authority=CONTROLLED_AUTHORITY,
    )


def _controlled_result(
    project_root: Path,
    name: str,
    *,
    status: str,
    blockers: list[str],
    **extra: Any,
) -> dict[str, Any]:
    report = {
        "schema": f"controlled_live_{name}_v1",
        "status": status,
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "submits_orders": False,
        "broker_calls": 0,
        "live_place_order_calls": 0,
        **extra,
    }
    _write_json(
        project_root / f"output/ibkr/live/controlled-{name}.json",
        report,
    )
    return report


def _contract_map(
    path: Path,
    *,
    asset_metadata: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return {}
    required = {"symbol", "security_type", "con_id", "contract_hash"}
    if not required.issubset(frame.columns):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for symbol, group in frame.groupby(
        frame["symbol"].astype(str).str.upper(), sort=True
    ):
        stocks = group.loc[group["security_type"].astype(str).eq("STK")]
        if len(stocks) == 1:
            result[str(symbol)] = stocks.iloc[0].to_dict()
    for portfolio_symbol, metadata in (asset_metadata or {}).items():
        if not isinstance(metadata, dict):
            continue
        broker_symbol = str(metadata.get("broker_symbol") or "").upper()
        portfolio_symbol = str(portfolio_symbol).upper()
        if broker_symbol in result:
            result[portfolio_symbol] = {
                **result[broker_symbol],
                "broker_symbol": broker_symbol,
                "portfolio_symbol": portfolio_symbol,
            }
    return result


def _decimal(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() else Decimal("0")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "controlled_live_approve",
    "controlled_live_preflight",
    "controlled_live_prepare",
    "controlled_live_submit",
    "evaluate_controlled_purchase_target",
    "publish_controlled_purchase_plan",
]
