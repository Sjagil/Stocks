from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

from stocks.costs import estimate_transaction_cost, load_shared_cost_model
from stocks.execution.idempotency import stable_hash
from stocks.portfolio.dynamic_risk import level_one_canary_risk_budget


SUPPORTED_ASSET_CLASSES = frozenset(
    {"STOCK", "ETF", "COMMODITY_VEHICLE"}
)


@dataclass(frozen=True)
class LevelOneCanaryPolicy:
    policy_version: str
    canary_risk_pct: Decimal
    maximum_risk_eur: Decimal
    hard_notional_cap_eur: Decimal
    maximum_total_exposure_pct: Decimal
    maximum_stock_weight: Decimal
    maximum_pooled_vehicle_weight: Decimal
    maximum_portfolio_heat_pct: Decimal
    minimum_economic_notional_eur: Decimal
    maximum_cost_to_expected_edge_ratio: Decimal
    maximum_execution_quantity: int
    whole_share_required: bool
    fractional_shares_allowed: bool
    prefer_minimum_valid_quantity: bool

    def jsonable(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(self).items()
        }


def default_level_one_canary_policy() -> LevelOneCanaryPolicy:
    """Return the conservative policy for pure/offline evaluation only."""

    return LevelOneCanaryPolicy(
        policy_version="P0.1_WHOLE_SHARE_CANARY_V2_EUR9_RISK_CAP",
        canary_risk_pct=Decimal("0.005"),
        maximum_risk_eur=Decimal("9"),
        hard_notional_cap_eur=Decimal("250"),
        maximum_total_exposure_pct=Decimal("0.20"),
        maximum_stock_weight=Decimal("0.15"),
        maximum_pooled_vehicle_weight=Decimal("0.20"),
        maximum_portfolio_heat_pct=Decimal("0.005"),
        minimum_economic_notional_eur=Decimal("5"),
        maximum_cost_to_expected_edge_ratio=Decimal("0.50"),
        maximum_execution_quantity=100,
        whole_share_required=True,
        fractional_shares_allowed=False,
        prefer_minimum_valid_quantity=True,
    )


def load_level_one_canary_policy(project_root: Path) -> LevelOneCanaryPolicy:
    path = project_root / "config/capital_scaling/levels_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    level = raw.get("levels", {}).get("1", {})
    if level.get("name") != "WHOLE_SHARE_EXECUTION_CANARY":
        raise ValueError("LEVEL_ONE_WHOLE_SHARE_POLICY_REQUIRED")
    policy = LevelOneCanaryPolicy(
        policy_version=str(level.get("policy_version") or "P0.1_V1"),
        canary_risk_pct=_positive_decimal(
            level.get("base_risk_per_trade_pct"),
            "LEVEL_ONE_RISK_PCT",
        ),
        maximum_risk_eur=_positive_decimal(
            level.get("maximum_risk_eur"),
            "LEVEL_ONE_MAXIMUM_RISK_EUR",
        ),
        hard_notional_cap_eur=_positive_decimal(
            level.get("maximum_order_eur"),
            "LEVEL_ONE_NOTIONAL_BACKSTOP",
        ),
        maximum_total_exposure_pct=_fraction(
            level.get("maximum_exposure_pct"),
            "LEVEL_ONE_EXPOSURE_PCT",
        ),
        maximum_stock_weight=_fraction(
            level.get("maximum_position_pct"),
            "LEVEL_ONE_STOCK_WEIGHT",
        ),
        maximum_pooled_vehicle_weight=_fraction(
            level.get("maximum_etf_position_pct"),
            "LEVEL_ONE_POOLED_WEIGHT",
        ),
        maximum_portfolio_heat_pct=_fraction(
            level.get("maximum_portfolio_risk_pct"),
            "LEVEL_ONE_HEAT_PCT",
        ),
        minimum_economic_notional_eur=_positive_decimal(
            level.get("minimum_economic_notional_eur"),
            "LEVEL_ONE_ECONOMIC_MINIMUM",
        ),
        maximum_cost_to_expected_edge_ratio=_fraction(
            level.get("maximum_cost_to_expected_edge_ratio"),
            "LEVEL_ONE_COST_EDGE_RATIO",
        ),
        maximum_execution_quantity=_positive_integer(
            level.get("maximum_execution_quantity"),
            "LEVEL_ONE_MAXIMUM_EXECUTION_QUANTITY",
        ),
        whole_share_required=bool(level.get("whole_share_required")),
        fractional_shares_allowed=bool(
            level.get("fractional_shares_allowed", True)
        ),
        prefer_minimum_valid_quantity=bool(
            level.get("prefer_minimum_valid_quantity")
        ),
    )
    if not policy.whole_share_required or policy.fractional_shares_allowed:
        raise ValueError("LEVEL_ONE_FRACTIONAL_POLICY_BLOCKED")
    if policy.maximum_portfolio_heat_pct < policy.canary_risk_pct:
        raise ValueError("LEVEL_ONE_HEAT_BELOW_TRADE_RISK")
    return policy


def evaluate_whole_share_canary(
    project_root: Path,
    *,
    asset_class: str,
    instrument_currency: str,
    desired_qty: Decimal,
    account_equity_eur: Decimal,
    available_cash_eur: Decimal,
    reserved_cash_eur: Decimal,
    entry_price_local: Decimal,
    protective_stop_local: Decimal,
    take_profit_local: Decimal,
    fx_rate_to_eur: Decimal,
    normal_risk_budget_eur: Decimal,
    normal_maximum_position_weight: Decimal,
    normal_maximum_portfolio_heat_pct: Decimal,
    liquidity_notional_eur: Decimal,
    existing_position_notional_eur: Decimal = Decimal("0"),
    existing_total_exposure_eur: Decimal = Decimal("0"),
    existing_portfolio_risk_eur: Decimal = Decimal("0"),
) -> dict[str, Any]:
    """Resolve economic targets into a risk-first Level-1 whole-share order.

    The function is pure with respect to broker state.  It performs no network
    or persistence operation and never rounds a fractional requested quantity.
    """

    try:
        policy = load_level_one_canary_policy(project_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        policy = default_level_one_canary_policy()
    normalized_asset_class = str(asset_class).strip().upper()
    normalized_currency = str(instrument_currency or "EUR").strip().upper()
    base = {
        "schema": "whole_share_execution_canary_sizing_v1",
        "capital_level": 1,
        "capital_level_name": "WHOLE_SHARE_EXECUTION_CANARY",
        "policy_version": policy.policy_version,
        "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
        "notional_cap_role": "SECONDARY_EMERGENCY_BACKSTOP",
        "fractional_shares_allowed": False,
        "asset_class": normalized_asset_class,
        "instrument_currency": normalized_currency,
    }
    if normalized_asset_class not in SUPPORTED_ASSET_CLASSES:
        return _blocked(base, "ASSET_CLASS_POLICY_BLOCKED")
    values = (
        desired_qty,
        account_equity_eur,
        available_cash_eur,
        reserved_cash_eur,
        entry_price_local,
        protective_stop_local,
        take_profit_local,
        fx_rate_to_eur,
        normal_risk_budget_eur,
        normal_maximum_position_weight,
        normal_maximum_portfolio_heat_pct,
        liquidity_notional_eur,
        existing_position_notional_eur,
        existing_total_exposure_eur,
        existing_portfolio_risk_eur,
    )
    if any(not value.is_finite() for value in values):
        return _blocked(base, "CANARY_INPUT_NOT_FINITE")
    if desired_qty != desired_qty.to_integral_value():
        return _blocked(base, "FRACTIONAL_QUANTITY_FORBIDDEN")
    if desired_qty < 1:
        return _blocked(base, "CANARY_NON_EXECUTABLE_WHOLE_SHARE")
    if account_equity_eur <= 0:
        return _blocked(base, "POSITIVE_ACCOUNT_EQUITY_REQUIRED")
    if entry_price_local <= 0 or fx_rate_to_eur <= 0:
        return _blocked(base, "POSITIVE_ACCOUNT_CURRENCY_PRICE_REQUIRED")
    if (
        protective_stop_local <= 0
        or protective_stop_local >= entry_price_local
    ):
        return _blocked(base, "NO_VALID_PROTECTIVE_STOP")
    if take_profit_local <= entry_price_local:
        return _blocked(base, "EXPECTED_NET_OPPORTUNITY_REQUIRED")

    desired = int(desired_qty)
    share_price_eur = entry_price_local * fx_rate_to_eur
    stop_distance_eur = (
        abs(entry_price_local - protective_stop_local) * fx_rate_to_eur
    )
    model = load_shared_cost_model(project_root)
    one_share_costs = estimate_transaction_cost(
        share_price_eur,
        currency=normalized_currency,
        model=model,
        round_trip=True,
    )
    adverse_cost_per_share = sum(
        Decimal(str(one_share_costs[key]))
        for key in (
            "exchange_fees_eur",
            "spread_eur",
            "slippage_eur",
            "market_impact_eur",
            "fx_conversion_eur",
        )
    )
    risk_per_share_eur = stop_distance_eur + adverse_cost_per_share
    if risk_per_share_eur <= 0:
        return _blocked(base, "NO_VALID_PROTECTIVE_STOP")

    canary_risk_budget_eur = level_one_canary_risk_budget(
        account_equity_eur=account_equity_eur,
        normal_risk_budget_eur=normal_risk_budget_eur,
        configured_canary_risk_pct=policy.canary_risk_pct,
        configured_absolute_cap_eur=policy.maximum_risk_eur,
    )
    spendable_cash = max(
        Decimal("0"), available_cash_eur - reserved_cash_eur
    )
    pooled = normalized_asset_class in {"ETF", "COMMODITY_VEHICLE"}
    canary_position_weight = (
        policy.maximum_pooled_vehicle_weight
        if pooled
        else policy.maximum_stock_weight
    )
    normal_position_room = max(
        Decimal("0"),
        account_equity_eur * normal_maximum_position_weight
        - existing_position_notional_eur,
    )
    canary_position_room = max(
        Decimal("0"),
        account_equity_eur * canary_position_weight
        - existing_position_notional_eur,
    )
    total_exposure_room = max(
        Decimal("0"),
        account_equity_eur * policy.maximum_total_exposure_pct
        - existing_total_exposure_eur,
    )
    normal_heat_room = max(
        Decimal("0"),
        account_equity_eur * normal_maximum_portfolio_heat_pct
        - existing_portfolio_risk_eur,
    )
    canary_heat_room = max(
        Decimal("0"),
        account_equity_eur * policy.maximum_portfolio_heat_pct
        - existing_portfolio_risk_eur,
    )

    limits = {
        "qty_by_strategy_target": desired,
        "qty_by_normal_risk": _whole(normal_risk_budget_eur / risk_per_share_eur),
        "qty_by_canary_risk": _whole(canary_risk_budget_eur / risk_per_share_eur),
        "qty_by_cash": _whole(spendable_cash / share_price_eur),
        "qty_by_normal_concentration": _whole(
            normal_position_room / share_price_eur
        ),
        "qty_by_canary_concentration": _whole(
            canary_position_room / share_price_eur
        ),
        "qty_by_normal_portfolio_heat": _whole(
            normal_heat_room / risk_per_share_eur
        ),
        "qty_by_canary_portfolio_heat": _whole(
            canary_heat_room / risk_per_share_eur
        ),
        "qty_by_total_exposure": _whole(
            total_exposure_room / share_price_eur
        ),
        "qty_by_liquidity": _whole(
            max(Decimal("0"), liquidity_notional_eur) / share_price_eur
        ),
        "qty_by_notional_backstop": _whole(
            policy.hard_notional_cap_eur / share_price_eur
        ),
        "qty_by_asset_policy": policy.maximum_execution_quantity,
    }
    normal_allowed = min(
        limits[key]
        for key in (
            "qty_by_strategy_target",
            "qty_by_normal_risk",
            "qty_by_cash",
            "qty_by_normal_concentration",
            "qty_by_normal_portfolio_heat",
            "qty_by_liquidity",
        )
    )
    maximum_canary = min(
        normal_allowed,
        limits["qty_by_canary_risk"],
        limits["qty_by_cash"],
        limits["qty_by_canary_concentration"],
        limits["qty_by_canary_portfolio_heat"],
        limits["qty_by_total_exposure"],
        limits["qty_by_liquidity"],
        limits["qty_by_notional_backstop"],
        limits["qty_by_asset_policy"],
    )
    reason = _zero_quantity_reason(limits, normal_allowed, maximum_canary)
    if maximum_canary < 1:
        return {
            **base,
            "status": "NO_GO",
            "sizing_reason": reason,
            "blocking_reason": reason,
            "desired_qty": desired,
            "normal_allowed_qty": normal_allowed,
            "canary_qty": 0,
            "share_price_eur": str(share_price_eur),
            "risk_per_share_eur": str(risk_per_share_eur),
            "normal_risk_budget_eur": str(normal_risk_budget_eur),
            "canary_risk_budget_eur": str(canary_risk_budget_eur),
            "hard_notional_backstop_eur": str(
                policy.hard_notional_cap_eur
            ),
            "quantity_limits": limits,
        }

    selected = 0
    selected_costs: dict[str, Decimal | bool | str] = {}
    expected_gross_upside_eur = Decimal("0")
    expected_net_opportunity_eur = Decimal("0")
    for quantity in range(1, maximum_canary + 1):
        notional = share_price_eur * quantity
        costs = estimate_transaction_cost(
            notional,
            currency=normalized_currency,
            model=model,
            round_trip=True,
        )
        gross = (
            (take_profit_local - entry_price_local)
            * fx_rate_to_eur
            * quantity
        )
        total_cost = Decimal(str(costs["total_cost_eur"]))
        net = gross - total_cost
        cost_ratio = total_cost / gross if gross > 0 else Decimal("Infinity")
        if (
            notional >= policy.minimum_economic_notional_eur
            and bool(costs["economic_minimum_met"])
            and net > 0
            and cost_ratio <= policy.maximum_cost_to_expected_edge_ratio
            and notional + total_cost <= spendable_cash
        ):
            selected = quantity
            selected_costs = costs
            expected_gross_upside_eur = gross
            expected_net_opportunity_eur = net
            break
    if selected == 0:
        return {
            **base,
            "status": "NO_GO",
            "sizing_reason": "ECONOMICALLY_TOO_SMALL",
            "blocking_reason": "ECONOMICALLY_TOO_SMALL",
            "desired_qty": desired,
            "normal_allowed_qty": normal_allowed,
            "canary_qty": 0,
            "share_price_eur": str(share_price_eur),
            "risk_per_share_eur": str(risk_per_share_eur),
            "normal_risk_budget_eur": str(normal_risk_budget_eur),
            "canary_risk_budget_eur": str(canary_risk_budget_eur),
            "hard_notional_backstop_eur": str(
                policy.hard_notional_cap_eur
            ),
            "quantity_limits": limits,
        }

    actual_notional = share_price_eur * selected
    estimated_costs = Decimal(str(selected_costs["total_cost_eur"]))
    planned_risk = risk_per_share_eur * selected
    sizing_reason = (
        "CANARY_DOWNSCALED_TO_ONE_SHARE"
        if selected == 1 and normal_allowed > 1
        else "CANARY_EXECUTABLE"
    )
    return {
        **base,
        "status": "GO",
        "sizing_reason": sizing_reason,
        "blocking_reason": None,
        "desired_qty": desired,
        "normal_allowed_qty": normal_allowed,
        "canary_qty": selected,
        "downscaled_for_canary": selected < normal_allowed,
        "local_share_price": str(entry_price_local),
        "fx_rate_to_eur": str(fx_rate_to_eur),
        "share_price_eur": str(share_price_eur),
        "actual_notional_eur": str(actual_notional),
        "actual_portfolio_weight": str(actual_notional / account_equity_eur),
        "risk_per_share_eur": str(risk_per_share_eur),
        "planned_total_risk_eur": str(planned_risk),
        "normal_risk_budget_eur": str(normal_risk_budget_eur),
        "canary_risk_budget_eur": str(canary_risk_budget_eur),
        "canary_risk_utilization": str(
            planned_risk / canary_risk_budget_eur
            if canary_risk_budget_eur > 0
            else Decimal("0")
        ),
        "estimated_commission_eur": str(
            selected_costs["commission_eur"]
        ),
        "estimated_spread_eur": str(selected_costs["spread_eur"]),
        "estimated_slippage_eur": str(selected_costs["slippage_eur"]),
        "estimated_total_cost_eur": str(estimated_costs),
        "expected_gross_upside_eur": str(expected_gross_upside_eur),
        "expected_net_opportunity_eur": str(expected_net_opportunity_eur),
        "cost_to_expected_edge_ratio": str(
            estimated_costs / expected_gross_upside_eur
        ),
        "cash_before_eur": str(spendable_cash),
        "cash_after_eur": str(
            spendable_cash - actual_notional - estimated_costs
        ),
        "hard_notional_backstop_eur": str(policy.hard_notional_cap_eur),
        "quantity_limits": limits,
    }


def publish_level_one_canary_policy(project_root: Path) -> dict[str, Any]:
    """Publish a versioned, non-authorizing Level-1 policy artifact."""

    from stocks.capital.service import capital_level_limits

    policy = load_level_one_canary_policy(project_root)
    private = _read_json(
        project_root / "data/portfolio/private/current-state.json"
    )
    account = private.get("account_state", {})
    equity = Decimal(str(account.get("net_liquidation_eur") or "0"))
    resolved = (
        capital_level_limits(
            project_root,
            level=1,
            account_equity_eur=equity,
        )
        if equity > 0
        else {}
    )
    report: dict[str, Any] = {
        "schema": "whole_share_canary_policy_artifact_v1",
        "status": "GO" if resolved else "NO_GO",
        "capital_level": 1,
        "capital_level_name": "WHOLE_SHARE_EXECUTION_CANARY",
        "policy_version": policy.policy_version,
        "fractional_shares_allowed": False,
        "whole_share_required": True,
        "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
        "account_equity_eur": str(equity) if equity > 0 else None,
        "canary_risk_budget_eur": resolved.get(
            "maximum_risk_per_trade_eur"
        ),
        "hard_notional_backstop_eur": str(
            policy.hard_notional_cap_eur
        ),
        "notional_cap_role": "SECONDARY_EMERGENCY_BACKSTOP",
        "normal_limits": {
            "small_account_stock_weight": "0.35",
            "small_account_pooled_vehicle_weight": "0.40",
            "small_account_portfolio_heat": "0.06",
        },
        "level_one_limits": resolved,
        "promotion_requirements": {
            "minimum_verified_whole_share_round_trips": 5,
            "complete_broker_lifecycle_required": True,
            "financial_finalist_evidence_required": True,
            "strategy_deployment_evidence_required": True,
            "operator_approval_required": True,
            "automatic_promotion": False,
        },
        "legacy_fixed_notional_eur": "10",
        "legacy_fixed_notional_authoritative": False,
        "small_account_matrix": build_small_account_canary_matrix(
            project_root
        ),
        "level_two_activated": False,
        "broker_calls": 0,
        "broker_writes": 0,
    }
    report["policy_hash"] = stable_hash(
        {"configured_policy": policy.jsonable(), "resolved_limits": resolved}
    )
    report["content_hash"] = stable_hash(report)
    _write_json(
        project_root
        / "output/ibkr/live/whole-share-canary-policy-v1.json",
        report,
    )
    return report


def build_small_account_canary_matrix(
    project_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equities = (1000, 1870, 2500, 5000, 10000)
    prices = (5, 10, 25, 50, 70, 100, 150, 250, 500, 1000, 2000)
    for equity_value in equities:
        equity = Decimal(equity_value)
        for price_value in prices:
            price = Decimal(price_value)
            desired_notional = equity * Decimal("0.10")
            desired_qty = max(1, _whole(desired_notional / price))
            result = evaluate_whole_share_canary(
                project_root,
                asset_class="STOCK",
                instrument_currency="EUR",
                desired_qty=Decimal(desired_qty),
                account_equity_eur=equity,
                available_cash_eur=equity,
                reserved_cash_eur=Decimal("0"),
                entry_price_local=price,
                protective_stop_local=price * Decimal("0.97"),
                take_profit_local=price * Decimal("1.15"),
                fx_rate_to_eur=Decimal("1"),
                normal_risk_budget_eur=equity * Decimal("0.01"),
                normal_maximum_position_weight=Decimal("0.35"),
                normal_maximum_portfolio_heat_pct=Decimal("0.06"),
                liquidity_notional_eur=Decimal("1000000"),
            )
            rows.append(
                {
                    "account_equity_eur": str(equity),
                    "share_price_eur": str(price),
                    "desired_allocation_eur": str(desired_notional),
                    "stop_price_eur": str(price * Decimal("0.97")),
                    "risk_per_share_eur": result.get(
                        "risk_per_share_eur"
                    ),
                    "desired_qty": result.get("desired_qty"),
                    "normal_allowed_qty": result.get("normal_allowed_qty"),
                    "level1_canary_qty": result.get("canary_qty"),
                    "actual_notional_eur": result.get(
                        "actual_notional_eur", "0"
                    ),
                    "actual_portfolio_weight": result.get(
                        "actual_portfolio_weight", "0"
                    ),
                    "planned_loss_at_stop_eur": result.get(
                        "planned_total_risk_eur", "0"
                    ),
                    "cash_after_eur": result.get("cash_after_eur"),
                    "status": result.get("sizing_reason"),
                    "blocking_reason": result.get("blocking_reason"),
                }
            )
    return rows


def publish_current_canary_preview(project_root: Path) -> dict[str, Any]:
    """Publish the latest portfolio funnel without creating an intent."""

    private = _read_json(
        project_root / "data/portfolio/private/current-state.json"
    )
    whole_share = private.get("whole_share_sizing", {})
    preflight = whole_share.get("candidate_preflight", {})
    candidates = [
        row
        for row in preflight.get("candidate_results", [])
        if isinstance(row, dict)
    ]
    rows = []
    for row in candidates:
        rows.append(
            {
                "symbol": row.get("ticker"),
                "asset_class": row.get("asset_class", "STOCK"),
                "score": row.get("opportunity_score"),
                "local_share_price": row.get("share_price_local"),
                "currency": row.get("currency"),
                "account_currency_share_price": row.get("share_price_eur"),
                "desired_qty": row.get("desired_qty"),
                "normal_allowed_qty": row.get("normal_allowed_qty"),
                "level1_canary_qty": row.get("level1_canary_qty"),
                "actual_notional_eur": row.get("actual_notional_eur"),
                "planned_risk_eur": row.get("planned_risk_eur"),
                "actual_portfolio_weight": row.get(
                    "actual_portfolio_weight"
                ),
                "cash_after_eur": row.get("cash_after_eur"),
                "status": row.get("canary_sizing_reason"),
                "blocking_reason": row.get("canary_blocking_reason"),
            }
        )
    executable = sum(
        int(Decimal(str(row.get("level1_canary_qty") or "0")) >= 1)
        for row in rows
    )
    report: dict[str, Any] = {
        "schema": "current_whole_share_canary_preview_v1",
        "status": "GO" if preflight.get("status") == "GO" else "NO_GO",
        "snapshot_hash": whole_share.get("account_snapshot_hash"),
        "candidate_count": len(rows),
        "executable_level_one_candidate_count": executable,
        "candidates": rows,
        "selection_basis": "EXPECTED_NET_OPPORTUNITY_AND_RISK_NOT_SHARE_PRICE",
        "intent_created": False,
        "orders_submitted": 0,
        "orders_cancelled": 0,
        "orders_modified": 0,
        "fx_transactions": 0,
        "broker_writes": 0,
        "level_two_promotion": False,
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(report)
    _write_json(
        project_root / "output/ibkr/live/current-canary-preview.json",
        report,
    )
    return report


def _zero_quantity_reason(
    limits: dict[str, int], normal_allowed: int, maximum_canary: int
) -> str:
    if limits["qty_by_normal_risk"] < 1 or limits["qty_by_canary_risk"] < 1:
        return "WHOLE_SHARE_RISK_INFEASIBLE"
    if limits["qty_by_cash"] < 1:
        return "INSUFFICIENT_CASH_FOR_ONE_SHARE"
    if (
        limits["qty_by_normal_concentration"] < 1
        or limits["qty_by_canary_concentration"] < 1
    ):
        return "CONCENTRATION_LIMIT"
    if (
        limits["qty_by_normal_portfolio_heat"] < 1
        or limits["qty_by_canary_portfolio_heat"] < 1
    ):
        return "PORTFOLIO_HEAT_LIMIT"
    if limits["qty_by_total_exposure"] < 1:
        return "TOTAL_EXPOSURE_LIMIT"
    if limits["qty_by_liquidity"] < 1:
        return "LIQUIDITY_LIMIT"
    if limits["qty_by_notional_backstop"] < 1:
        return "CANARY_NOTIONAL_BACKSTOP"
    if limits["qty_by_asset_policy"] < 1:
        return "ASSET_QUANTITY_POLICY_LIMIT"
    if normal_allowed < 1 or maximum_canary < 1:
        return "CANARY_NON_EXECUTABLE_WHOLE_SHARE"
    return "CANARY_NON_EXECUTABLE_WHOLE_SHARE"


def _blocked(base: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **base,
        "status": "NO_GO",
        "sizing_reason": reason,
        "blocking_reason": reason,
        "desired_qty": 0,
        "normal_allowed_qty": 0,
        "canary_qty": 0,
    }


def _whole(value: Decimal) -> int:
    if not value.is_finite() or value <= 0:
        return 0
    return int(value.to_integral_value(rounding=ROUND_DOWN))


def _positive_decimal(value: Any, name: str) -> Decimal:
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name}_INVALID")
    return parsed


def _fraction(value: Any, name: str) -> Decimal:
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0 or parsed > 1:
        raise ValueError(f"{name}_INVALID")
    return parsed


def _positive_integer(value: Any, name: str) -> int:
    parsed = Decimal(str(value))
    if (
        not parsed.is_finite()
        or parsed < 1
        or parsed != parsed.to_integral_value()
    ):
        raise ValueError(f"{name}_INVALID")
    return int(parsed)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "LevelOneCanaryPolicy",
    "SUPPORTED_ASSET_CLASSES",
    "default_level_one_canary_policy",
    "build_small_account_canary_matrix",
    "evaluate_whole_share_canary",
    "load_level_one_canary_policy",
    "publish_current_canary_preview",
    "publish_level_one_canary_policy",
]
