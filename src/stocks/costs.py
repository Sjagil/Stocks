from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("config/costs/shared_transaction_cost_v1.json")


@dataclass(frozen=True)
class SharedTransactionCostModel:
    base_currency: str = "EUR"
    minimum_commission_eur: Decimal = Decimal("0.35")
    commission_bps: Decimal = Decimal("0")
    exchange_fees_bps: Decimal = Decimal("0")
    half_spread_bps: Decimal = Decimal("1")
    slippage_bps: Decimal = Decimal("5")
    market_impact_bps: Decimal = Decimal("1")
    fx_conversion_bps: Decimal = Decimal("10")
    minimum_practical_trade_eur: Decimal = Decimal("5")
    round_trip_for_opportunity_economics: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Decimal) else value for key, value in asdict(self).items()}


def load_shared_cost_model(project_root: Path) -> SharedTransactionCostModel:
    path = project_root / CONFIG_PATH
    if not path.is_file():
        return SharedTransactionCostModel()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SharedTransactionCostModel(
        base_currency=str(raw.get("base_currency") or "EUR").upper(),
        minimum_commission_eur=Decimal(str(raw["commission"]["minimum_per_order_eur"])),
        commission_bps=Decimal(str(raw["commission"].get("variable_bps", 0))),
        exchange_fees_bps=Decimal(str(raw.get("exchange_fees_bps", 0))),
        half_spread_bps=Decimal(str(raw.get("half_spread_bps", 0))),
        slippage_bps=Decimal(str(raw.get("slippage_bps", 0))),
        market_impact_bps=Decimal(str(raw.get("market_impact_bps", 0))),
        fx_conversion_bps=Decimal(str(raw.get("fx_conversion_bps", 0))),
        minimum_practical_trade_eur=Decimal(str(raw.get("minimum_practical_trade_eur", 0))),
        round_trip_for_opportunity_economics=bool(raw.get("round_trip_for_opportunity_economics", True)),
    )


def estimate_transaction_cost(
    notional_eur: Decimal,
    *,
    currency: str = "EUR",
    model: SharedTransactionCostModel | None = None,
    round_trip: bool | None = None,
) -> dict[str, Decimal | bool | str]:
    model = model or SharedTransactionCostModel()
    notional = max(Decimal("0"), Decimal(notional_eur))
    sides = Decimal("2") if (model.round_trip_for_opportunity_economics if round_trip is None else round_trip) else Decimal("1")
    commission_per_side = max(
        model.minimum_commission_eur,
        notional * model.commission_bps / Decimal("10000"),
    ) if notional > 0 else Decimal("0")
    commission = commission_per_side * sides
    exchange = notional * model.exchange_fees_bps / Decimal("10000") * sides
    spread = notional * model.half_spread_bps / Decimal("10000") * sides
    slippage = notional * model.slippage_bps / Decimal("10000") * sides
    impact = notional * model.market_impact_bps / Decimal("10000") * sides
    fx = (
        notional * model.fx_conversion_bps / Decimal("10000") * sides
        if str(currency).upper() != model.base_currency
        else Decimal("0")
    )
    total = commission + exchange + spread + slippage + impact + fx
    return {
        "commission_eur": commission,
        "exchange_fees_eur": exchange,
        "spread_eur": spread,
        "slippage_eur": slippage,
        "market_impact_eur": impact,
        "fx_conversion_eur": fx,
        "total_cost_eur": total,
        "economic_minimum_met": notional >= model.minimum_practical_trade_eur,
        "cost_model": "SHARED_TRANSACTION_COST_MODEL_V1",
    }


def whole_share_economics(
    *,
    desired_notional_eur: Decimal,
    price_eur: Decimal,
    risk_budget_eur: Decimal,
    risk_per_share_eur: Decimal,
    available_cash_eur: Decimal,
    expected_gross_return: Decimal | None,
    currency: str,
    model: SharedTransactionCostModel,
) -> dict[str, Any]:
    if price_eur <= 0 or risk_per_share_eur <= 0:
        quantity = Decimal("0")
    else:
        quantity = min(
            (risk_budget_eur / risk_per_share_eur).to_integral_value(rounding=ROUND_DOWN),
            (available_cash_eur / price_eur).to_integral_value(rounding=ROUND_DOWN),
        )
    actual_notional = quantity * price_eur
    costs = estimate_transaction_cost(actual_notional, currency=currency, model=model)
    expected_profit = (
        actual_notional * expected_gross_return - Decimal(costs["total_cost_eur"])
        if expected_gross_return is not None and quantity >= 1
        else None
    )
    return {
        "desired_notional_eur": str(desired_notional_eur),
        "share_price_eur": str(price_eur),
        "risk_based_qty": str((risk_budget_eur / risk_per_share_eur).to_integral_value(rounding=ROUND_DOWN)) if risk_per_share_eur > 0 else "0",
        "capital_based_qty": str((available_cash_eur / price_eur).to_integral_value(rounding=ROUND_DOWN)) if price_eur > 0 else "0",
        "whole_share_qty": str(quantity),
        "actual_notional_eur": str(actual_notional),
        "actual_risk_eur": str(quantity * risk_per_share_eur),
        "estimated_cost_eur": str(costs["total_cost_eur"]),
        "expected_net_profit_eur": None if expected_profit is None else str(expected_profit),
        "execution_candidate_status": "EXECUTABLE_WHOLE_SHARE" if quantity >= 1 else "NON_EXECUTABLE_WHOLE_SHARE",
    }


def cost_calibration_evidence(project_root: Path) -> dict[str, Any]:
    databases = [
        project_root / "data/execution/phase9/private/paper_execution.sqlite3",
        project_root / "data/execution/live/private/live_execution.sqlite3",
    ]
    execution_count = 0
    commission_count = 0
    for path in databases:
        if not path.is_file():
            continue
        try:
            with sqlite3.connect(path) as connection:
                execution_count += int(connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0])
                commission_count += int(connection.execute("SELECT COUNT(*) FROM commissions").fetchone()[0])
        except sqlite3.Error:
            continue
    matched = min(execution_count, commission_count)
    return {
        "schema": "shared_transaction_cost_calibration_v1",
        "status": "OBSERVED_CALIBRATION_AVAILABLE" if matched else "BASELINE_CONFIGURED_OBSERVED_CALIBRATION_PENDING",
        "execution_observation_count": execution_count,
        "commission_observation_count": commission_count,
        "matched_cost_observation_count": matched,
        "predicted_vs_observed_comparison_available": matched > 0,
        "model_path": CONFIG_PATH.as_posix(),
        "research_model": "SHARED_TRANSACTION_COST_MODEL_V1",
        "portfolio_model": "SHARED_TRANSACTION_COST_MODEL_V1",
        "paper_model": "SHARED_TRANSACTION_COST_MODEL_V1",
        "execution_preflight_model": "SHARED_TRANSACTION_COST_MODEL_V1",
        "parallel_financial_ledger_created": False,
        "execution_authority": "NONE",
    }


__all__ = [
    "SharedTransactionCostModel",
    "cost_calibration_evidence",
    "estimate_transaction_cost",
    "load_shared_cost_model",
    "whole_share_economics",
]
