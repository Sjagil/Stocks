from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.costs import estimate_transaction_cost, load_shared_cost_model
from stocks.execution.idempotency import stable_hash
from stocks.portfolio.coverage import (
    normalize_asset_class,
    normalize_asset_subclass,
    normalize_shariah_status,
)
from stocks.portfolio.p1_contracts import (
    NormalizedOpportunity,
    OpportunityAssetClass,
    OpportunityShariahStatus,
    ValidationStatus,
)


OUTPUT_PATH = Path("output/portfolio/normalized-opportunities.json")


def annotate_opportunity_economics(
    project_root: Path,
    ranked: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    model = load_shared_cost_model(project_root)
    annotated: list[dict[str, Any]] = []
    for original in ranked:
        row = dict(original)
        price = _decimal(row.get("preferred_entry"))
        stop = _decimal(row.get("stop_loss"))
        target = _decimal(row.get("take_profit_1"))
        confidence = _bounded(
            row.get("components", {}).get(
                "signal_quality", row.get("opportunity_score")
            )
        )
        expected_return = (
            float((target - price) / price)
            if price > 0 and target > price
            else None
        )
        expected_loss = (
            float((price - stop) / price)
            if price > 0 and 0 < stop < price
            else None
        )
        costs = estimate_transaction_cost(
            price,
            currency=str(row.get("currency") or "EUR"),
            model=model,
        )
        cost_return = (
            float(Decimal(costs["total_cost_eur"]) / price)
            if price > 0
            else None
        )
        gross = (
            confidence * expected_return
            - (1.0 - confidence) * expected_loss
            if expected_return is not None and expected_loss is not None
            else None
        )
        net = (
            gross - cost_return
            if gross is not None and cost_return is not None
            else None
        )
        net_component = (
            _bounded(0.5 + net / 0.20) if net is not None else 0.0
        )
        liquidity = _bounded(
            row.get("components", {}).get("liquidity", 0.0)
        )
        validation_confidence = _validation_confidence(row)
        portfolio_score = (
            0.40 * float(row.get("opportunity_score") or 0.0)
            + 0.30 * net_component
            + 0.15 * liquidity
            + 0.15 * validation_confidence
        )
        row.update(
            {
                "expected_return": _rounded(expected_return),
                "expected_loss": _rounded(expected_loss),
                "expected_net_return": _rounded(net),
                "expected_r": _rounded(
                    net / expected_loss
                    if net is not None and expected_loss
                    else None
                ),
                "estimated_round_trip_cost_eur_one_share": _rounded(
                    costs["total_cost_eur"]
                ),
                "validation_confidence": round(
                    validation_confidence, 8
                ),
                "portfolio_objective_score": round(
                    _bounded(portfolio_score), 8
                ),
                "portfolio_objective": (
                    "ROBUST_EXPECTED_NET_PNL_RISK_ADJUSTED_WITH_CONSTRAINTS"
                ),
            }
        )
        annotated.append(row)
    annotated.sort(
        key=lambda row: (
            -float(row["portfolio_objective_score"]),
            -float(row.get("opportunity_score") or 0.0),
            str(row.get("ticker")),
        )
    )
    return annotated


def normalize_cross_asset_opportunities(
    project_root: Path,
    *,
    ranked: Iterable[dict[str, Any]],
    stage0_report: dict[str, Any],
    sizing_rows: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    model = load_shared_cost_model(project_root)
    metadata = _metadata_map(project_root)
    sizing = {
        str(row.get("ticker") or "").upper(): row
        for row in sizing_rows
        if row.get("ticker")
    }
    opportunities: list[dict[str, Any]] = []
    ranked_symbols: set[str] = set()
    for row in ranked:
        symbol = str(row.get("ticker") or "").upper()
        if not symbol:
            continue
        ranked_symbols.add(symbol)
        instrument = metadata.get(symbol, {})
        price = _decimal(row.get("preferred_entry"))
        stop = _decimal(row.get("stop_loss"))
        target = _decimal(row.get("take_profit_1"))
        confidence = _bounded(row.get("components", {}).get("signal_quality", row.get("opportunity_score")))
        expected_return = float((target - price) / price) if price > 0 and target > price else None
        expected_loss = float((price - stop) / price) if price > 0 and 0 < stop < price else None
        whole = sizing.get(symbol, {})
        notional = _decimal(whole.get("actual_notional_eur"))
        if notional <= 0 and price > 0:
            notional = price
        costs = estimate_transaction_cost(
            notional,
            currency=str(row.get("currency") or instrument.get("currency") or "EUR"),
            model=model,
        )
        gross_expectancy = (
            confidence * expected_return - (1.0 - confidence) * expected_loss
            if expected_return is not None and expected_loss is not None
            else None
        )
        cost_return = float(Decimal(costs["total_cost_eur"]) / notional) if notional > 0 else None
        expected_net = gross_expectancy - cost_return if gross_expectancy is not None and cost_return is not None else None
        expected_r = expected_net / expected_loss if expected_net is not None and expected_loss and expected_loss > 0 else None
        blockers = tuple(sorted(set(str(value) for value in row.get("research_allocation_blockers", []))))
        shariah = normalize_shariah_status(row.get("shariah_status"))
        asset_class = normalize_asset_class({**instrument, **row})
        normalized = NormalizedOpportunity(
            instrument_id=str(instrument.get("instrument_id") or row.get("con_id") or symbol),
            symbol=symbol,
            asset_class=asset_class,
            subclass=normalize_asset_subclass({**instrument, **row}),
            strategy_family="+".join(str(value) for value in row.get("strategy_families", [])) or "UNKNOWN",
            timeframe="+".join(str(value) for value in row.get("timeframes", [])) or "UNKNOWN",
            signal_timestamp=str(row.get("data_timestamp") or "UNKNOWN"),
            signal_expiry=_first_expiry(row),
            direction="LONG",
            expected_return=_rounded(expected_return),
            expected_loss=_rounded(expected_loss),
            expected_net_return=_rounded(expected_net),
            expected_r=_rounded(expected_r),
            confidence=round(confidence, 8),
            volatility=_rounded(row.get("volatility")),
            liquidity=_rounded(row.get("components", {}).get("liquidity")),
            spread_bps=_rounded(row.get("spread_bps")),
            estimated_slippage_bps=float(model.slippage_bps),
            fees_eur=_rounded(costs["total_cost_eur"]),
            event_risk=_bounded(row.get("event_risk")),
            regime_fit=_bounded(row.get("components", {}).get("regime_fit", 0.5)),
            relative_strength=_bounded(row.get("components", {}).get("relative_strength", 0.5)),
            data_quality=1.0 if "CURRENT_DATA_STALE" not in blockers else 0.0,
            shariah_status=shariah,
            broker_resolvable=bool(row.get("contract_resolved")),
            whole_share_feasibility=str(whole.get("whole_share_feasibility_status") or "NOT_EVALUATED"),
            correlation_cluster=economic_cluster(symbol, {**instrument, **row}),
            validation_status=_validation_status(row),
            research_eligible=bool(row.get("research_allocation_eligible")),
            portfolio_eligible=bool(row.get("research_allocation_eligible") and shariah == OpportunityShariahStatus.ALLOWED.value),
            execution_eligible=bool(row.get("deployment_eligible") and whole.get("execution_candidate_status") == "EXECUTABLE_WHOLE_SHARE"),
            blockers=blockers,
            source="ACTIVE_PORTFOLIO_RANKING",
        )
        opportunities.append(normalized.as_dict())
    for row in stage0_report.get("survivors", []):
        symbol = str(row.get("symbol") or "").upper()
        if not symbol or symbol in ranked_symbols:
            continue
        instrument = metadata.get(symbol, {})
        shariah = normalize_shariah_status(instrument.get("compliance_status"))
        price = _decimal(row.get("last_price"))
        cost = estimate_transaction_cost(price, currency=str(instrument.get("currency") or "EUR"), model=model)
        cost_return = float(Decimal(cost["total_cost_eur"]) / price) if price > 0 else None
        score = _bounded(row.get("score"))
        gross_proxy = max(0.0, (score - 0.5) * 0.20)
        net_proxy = gross_proxy - cost_return if cost_return is not None else None
        normalized = NormalizedOpportunity(
            instrument_id=str(row.get("instrument_id") or symbol),
            symbol=symbol,
            asset_class=str(row.get("asset_class")),
            subclass=str(row.get("subclass")),
            strategy_family=str(row.get("strategy_family")),
            timeframe=str(row.get("timeframe")),
            signal_timestamp=str(row.get("data_timestamp")),
            signal_expiry=None,
            direction="LONG_RESEARCH_ONLY",
            expected_return=_rounded(gross_proxy),
            expected_loss=_rounded(row.get("volatility")),
            expected_net_return=_rounded(net_proxy),
            expected_r=None,
            confidence=score,
            volatility=_rounded(row.get("volatility")),
            liquidity=None,
            spread_bps=None,
            estimated_slippage_bps=float(model.slippage_bps),
            fees_eur=_rounded(cost["total_cost_eur"]),
            event_risk=0.5,
            regime_fit=0.5,
            relative_strength=_bounded(row.get("components", {}).get("medium_momentum", 0.5)),
            data_quality=1.0,
            shariah_status=shariah,
            broker_resolvable=False,
            whole_share_feasibility="NOT_EVALUATED_STAGE0_ONLY",
            correlation_cluster=economic_cluster(symbol, {**instrument, **row}),
            validation_status=ValidationStatus.STAGE0_SURVIVOR.value,
            research_eligible=True,
            portfolio_eligible=False,
            execution_eligible=False,
            blockers=("EXACT_EVENT_DRIVEN_VALIDATION_REQUIRED", "WALK_FORWARD_REQUIRED", "SHARIAH_AND_CONTRACT_PREFLIGHT_REQUIRED"),
            source="VECTORIZED_STAGE0_APPROXIMATION",
        )
        opportunities.append(normalized.as_dict())
    opportunities.append(_cash_opportunity())
    opportunities.sort(key=lambda row: (_sortable_net(row.get("expected_net_return")), row["confidence"]), reverse=True)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for asset_class in (
        OpportunityAssetClass.EQUITY.value,
        OpportunityAssetClass.ETF.value,
        OpportunityAssetClass.COMMODITY_EXPOSURE.value,
        OpportunityAssetClass.CASH.value,
    ):
        by_class[asset_class] = [row for row in opportunities if row["asset_class"] == asset_class]
    report: dict[str, Any] = {
        "schema": "normalized_cross_asset_opportunity_book_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "objective": "ROBUST_EXPECTED_NET_PNL_AND_RISK_ADJUSTED_EXPECTANCY_WITHIN_HARD_CONSTRAINTS",
        "opportunity_count": len(opportunities),
        "asset_class_counts": {key: len(value) for key, value in by_class.items()},
        "top_by_asset_class": {key: value[:20] for key, value in by_class.items()},
        "combined_ranking": opportunities,
        "cash_competes_for_capital": True,
        "stage0_direct_promotion": False,
        "whole_share_risk_engine_authoritative": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def economic_cluster(symbol: str, metadata: dict[str, Any]) -> str:
    sector = str(metadata.get("sector") or "UNKNOWN").upper()
    underlying = str(metadata.get("underlying_commodity") or "NONE").upper()
    if underlying not in {"", "NONE", "UNSPECIFIED"}:
        if "GOLD" in underlying or "PRECIOUS" in underlying:
            return "PRECIOUS_METALS_GOLD_FACTOR"
        if "SILVER" in underlying:
            return "PRECIOUS_METALS_SILVER_FACTOR"
        if "COPPER" in underlying or "INDUSTRIAL" in underlying:
            return "INDUSTRIAL_METALS_FACTOR"
        if "URANIUM" in underlying:
            return "URANIUM_NUCLEAR_FACTOR"
        if any(token in underlying for token in ("OIL", "CRUDE", "GAS", "ENERGY")):
            return "ENERGY_COMMODITY_FACTOR"
        return f"COMMODITY_{underlying}"
    if symbol.upper() in {"NVDA", "AMD", "AVGO", "SMH", "SOXX"} or "SEMICONDUCTOR" in sector:
        return "SEMICONDUCTOR_FACTOR"
    if "TECHNOLOGY" in sector:
        return "TECHNOLOGY_FACTOR"
    if "ENERGY" in sector:
        return "ENERGY_EQUITY_FACTOR"
    return f"SECTOR_{sector or 'UNKNOWN'}"


def _metadata_map(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "output/universe/instruments.parquet"
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    return {str(row["symbol"]).upper(): row.to_dict() for _, row in frame.iterrows()}


def _validation_status(row: dict[str, Any]) -> str:
    tiers = {str(value).upper() for value in row.get("evidence_tiers", [])}
    if any("FORWARD" in value for value in tiers) and any("WALK" in value or "OOS" in value for value in tiers):
        return ValidationStatus.VALIDATED.value
    return ValidationStatus.EXACT_VALIDATION_REQUIRED.value


def _validation_confidence(row: dict[str, Any]) -> float:
    tiers = {str(value).upper() for value in row.get("evidence_tiers", [])}
    if any("FORWARD" in value for value in tiers):
        return 1.0
    if any("ROBUSTNESS" in value or "OOS" in value for value in tiers):
        return 0.8
    if tiers:
        return 0.6
    return 0.35


def _first_expiry(row: dict[str, Any]) -> str | None:
    value = row.get("expiration_timestamp") or row.get("signal_expiry")
    return None if value in (None, "") else str(value)


def _cash_opportunity() -> dict[str, Any]:
    return NormalizedOpportunity(
        instrument_id="CASH-EUR",
        symbol="CASH_EUR",
        asset_class=OpportunityAssetClass.CASH.value,
        subclass="BASE_CURRENCY_CASH",
        strategy_family="CASH_OPTIONALITY",
        timeframe="CONTINUOUS",
        signal_timestamp=datetime.now(UTC).isoformat(),
        signal_expiry=None,
        direction="HOLD",
        expected_return=0.0,
        expected_loss=0.0,
        expected_net_return=0.0,
        expected_r=None,
        confidence=1.0,
        volatility=0.0,
        liquidity=1.0,
        spread_bps=0.0,
        estimated_slippage_bps=0.0,
        fees_eur=0.0,
        event_risk=0.0,
        regime_fit=0.5,
        relative_strength=0.5,
        data_quality=1.0,
        shariah_status=OpportunityShariahStatus.ALLOWED.value,
        broker_resolvable=True,
        whole_share_feasibility="NOT_APPLICABLE",
        correlation_cluster="CASH",
        validation_status=ValidationStatus.VALIDATED.value,
        research_eligible=True,
        portfolio_eligible=True,
        execution_eligible=False,
        blockers=("EXECUTION_AUTHORITY_NONE",),
        source="BASE_CURRENCY_ACCOUNT_STATE",
    ).as_dict()


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ArithmeticError, ValueError, TypeError):
        return Decimal("0")
    return result if result.is_finite() else Decimal("0")


def _bounded(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _rounded(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, 8)


def _sortable_net(value: Any) -> float:
    result = _rounded(value)
    return -999.0 if result is None else result


__all__ = [
    "annotate_opportunity_economics",
    "economic_cluster",
    "normalize_cross_asset_opportunities",
]
