from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.portfolio.p1_contracts import (
    OpportunityAssetClass,
    OpportunityShariahStatus,
)


UNIVERSE_PATH = Path("output/universe/instruments.parquet")
CONTRACT_PATH = Path("output/ibkr/contracts/stocks.parquet")
DETAIL_PATH = Path("output/portfolio/coverage-waterfall.parquet")
SUMMARY_PATH = Path("output/portfolio/coverage-waterfall.json")
SUPPORTED_CURRENCIES = frozenset({"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "USD"})
COVERAGE_REASONS = (
    "ELIGIBLE_AND_ANALYZABLE",
    "NO_PRICE_DATA",
    "INSUFFICIENT_HISTORY",
    "NO_IBKR_CONTRACT",
    "AMBIGUOUS_CONTRACT",
    "NO_SHARIAH_CLASSIFICATION",
    "SHARIAH_BLOCKED",
    "NO_FUNDAMENTAL_DATA",
    "NO_REQUIRED_STRATEGY_FEATURES",
    "ILLIQUID",
    "UNSUPPORTED_ASSET_TYPE",
    "INVALID_CURRENCY",
    "DELISTED_OR_INACTIVE",
    "DUPLICATE_INSTRUMENT",
    "OTHER",
)
STAGES = (
    "discovered",
    "data_analyzable",
    "research_eligible",
    "shariah_eligible",
    "broker_resolvable",
    "strategy_signal",
    "qualified",
    "ranked",
    "portfolio_candidate",
    "whole_share_feasible",
    "execution_preflight_eligible",
)


def normalize_asset_class(row: dict[str, Any]) -> str:
    instrument_type = str(row.get("instrument_type") or "").upper()
    asset_type = str(row.get("asset_type") or "").upper()
    exposure = str(row.get("commodity_exposure_type") or "").upper()
    if instrument_type == "COMMODITY_EXPOSURE" or exposure not in {
        "",
        "NONE",
        "UNSPECIFIED",
    }:
        return OpportunityAssetClass.COMMODITY_EXPOSURE.value
    if instrument_type == "ETF" or asset_type.endswith("ETF"):
        return OpportunityAssetClass.ETF.value
    return OpportunityAssetClass.EQUITY.value


def normalize_asset_subclass(row: dict[str, Any]) -> str:
    asset_class = normalize_asset_class(row)
    if asset_class == OpportunityAssetClass.COMMODITY_EXPOSURE.value:
        structure = str(row.get("product_structure") or "").upper()
        exposure = str(row.get("commodity_exposure_type") or "").upper()
        if structure.startswith("PHYSICAL_BACKED") or structure.startswith(
            "PHYSICAL_CLOSED"
        ):
            return "PHYSICAL_BACKED_ETF"
        if "FUTURES" in structure or "FUTURES" in exposure:
            return "FUTURES_BACKED_ETF"
        if "INDEX" in structure or exposure == "FUTURES_BASKET":
            return "COMMODITY_INDEX_ETF"
        if "PRODUCER" in exposure:
            return "PRODUCER_EQUITY"
        if "MINER" in exposure:
            return "MINER"
        if "ROYALTY" in exposure or "STREAM" in exposure:
            return "ROYALTY_STREAMING_COMPANY"
        if "ENERGY" in exposure:
            return "ENERGY_EQUITY"
        return "OTHER_REAL_ASSET_EXPOSURE"
    if asset_class == OpportunityAssetClass.ETF.value:
        sector = str(row.get("sector") or "").upper()
        group = str(row.get("group") or "").upper()
        if "SHARIAH" in sector:
            return "SHARIAH"
        if "FACTOR" in sector or "FACTOR" in group:
            return "FACTOR"
        if "COUNTRY" in group or str(row.get("region") or "GLOBAL") not in {
            "GLOBAL",
            "UNITED_STATES",
        }:
            return "COUNTRY"
        if sector in {"BROAD_MARKET", "LARGE_CAP", "SMALL_CAP"}:
            return "BROAD_MARKET"
        return "SECTOR"
    market_cap = _number(row.get("market_cap"))
    if market_cap >= 200_000_000_000:
        return "MEGA_CAP"
    if market_cap >= 10_000_000_000:
        return "LARGE_CAP"
    if market_cap >= 2_000_000_000:
        return "MID_CAP"
    if market_cap > 0:
        return "SMALL_CAP"
    return "EQUITY_CAP_UNKNOWN"


def normalize_shariah_status(value: Any) -> str:
    status = str(value or "").upper()
    if status in {
        "SHARIAH_ALLOWED",
        "SHARIAH_COMPLIANT",
        "SHARIAH_ELIGIBLE_PIT",
        "ELIGIBLE",
    }:
        return OpportunityShariahStatus.ALLOWED.value
    if any(token in status for token in ("BLOCKED", "INELIGIBLE", "NON_COMPLIANT")):
        return OpportunityShariahStatus.BLOCKED.value
    if any(token in status for token in ("INCOMPLETE", "REVIEW")):
        return OpportunityShariahStatus.REVIEW_REQUIRED.value
    return OpportunityShariahStatus.DATA_MISSING.value


def build_coverage_waterfall(
    project_root: Path,
    *,
    ranked: Iterable[dict[str, Any]] = (),
    signals: Iterable[dict[str, Any]] = (),
    portfolio_symbols: Iterable[str] = (),
    whole_share_symbols: Iterable[str] = (),
) -> dict[str, Any]:
    universe_path = project_root / UNIVERSE_PATH
    if not universe_path.is_file():
        return _blocked("DISCOVERY_UNIVERSE_MISSING")
    universe = pd.read_parquet(universe_path)
    contract_counts = _contract_counts(project_root)
    data_symbols, insufficient = _local_data_symbols(project_root)
    signal_symbols = {
        str(row.get("ticker") or row.get("asset") or "").upper()
        for row in signals
        if row.get("ticker") or row.get("asset")
    }
    ranked_rows = {
        str(row.get("ticker") or "").upper(): row
        for row in ranked
        if row.get("ticker")
    }
    portfolio = {str(value).upper() for value in portfolio_symbols}
    whole_share = {str(value).upper() for value in whole_share_symbols}
    duplicate_symbols = set(
        universe.loc[
            universe["symbol"].astype(str).str.upper().duplicated(False),
            "symbol",
        ]
        .astype(str)
        .str.upper()
    )
    rows: list[dict[str, Any]] = []
    for raw in universe.to_dict(orient="records"):
        symbol = str(raw.get("symbol") or "").upper()
        asset_class = normalize_asset_class(raw)
        active = bool(raw.get("active_listing", True))
        supported = asset_class in {
            OpportunityAssetClass.EQUITY.value,
            OpportunityAssetClass.ETF.value,
            OpportunityAssetClass.COMMODITY_EXPOSURE.value,
        }
        currency_valid = str(raw.get("currency") or "").upper() in SUPPORTED_CURRENCIES
        has_data = symbol in data_symbols
        data_analyzable = active and supported and currency_valid and has_data and symbol not in insufficient
        shariah = normalize_shariah_status(raw.get("compliance_status"))
        research_eligible = data_analyzable
        shariah_eligible = research_eligible and shariah == OpportunityShariahStatus.ALLOWED.value
        contract_count = contract_counts.get(symbol, 0)
        broker_resolvable = shariah_eligible and contract_count == 1
        strategy_signal = research_eligible and symbol in signal_symbols
        ranked_row = ranked_rows.get(symbol, {})
        qualified = bool(ranked_row.get("research_allocation_eligible", False))
        ranked_stage = symbol in ranked_rows
        portfolio_candidate = symbol in portfolio
        whole_share_feasible = symbol in whole_share
        execution_preflight = bool(
            portfolio_candidate
            and whole_share_feasible
            and ranked_row.get("deployment_eligible", False)
        )
        reason = _coverage_reason(
            active=active,
            duplicate=symbol in duplicate_symbols,
            supported=supported,
            currency_valid=currency_valid,
            has_data=has_data,
            insufficient=symbol in insufficient,
            shariah=shariah,
            contract_count=contract_count,
            strategy_signal=strategy_signal,
            qualified=qualified,
        )
        rows.append(
            {
                "instrument_id": str(raw.get("instrument_id") or symbol),
                "symbol": symbol,
                "asset_class": asset_class,
                "subclass": normalize_asset_subclass(raw),
                "coverage_reason": reason,
                "shariah_status": shariah,
                "discovered": True,
                "data_analyzable": data_analyzable,
                "research_eligible": research_eligible,
                "shariah_eligible": shariah_eligible,
                "broker_resolvable": broker_resolvable,
                "strategy_signal": strategy_signal,
                "qualified": qualified,
                "ranked": ranked_stage,
                "portfolio_candidate": portfolio_candidate,
                "whole_share_feasible": whole_share_feasible,
                "execution_preflight_eligible": execution_preflight,
                "execution_authority": "NONE",
            }
        )
    frame = pd.DataFrame(rows)
    summary = _summarize(frame)
    summary.update(
        {
            "schema": "multi_asset_coverage_waterfall_v1",
            "status": "GO",
            "generated_at": datetime.now(UTC).isoformat(),
            "coverage_reasons": list(COVERAGE_REASONS),
            "recoverable_categories": _recoverable_categories(frame),
            "detail_artifact": DETAIL_PATH.as_posix(),
            "research_execution_separated": True,
            "missing_shariah_is_not_blocked": True,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    )
    summary["content_hash"] = stable_hash(summary)
    _write(project_root, frame, summary)
    return summary


def _coverage_reason(**state: Any) -> str:
    if not state["active"]:
        return "DELISTED_OR_INACTIVE"
    if state["duplicate"]:
        return "DUPLICATE_INSTRUMENT"
    if not state["supported"]:
        return "UNSUPPORTED_ASSET_TYPE"
    if not state["currency_valid"]:
        return "INVALID_CURRENCY"
    if not state["has_data"]:
        return "NO_PRICE_DATA"
    if state["insufficient"]:
        return "INSUFFICIENT_HISTORY"
    if state["shariah"] == OpportunityShariahStatus.BLOCKED.value:
        return "SHARIAH_BLOCKED"
    if state["shariah"] in {
        OpportunityShariahStatus.DATA_MISSING.value,
        OpportunityShariahStatus.REVIEW_REQUIRED.value,
    }:
        return "NO_SHARIAH_CLASSIFICATION"
    if state["contract_count"] > 1:
        return "AMBIGUOUS_CONTRACT"
    if state["contract_count"] == 0:
        return "NO_IBKR_CONTRACT"
    if not state["strategy_signal"]:
        return "NO_REQUIRED_STRATEGY_FEATURES"
    if not state["qualified"]:
        return "NO_FUNDAMENTAL_DATA"
    return "ELIGIBLE_AND_ANALYZABLE"


def _summarize(frame: pd.DataFrame) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for asset_class in (
        "ALL",
        OpportunityAssetClass.EQUITY.value,
        OpportunityAssetClass.ETF.value,
        OpportunityAssetClass.COMMODITY_EXPOSURE.value,
    ):
        selected = frame if asset_class == "ALL" else frame.loc[frame["asset_class"] == asset_class]
        total = len(selected)
        groups[asset_class] = {
            "instrument_count": total,
            "stages": {
                stage: int(selected[stage].sum()) for stage in STAGES
            },
            "stage_percentages": {
                stage: round(100.0 * float(selected[stage].sum()) / total, 6)
                if total
                else 0.0
                for stage in STAGES
            },
            "rejection_reasons": dict(
                sorted(Counter(selected["coverage_reason"]).items())
            ),
            "shariah_statuses": dict(
                sorted(Counter(selected["shariah_status"]).items())
            ),
        }
    return {"funnels": groups}


def _recoverable_categories(frame: pd.DataFrame) -> list[dict[str, Any]]:
    recoverable = {
        "NO_PRICE_DATA": 1.0,
        "NO_SHARIAH_CLASSIFICATION": 0.9,
        "NO_IBKR_CONTRACT": 0.7,
        "NO_REQUIRED_STRATEGY_FEATURES": 0.8,
        "NO_FUNDAMENTAL_DATA": 0.6,
        "INSUFFICIENT_HISTORY": 0.5,
    }
    rows = []
    for reason, count in Counter(frame["coverage_reason"]).items():
        if reason in recoverable:
            by_class = Counter(
                frame.loc[frame["coverage_reason"] == reason, "asset_class"]
            )
            rows.append(
                {
                    "reason": reason,
                    "instrument_count": int(count),
                    "recovery_priority_score": round(count * recoverable[reason], 3),
                    "asset_class_counts": dict(sorted(by_class.items())),
                }
            )
    return sorted(rows, key=lambda row: (-row["recovery_priority_score"], row["reason"]))


def _contract_counts(project_root: Path) -> Counter[str]:
    path = project_root / CONTRACT_PATH
    if not path.is_file():
        return Counter()
    frame = pd.read_parquet(path, columns=["symbol", "con_id"])
    return Counter(frame["symbol"].astype(str).str.upper())


def _local_data_symbols(project_root: Path) -> tuple[set[str], set[str]]:
    symbols: set[str] = set()
    insufficient: set[str] = set()
    daily_root = project_root / "data/research/critical_trading/yfinance"
    for path in daily_root.glob("*.parquet") if daily_root.exists() else ():
        symbol = path.stem.upper()
        symbols.add(symbol)
        try:
            rows = len(pd.read_parquet(path, columns=["close"]))
        except (OSError, ValueError, KeyError):
            rows = 0
        if rows < 60:
            insufficient.add(symbol)
    mtf_root = project_root / "data/research/multitimeframe/private"
    if mtf_root.exists():
        for path in mtf_root.rglob("bars.parquet"):
            symbol_part = next(
                (part for part in path.parts if part.startswith("symbol=")),
                "",
            )
            if symbol_part:
                symbols.add(symbol_part.split("=", 1)[1].upper())
    return symbols, insufficient


def _write(project_root: Path, frame: pd.DataFrame, report: dict[str, Any]) -> None:
    detail = project_root / DETAIL_PATH
    detail.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(detail, index=False)
    path = project_root / SUMMARY_PATH
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "multi_asset_coverage_waterfall_v1",
        "status": "NO_GO",
        "blockers": [reason],
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


__all__ = [
    "COVERAGE_REASONS",
    "STAGES",
    "build_coverage_waterfall",
    "normalize_asset_class",
    "normalize_asset_subclass",
    "normalize_shariah_status",
]
