from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.costs import (
    cost_calibration_evidence,
    load_shared_cost_model,
    whole_share_economics,
)
from stocks.execution.idempotency import stable_hash
from stocks.portfolio.manager import build_active_portfolio_report
from stocks.portfolio.monitoring import publish_monitoring_architecture
from stocks.portfolio.p1_audit import publish_p1_requirement_audit
from stocks.research.performance_validation import (
    publish_independent_performance_check,
)
from stocks.research.strategy_inventory import publish_strategy_inventory
from stocks.research.walk_forward import publish_standard_walk_forward_manifests


OUTPUT_PATH = Path("output/portfolio/p1-readiness.json")
BASELINE_ANALYZABLE = 297
COMMODITY_FAMILIES = {
    "GOLD": ("GOLD", "PRECIOUS"),
    "SILVER": ("SILVER",),
    "COPPER": ("COPPER", "INDUSTRIAL_METALS"),
    "URANIUM": ("URANIUM",),
    "OIL": ("OIL", "CRUDE", "GASOLINE"),
    "ENERGY": ("ENERGY", "OIL", "CRUDE", "NATURAL_GAS"),
}


def run_p1_readiness(project_root: Path) -> dict[str, Any]:
    p0 = verify_p0(project_root)
    portfolio = build_active_portfolio_report(project_root)
    coverage = portfolio["coverage_waterfall"]
    normalized = portfolio["normalized_opportunities"]
    monitoring = publish_monitoring_architecture(project_root)
    walk_forward = publish_standard_walk_forward_manifests(project_root)
    inventory = publish_strategy_inventory(project_root)
    performance = publish_independent_performance_check(project_root)
    costs = cost_calibration_evidence(project_root)
    contracts = contract_identity_audit(project_root)
    shariah = shariah_coverage_audit(project_root)
    commodities = commodity_family_status(project_root, normalized)
    capital = capital_size_coherence(project_root)
    attribution = portfolio["performance_attribution"]
    checks = _completion_checks(
        p0=p0,
        portfolio=portfolio,
        coverage=coverage,
        normalized=normalized,
        monitoring=monitoring,
        walk_forward=walk_forward,
        costs=costs,
        commodities=commodities,
        capital=capital,
    )
    unresolved = [name for name, passed in checks.items() if not passed]
    funnels = coverage.get("funnels", {})
    report: dict[str, Any] = {
        "schema": "p1_multi_asset_opportunity_readiness_v1",
        "status": "GO" if not unresolved else "PARTIAL",
        "marker": "P1_COMPLETE" if not unresolved else "P1_PARTIAL",
        "generated_at": datetime.now(UTC).isoformat(),
        "objective": (
            "MAXIMIZE_ROBUST_EXPECTED_NET_PNL_AND_RISK_ADJUSTED_EXPECTANCY_"
            "WITHIN_HARD_CONSTRAINTS"
        ),
        "p0": p0,
        "coverage_before_after": {
            "baseline_analyzable": BASELINE_ANALYZABLE,
            "current_analyzable": funnels.get("ALL", {}).get("stages", {}).get(
                "data_analyzable", 0
            ),
            "material_improvement_threshold": int(BASELINE_ANALYZABLE * 1.25),
            "by_asset_class": {
                name: funnels.get(name, {})
                for name in ("EQUITY", "ETF", "COMMODITY_EXPOSURE")
            },
        },
        "lost_coverage_root_causes": coverage.get("recoverable_categories", []),
        "commodity_real_asset_status": commodities,
        "normalized_opportunity_schema": {
            "schema": normalized.get("schema"),
            "field_contract": normalized.get("field_contract", []),
            "opportunity_count": normalized.get("opportunity_count", 0),
            "asset_class_counts": normalized.get("asset_class_counts", {}),
        },
        "research_acceleration": portfolio["vectorized_stage0"],
        "walk_forward_standardization": walk_forward,
        "cost_model": costs,
        "strategy_inventory": inventory,
        "independent_performance_check": performance,
        "contract_identity": contracts,
        "shariah_coverage": shariah,
        "monitoring": monitoring,
        "cross_asset_intelligence": portfolio[
            "cross_asset_intelligence"
        ],
        "overlap": portfolio["overlap"],
        "portfolio_targets": portfolio["desired_targets"],
        "capital_size_coherence": capital,
        "performance_attribution": attribution,
        "completion_checks": checks,
        "unresolved_completion_checks": unresolved,
        "remaining_gaps": _remaining_gaps(
            unresolved=unresolved,
            shariah=shariah,
            contracts=contracts,
            costs=costs,
            strategy=inventory,
            p0=p0,
        ),
        "live_side_effects": {
            "orders_placed": 0,
            "orders_cancelled": 0,
            "orders_modified": 0,
            "fx_transactions": 0,
            "authority_increase": False,
            "broker_write_calls": 0,
        },
        "execution_authority": "NONE",
    }
    requirement_audit = publish_p1_requirement_audit(project_root, report)
    report["requirement_audit"] = requirement_audit
    if requirement_audit["status"] != "GO":
        report["status"] = "PARTIAL"
        report["marker"] = "P1_PARTIAL"
    report["content_hash"] = stable_hash(report)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def verify_p0(project_root: Path) -> dict[str, Any]:
    readiness = _read_json(
        project_root / "output/ibkr/live/p0-execution-readiness.json"
    )
    matrix = _read_json(
        project_root / "output/ibkr/phase9/p0-safety-matrix.json"
    )
    sub_gates = readiness.get("sub_gates", {})
    matrix_statuses = matrix.get("scenario_statuses", {})
    current_source_hashes = {
        relative: _file_hash(project_root / relative)
        for relative in matrix.get("source_hashes", {})
    }
    stored_source_hashes = matrix.get("source_hashes", {})
    source_binding_current = bool(stored_source_hashes) and all(
        current_source_hashes.get(path) == expected
        for path, expected in stored_source_hashes.items()
    )
    checks = {
        "P0_EXECUTION_INFRASTRUCTURE_READY": (
            readiness.get("status") == "GO"
            and readiness.get("marker") == "P0_EXECUTION_INFRASTRUCTURE_READY"
        ),
        "ACCOUNT_STATE_READY": bool(sub_gates.get("ACCOUNT_STATE_READY")),
        "ACCOUNT_STATE_FRESH": bool(sub_gates.get("ACCOUNT_STATE_FRESH")),
        "ORDER_RECONCILIATION_READY": bool(
            sub_gates.get("ORDER_RECONCILIATION_READY")
        ),
        "POSITION_RECONCILIATION_READY": bool(
            sub_gates.get("POSITION_RECONCILIATION_READY")
        ),
        "CASH_RECONCILIATION_READY": bool(
            sub_gates.get("CASH_RECONCILIATION_READY")
        ),
        "IBKR_REGRESSION_MATRIX_STATUS": (
            matrix.get("status") == "GO"
            and bool(matrix_statuses)
            and all(value == "GO" for value in matrix_statuses.values())
        ),
        "P0_SOURCE_BINDING_CURRENT": source_binding_current,
    }
    return {
        "status": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "regression_scenario_count": len(matrix_statuses),
        "execution_authority": readiness.get("execution_authority", "NONE"),
        "orders_submitted": readiness.get("orders_submitted", 0),
        "orders_cancelled": readiness.get("orders_cancelled", 0),
        "orders_modified": readiness.get("orders_modified", 0),
        "fx_trades": readiness.get("fx_trades", 0),
    }


def contract_identity_audit(project_root: Path) -> dict[str, Any]:
    path = project_root / "output/ibkr/contracts/stocks.parquet"
    if not path.is_file():
        return {"status": "NO_GO", "blockers": ["CONTRACT_CACHE_MISSING"]}
    frame = pd.read_parquet(path)
    aliases = {
        "conId": ("con_id", "conId"),
        "symbol": ("symbol",),
        "localSymbol": ("local_symbol", "localSymbol"),
        "secType": ("security_type", "sec_type", "secType"),
        "exchange": ("exchange",),
        "primaryExchange": ("primary_exchange", "primaryExchange"),
        "currency": ("currency",),
        "tradingClass": ("trading_class", "tradingClass"),
        "multiplier": ("multiplier",),
        "expiry": ("expiry", "last_trade_date_or_contract_month"),
    }
    retained = {
        field: next((name for name in names if name in frame), None)
        for field, names in aliases.items()
    }
    con_column = retained["conId"]
    duplicate_conids = (
        int(frame[con_column].duplicated(False).sum()) if con_column else len(frame)
    )
    symbol_column = retained["symbol"]
    symbol_collisions = (
        int((frame.groupby(symbol_column).size() > 1).sum())
        if symbol_column
        else 0
    )
    durable_complete = bool(con_column) and bool(retained["currency"])
    return {
        "schema": "p1_contract_identity_recovery_audit_v1",
        "status": "GO" if durable_complete and duplicate_conids == 0 else "NO_GO",
        "contract_count": len(frame),
        "retained_fields": retained,
        "duplicate_conid_rows": duplicate_conids,
        "ticker_collision_count": symbol_collisions,
        "matching_by_ticker_only_allowed": False,
        "qualified_conid_is_durable_identity": True,
        "unqualified_mapping_fabricated": False,
        "research_without_contract_allowed": True,
        "execution_without_contract_allowed": False,
        "execution_authority": "NONE",
    }


def shariah_coverage_audit(project_root: Path) -> dict[str, Any]:
    path = project_root / "output/portfolio/coverage-waterfall.parquet"
    if not path.is_file():
        return {"status": "NO_GO", "blockers": ["COVERAGE_DETAIL_MISSING"]}
    frame = pd.read_parquet(path)
    groups = {}
    for name, selected in {
        "EQUITY": frame["asset_class"] == "EQUITY",
        "ETF": frame["asset_class"] == "ETF",
        "COMMODITY_PROXIES": frame["asset_class"] == "COMMODITY_EXPOSURE",
        "SMALL_MID_OR_UNKNOWN_CAP": frame["subclass"].isin(
            ["SMALL_CAP", "MID_CAP", "EQUITY_CAP_UNKNOWN"]
        ),
    }.items():
        subset = frame.loc[selected]
        counts = Counter(subset["shariah_status"])
        groups[name] = {
            "instrument_count": len(subset),
            "statuses": dict(sorted(counts.items())),
            "missing_or_review_ratio": round(
                sum(
                    counts.get(status, 0)
                    for status in ("SHARIAH_DATA_MISSING", "SHARIAH_REVIEW_REQUIRED")
                ) / len(subset),
                6,
            ) if len(subset) else 0.0,
        }
    return {
        "schema": "p1_shariah_coverage_audit_v1",
        "status": "GO",
        "groups": groups,
        "missing_is_blocked_in_analytics": False,
        "missing_is_fail_closed_for_execution": True,
        "restrictions_weakened": False,
        "execution_authority": "NONE",
    }


def commodity_family_status(
    project_root: Path, normalized: dict[str, Any]
) -> dict[str, Any]:
    universe = pd.read_parquet(project_root / "output/universe/instruments.parquet")
    detail = pd.read_parquet(project_root / "output/portfolio/coverage-waterfall.parquet")
    metadata = universe.set_index(universe["symbol"].astype(str).str.upper()).to_dict(
        orient="index"
    )
    ranked = normalized.get("combined_ranking", [])
    results: dict[str, Any] = {}
    for family, terms in COMMODITY_FAMILIES.items():
        symbols = {
            str(row.get("symbol") or "").upper()
            for row in universe.to_dict(orient="records")
            if _contains_terms(row, terms)
        }
        selected = detail.loc[detail["symbol"].isin(symbols)]
        ranked_rows = [
            row
            for row in ranked
            if str(row.get("symbol") or "").upper() in symbols
        ]
        results[family] = {
            "available_instruments": len(symbols),
            "symbols": sorted(symbols),
            "analyzable": int(selected["data_analyzable"].sum()),
            "strategy_capable": sum(
                row.get("validation_status")
                in {"STAGE0_FALSIFICATION_SURVIVOR", "EXACT_VALIDATION_REQUIRED"}
                for row in ranked_rows
            ),
            "ranked": len(ranked_rows),
            "portfolio_candidates": int(selected["portfolio_candidate"].sum()),
            "execution_candidates": int(
                selected["execution_preflight_eligible"].sum()
            ),
            "economic_structures": sorted(
                {
                    str(metadata.get(symbol, {}).get("product_structure") or "UNKNOWN")
                    for symbol in symbols
                }
            ),
        }
    return {
        "schema": "p1_commodity_real_asset_family_status_v1",
        "status": "GO" if any(row["ranked"] for row in results.values()) else "NO_GO",
        "families": results,
        "direct_futures_execution_required": False,
        "different_structures_treated_as_equivalent": False,
        "execution_authority": "NONE",
    }


def capital_size_coherence(project_root: Path) -> dict[str, Any]:
    model = load_shared_cost_model(project_root)
    capitals = (1000, 1870, 2500, 5000, 10000, 25000, 50000)
    prices = (5, 25, 100, 250, 500, 1000, 2000)
    scenarios = []
    for capital in capitals:
        feasible = 0
        rows = []
        for price in prices:
            maximum_position = Decimal(str(capital)) * Decimal("0.30")
            available = min(Decimal(str(capital)), maximum_position)
            risk_budget = Decimal(str(capital)) * Decimal("0.0035")
            result = whole_share_economics(
                desired_notional_eur=maximum_position,
                price_eur=Decimal(str(price)),
                risk_budget_eur=risk_budget,
                risk_per_share_eur=Decimal(str(price)) * Decimal("0.08"),
                available_cash_eur=available,
                expected_gross_return=Decimal("0.08"),
                currency="EUR",
                model=model,
            )
            feasible += result["execution_candidate_status"] == "EXECUTABLE_WHOLE_SHARE"
            rows.append({"price_eur": price, **result})
        target_count = max(1, min(10, int(capital / 1000)))
        scenarios.append(
            {
                "capital_eur": capital,
                "target_position_count": target_count,
                "feasible_price_points": feasible,
                "price_scenarios": rows,
            }
        )
    return {
        "schema": "p1_capital_size_coherence_v1",
        "status": "GO",
        "scenarios": scenarios,
        "twenty_tiny_positions_forced": False,
        "risk_first": True,
        "whole_shares": True,
        "cash_is_competing_allocation": True,
        "execution_authority": "NONE",
    }


def recovery_cohort(project_root: Path, *, per_sector: int = 35) -> dict[str, Any]:
    universe = pd.read_parquet(project_root / "output/universe/instruments.parquet")
    available = {
        path.stem.upper()
        for path in (
            project_root / "data/research/critical_trading/yfinance"
        ).glob("*.parquet")
    }
    work = universe.loc[
        universe["active_listing"].astype(bool)
        & universe["instrument_type"].eq("STOCK")
        & universe["currency"].eq("USD")
        & universe["primary_exchange"].isin(["NASDAQ", "NYSE", "NYSEMKT"])
    ].copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work = work.loc[
        ~work["symbol"].isin(available)
        & work["symbol"].str.fullmatch(r"[A-Z]{1,5}")
        & ~work["industry"].eq("Shell Companies")
        & ~work["name"].astype(str).str.contains(
            r"acquisition|warrant|rights|depositary preferred|units?(?:$|\s)",
            case=False,
            regex=True,
        )
    ]
    work["selection_key"] = work["symbol"].map(stable_hash)
    chosen = (
        work.sort_values(["sector", "selection_key", "symbol"])
        .groupby("sector", sort=True, group_keys=False)
        .head(per_sector)
    )
    curated_missing = universe.loc[
        universe["active_listing"].astype(bool)
        & universe["discovery_source"].eq("BROAD_MULTI_ASSET_V1")
        & ~universe["symbol"].astype(str).str.upper().isin(available),
        "symbol",
    ].astype(str).str.upper()
    symbols = sorted(set(chosen["symbol"]) | set(curated_missing))
    return {
        "schema": "p1_analyzable_coverage_recovery_cohort_v1",
        "status": "GO",
        "symbols": symbols,
        "symbol_count": len(symbols),
        "selection": (
            "ACTIVE_USD_COMMON_SYMBOLS_STRATIFIED_BY_SECTOR; SHELLS_UNITS_"
            "WARRANTS_RIGHTS_EXCLUDED; CURATED_MULTI_ASSET_MISSING_INCLUDED"
        ),
        "tradeable_assumed": False,
        "shariah_assumed": False,
        "broker_resolvable_assumed": False,
        "execution_authority": "NONE",
    }


def _completion_checks(**evidence: Any) -> dict[str, bool]:
    coverage = evidence["coverage"].get("funnels", {})
    normalized = evidence["normalized"]
    portfolio = evidence["portfolio"]
    monitoring = evidence["monitoring"]
    commodities = evidence["commodities"]
    current_analyzable = coverage.get("ALL", {}).get("stages", {}).get(
        "data_analyzable", 0
    )
    class_counts = normalized.get("asset_class_counts", {})
    return {
        "01_THREE_FIRST_CLASS_MONITORED_SOURCES": (
            monitoring.get("status") == "GO"
            and all(class_counts.get(name, 0) > 0 for name in (
                "EQUITY", "ETF", "COMMODITY_EXPOSURE"
            ))
        ),
        "02_COVERAGE_LOSSES_EXPLICIT": bool(
            evidence["coverage"].get("recoverable_categories")
        ),
        "03_RECOVERABLE_COVERAGE_MATERIALLY_IMPROVED": (
            current_analyzable >= int(BASELINE_ANALYZABLE * 1.25)
        ),
        "04_COMMODITY_ZERO_CANDIDATE_FIXED": commodities.get("status") == "GO",
        "05_RESEARCH_EXECUTION_SEPARATED": bool(
            evidence["coverage"].get("research_execution_separated")
        ),
        "06_VECTORIZED_FALSIFICATION": (
            portfolio["vectorized_stage0"].get("status") == "GO"
            and not portfolio["vectorized_stage0"].get("stage0_direct_promotion")
        ),
        "07_STANDARD_WALK_FORWARD": evidence["walk_forward"].get("status") == "GO",
        "08_SHARED_TRANSACTION_COSTS": (
            evidence["costs"].get("portfolio_model")
            == "SHARED_TRANSACTION_COST_MODEL_V1"
            and evidence["costs"].get("execution_preflight_model")
            == "SHARED_TRANSACTION_COST_MODEL_V1"
        ),
        "09_STRATEGY_OVERLAP_PROMOTION_GATE": bool(
            portfolio["overlap"].get("strategy_pnl_correlation_required_for_promotion")
        ),
        "10_CENTRAL_DESIRED_TARGET_LAYER": (
            portfolio["desired_targets"].get("status") == "GO"
            and not portfolio["desired_targets"].get("submits_orders")
        ),
        "11_CROSS_ASSET_AND_CASH_COMPETE": (
            class_counts.get("CASH", 0) == 1
            and all(
                class_counts.get(name, 0) > 0
                for name in ("EQUITY", "ETF", "COMMODITY_EXPOSURE")
            )
            and portfolio["target_allocation"].get(
                "research_cash_target_weight"
            ) is not None
        ),
        "12_NATIVE_WHOLE_SHARE_RISK_FIRST": bool(
            portfolio["desired_targets"].get("whole_share_risk_engine_authoritative")
        ),
        "13_CURRENT_POSITIONS_COMPETE": bool(
            monitoring.get("held_and_unheld_use_same_intelligence")
        ),
        "14_ROTATION_COST_AND_HYSTERESIS": bool(
            portfolio["desired_targets"].get("rotation_decisions") is not None
            and evidence["capital"].get("cash_is_competing_allocation")
        ),
        "15_CROSS_ASSET_CORRELATION_OVERLAP": portfolio["overlap"].get("status") == "GO",
        "16_SEPARATE_AND_COMBINED_FUNNELS": all(
            name in coverage for name in ("ALL", "EQUITY", "ETF", "COMMODITY_EXPOSURE")
        ),
        "17_NO_AUTHORITY_OR_RISK_INCREASE": (
            portfolio["engine_status"].get("execution_authority") == "NONE"
            and portfolio["engine_status"].get("orders_generated") == 0
        ),
        "18_P0_INVARIANTS_GREEN": evidence["p0"].get("status") == "GO",
    }


def _remaining_gaps(**evidence: Any) -> dict[str, Any]:
    shariah_groups = evidence["shariah"].get("groups", {})
    missing_shariah = sum(
        group.get("statuses", {}).get("SHARIAH_DATA_MISSING", 0)
        for group in shariah_groups.values()
        if group
    )
    return {
        "P1_COMPLETE": not evidence["unresolved"],
        "P1_PARTIAL": evidence["unresolved"],
        "P2": [
            "ASSET_CLASS_SPECIFIC_EXACT_MODEL_EXPANSION",
            "COMPLETE_ETF_PORTFOLIO_HOLDINGS_BEYOND_TOP_HOLDINGS",
            "CAMPAIGN_LEVEL_MULTI_HYPOTHESIS_FALSE_DISCOVERY_EXPANSION",
        ],
        "EXTERNAL_DATA_BLOCKER": [
            f"SHARIAH_DATA_MISSING_OR_REPEATED_GROUP_COUNT:{missing_shariah}",
            "OBSERVED_SLIPPAGE_CALIBRATION_PENDING"
            if evidence["costs"].get("matched_cost_observation_count", 0) == 0
            else "NONE",
        ],
        "IBKR_BLOCKER": [
            name
            for name, passed in evidence["p0"].get("checks", {}).items()
            if not passed
        ],
        "VALIDATION_BLOCKER": [
            "STAGE0_SURVIVORS_REQUIRE_EXACT_EVENT_DRIVEN_VALIDATION",
            *(
                ["STRATEGY_PNL_SERIES_REQUIRED_FOR_INCREMENTAL_PROMOTION"]
                if evidence["strategy"].get("strategy_pnl_series_count", 0)
                < evidence["strategy"].get("strategy_count", 0)
                else []
            ),
        ],
    }


def _contains_terms(row: dict[str, Any], terms: tuple[str, ...]) -> bool:
    text = " ".join(
        str(row.get(name) or "").upper()
        for name in (
            "underlying_commodity", "sector", "industry", "category",
            "exposure_type", "commodity_exposure_type", "name", "symbol",
        )
    )
    return any(term in text for term in terms)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


__all__ = [
    "capital_size_coherence",
    "commodity_family_status",
    "contract_identity_audit",
    "recovery_cohort",
    "run_p1_readiness",
    "shariah_coverage_audit",
    "verify_p0",
]
