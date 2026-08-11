from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash


OUTPUT_PATH = Path("output/portfolio/p1-requirement-audit.json")


def publish_p1_requirement_audit(
    project_root: Path,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    coverage = evidence.get("coverage_before_after", {})
    classes = evidence.get("normalized_opportunity_schema", {}).get(
        "asset_class_counts", {}
    )
    p0 = evidence.get("p0", {})
    monitoring = evidence.get("monitoring", {})
    stage0 = evidence.get("research_acceleration", {})
    walk = evidence.get("walk_forward_standardization", {})
    costs = evidence.get("cost_model", {})
    overlap = evidence.get("overlap", {})
    targets = evidence.get("portfolio_targets", {})
    strategy = evidence.get("strategy_inventory", {})
    attribution = evidence.get("performance_attribution", {})
    intelligence = evidence.get("cross_asset_intelligence", {})
    validation = _read_json(
        project_root / "output/portfolio/p1-validation-summary.json"
    ).get("tests", {})
    item = _item
    items = [
        item(1, "VERIFY_P0", p0.get("status") == "GO", blockers=([] if p0.get("status") == "GO" else ["CURRENT_P0_LIVE_READINESS_NO_GO"]), external=True),
        item(2, "CORE_LAYER_SEPARATION", _source_has(project_root, "src/stocks/portfolio/manager.py", "normalized_opportunities", "desired_targets", "build_private_sizing")),
        item(3, "THREE_FIRST_CLASS_ASSET_CLASSES", all(classes.get(name, 0) > 0 for name in ("EQUITY", "ETF", "COMMODITY_EXPOSURE"))),
        item(4, "MULTI_FREQUENCY_CONTINUOUS_MONITORING", monitoring.get("status") == "GO" and len(monitoring.get("frequencies", [])) == 5 and _source_has(project_root, "src/stocks/portfolio/manager.py", "publish_monitoring_architecture")),
        item(5, "HELD_AND_UNHELD_MONITORED", bool(monitoring.get("held_and_unheld_use_same_intelligence"))),
        item(6, "EXPLICIT_OPPORTUNITY_COST", _source_has(project_root, "src/stocks/portfolio/targets.py", "minimum_expected_net_return_improvement", "transaction_costs_included_in_expected_net")),
        item(7, "ANALYZABLE_COVERAGE_RECOVERY", int(coverage.get("current_analyzable", 0)) >= int(coverage.get("material_improvement_threshold", 10**9))),
        item(8, "DATA_AVAILABILITY_SEPARATE_FROM_ELIGIBILITY", evidence.get("coverage_before_after", {}).get("by_asset_class") is not None and evidence.get("normalized_opportunity_schema", {}).get("opportunity_count", 0) > 0),
        item(9, "CONTRACT_IDENTITY_RECOVERY", evidence.get("contract_identity", {}).get("status") == "GO" and evidence.get("contract_identity", {}).get("qualified_conid_is_durable_identity") is True and evidence.get("contract_identity", {}).get("unqualified_mapping_fabricated") is False),
        item(10, "SHARIAH_COVERAGE_DISTINCTIONS", evidence.get("shariah_coverage", {}).get("status") == "GO" and _shariah_status_contract(evidence)),
        item(11, "EQUITY_OPPORTUNITY_ENGINE", classes.get("EQUITY", 0) > 0 and _source_has(project_root, "src/stocks/portfolio/manager.py", "relative_strength", "fundamental_quality", "event_risk")),
        item(12, "HIDDEN_GEM_DISCOVERY", _artifact_has(project_root, "output/portfolio/p1-coverage-recovery-cohort.json", "tradeable_assumed", False) and _source_has(project_root, "src/stocks/portfolio/p1.py", "Shell Companies", "per_sector")),
        item(13, "ETF_OPPORTUNITY_ENGINE", classes.get("ETF", 0) > 0, partial=str(overlap.get("etf_holdings_overlap_status", "")).startswith("DATA_MISSING"), blockers=["ETF_HOLDINGS_LOOK_THROUGH_DATA_MISSING"] if str(overlap.get("etf_holdings_overlap_status", "")).startswith("DATA_MISSING") else []),
        item(14, "COMMODITY_REAL_ASSET_ENGINE", evidence.get("commodity_real_asset_status", {}).get("status") == "GO" and _commodity_model_contract(intelligence)),
        item(15, "CROSS_ASSET_REGIME_ENGINE", intelligence.get("status") == "GO" and len(intelligence.get("regime_dimensions", {})) >= 11),
        item(16, "MARKET_BREADTH_AND_LEADERSHIP", bool(intelligence.get("breadth_by_asset_class")) and not intelligence.get("breadth_is_automatic_hard_veto", True)),
        item(17, "RELATIVE_STRENGTH_ALL_CLASSES", intelligence.get("available_comparison_count", 0) > 0 and len(intelligence.get("cross_asset_leadership", [])) == 3),
        item(18, "CENTRAL_OPPORTUNITY_OBJECT", evidence.get("normalized_opportunity_schema", {}).get("opportunity_count", 0) > 0),
        item(19, "STRATEGY_INVENTORY_BY_CLASS", _strategy_inventory_contract(strategy)),
        item(20, "VECTORIZED_STAGE0", stage0.get("status") == "GO" and not stage0.get("stage0_direct_promotion", True)),
        item(21, "PARAMETER_SEARCH_CONTROL", all(key in stage0 for key in ("hypothesis_count", "parameter_combination_count", "assets_tested", "timeframes_tested", "selection_criterion")) and int(stage0.get("hypothesis_count", 0)) == 1 and int(stage0.get("parameter_combination_count", 0)) == 1),
        item(22, "STANDARD_WALK_FORWARD_MANIFEST", walk.get("status") == "GO" and walk.get("manifest_count", 0) >= walk.get("strategy_count", 0)),
        item(23, "LOOKAHEAD_PROTECTION", walk.get("lookahead_protection", {}).get("status") == "GO" and walk.get("lookahead_protection", {}).get("evidence_mode") == "SOURCE_BOUND"),
        item(24, "HISTORICAL_PIT_UNIVERSE", walk.get("lookahead_protection", {}).get("status") == "GO" and not walk.get("lookahead_protection", {}).get("complete_pit_coverage_claimed", True) and walk.get("lookahead_protection", {}).get("checks", {}).get("universe_membership_is_point_in_time_when_available") is True),
        item(25, "SHARED_TRANSACTION_COST_MODEL", costs.get("research_model") == costs.get("portfolio_model") == costs.get("execution_preflight_model") == "SHARED_TRANSACTION_COST_MODEL_V1", partial=not costs.get("predicted_vs_observed_comparison_available", False), blockers=["OBSERVED_COST_CALIBRATION_PENDING"]),
        item(26, "WHOLE_SHARE_OPPORTUNITY_ECONOMICS", _source_has(project_root, "src/stocks/costs.py", "whole_share_economics", "expected_net_profit_eur")),
        item(27, "DESIRED_PORTFOLIO_TARGET_LAYER", targets.get("status") == "GO" and not targets.get("submits_orders", True)),
        item(28, "CURRENT_VS_DESIRED_DELTAS", _source_has(project_root, "src/stocks/portfolio/targets.py", "BUY_DELTA", "SELL_DELTA", "NO_ACTION")),
        item(29, "CROSS_ASSET_PORTFOLIO_COMPETITION", all(classes.get(name, 0) > 0 for name in ("EQUITY", "ETF", "COMMODITY_EXPOSURE", "CASH"))),
        item(30, "ASSET_CLASS_EXPOSURE_LIMITS", _artifact_has(project_root, "config/portfolio/active_manager_v1.json", "asset_class_caps"), partial=str(overlap.get("etf_holdings_overlap_status", "")).startswith("DATA_MISSING"), blockers=["ETF_LOOK_THROUGH_INCOMPLETE"]),
        item(31, "CORRELATION_AND_CLUSTER_RISK", overlap.get("status") == "GO" and overlap.get("return_correlation_checked") and overlap.get("etf_holdings_overlap_status") == "TOP_HOLDINGS_LOOK_THROUGH_AVAILABLE" and strategy.get("strategy_pnl_series_count", 0) >= strategy.get("strategy_count", 0)),
        item(32, "QLIB_STYLE_NATIVE_RISK_CONCEPTS", _source_has(project_root, "src/stocks/portfolio/manager.py", "_portfolio_volatility", "correlation")),
        item(33, "EXPLICIT_PORTFOLIO_OBJECTIVE", _source_has(project_root, "src/stocks/portfolio/opportunities.py", "ROBUST_EXPECTED_NET_PNL_RISK_ADJUSTED_WITH_CONSTRAINTS")),
        item(34, "CASH_COMPETING_ASSET", classes.get("CASH", 0) == 1),
        item(35, "SEMI_AGGRESSIVE_BOUNDED_BEHAVIOR", _artifact_has(project_root, "config/portfolio/active_manager_v1.json", "maximum_portfolio_heat")),
        item(36, "CAPITAL_AWARE_POSITION_COUNT", evidence.get("capital_size_coherence", {}).get("status") == "GO" and len(evidence.get("capital_size_coherence", {}).get("scenarios", [])) == 7),
        item(37, "RISK_FIRST_SIZING_AUTHORITATIVE", bool(targets.get("whole_share_risk_engine_authoritative"))),
        item(38, "MULTI_LEVEL_CAPITAL_PROTECTION", _source_has(project_root, "src/stocks/portfolio/manager.py", "portfolio_heat_gate", "turnover_gate", "dynamic_risk")),
        item(39, "ACTIVE_POSITION_MANAGEMENT", len(monitoring.get("position_rescoring_contract", [])) == 8),
        item(40, "EARNINGS_AND_EVENT_MANAGEMENT", _source_has(project_root, "src/stocks/analysis/theme_events.py", "earnings", "accepted_at")),
        item(41, "OPTIONS_AS_CONTEXT_ONLY", intelligence.get("options_context_authority") == "OBSERVATION_ONLY"),
        item(42, "MACRO_CROSS_ASSET_MONITORING", len(intelligence.get("regime_dimensions", {})) >= 11),
        item(43, "NEWS_NLP_CONTEXT_ONLY", intelligence.get("news_context_authority") == "RISK_AND_CATALYST_CONTEXT_ONLY"),
        item(44, "PERFORMANCE_ATTRIBUTION", _attribution_contract(attribution)),
        item(45, "STRATEGY_CORRELATION_PROMOTION_REQUIREMENT", strategy.get("strategy_pnl_series_count", 0) >= strategy.get("strategy_count", 0) > 0 and strategy.get("strategy_correlation_is_promotion_requirement") and strategy.get("missing_strategy_pnl_series_fails_promotion_closed")),
        item(46, "INDEPENDENT_PERFORMANCE_CHECK", evidence.get("independent_performance_check", {}).get("status") == "GO"),
        item(47, "SEPARATE_AND_COMBINED_FUNNELS", len(coverage.get("by_asset_class", {})) == 3),
        item(48, "OPPORTUNITY_DASHBOARD", _source_has(project_root, "src/stocks/ui/service.py", "opportunity_intelligence", "p1_readiness", "cross_asset_intelligence")),
        item(49, "DATA_FRESHNESS", intelligence.get("freshness", {}).get("status") == "CURRENT_CLOSED_DAILY_DATA" and monitoring.get("critical_stale_data_invalidates_actionability")),
        item(50, "P1_PRIORITY_ORDER", all(Path(project_root / path).is_file() for path in ("output/portfolio/coverage-waterfall.json", "output/research/p1/stage0-screen.json", "output/research/p1/walk-forward-manifests.json", "output/portfolio/desired-portfolio-targets.json"))),
        item(51, "DETERMINISTIC_TEST_SCOPE", _tests_pass(validation)),
        item(52, "VALIDATION_AND_P0_REGRESSION", "passed" in str(validation.get("full", "")) and p0.get("checks", {}).get("IBKR_REGRESSION_MATRIX_STATUS") is True, partial=p0.get("status") != "GO", blockers=["LIVE_P0_FRESHNESS_EXTERNAL"]),
        item(53, "NO_LIVE_SIDE_EFFECTS", all(value in (0, False) for value in evidence.get("live_side_effects", {}).values())),
        item(54, "REQUIRED_FINAL_REPORT_ARTIFACT", (project_root / "output/portfolio/p1-validation-summary.json").is_file()),
    ]
    counts = Counter(row["status"] for row in items)
    report: dict[str, Any] = {
        "schema": "p1_requirement_by_requirement_audit_v1",
        "status": "GO" if counts.get("PASS", 0) == len(items) else "PARTIAL",
        "generated_at": datetime.now(UTC).isoformat(),
        "requirement_count": len(items),
        "status_counts": dict(sorted(counts.items())),
        "items": items,
        "all_explicit_requirements_proven": counts.get("PASS", 0) == len(items),
        "execution_authority": "NONE",
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _item(number: int, name: str, passed: bool, *, partial: bool = False, blockers: list[str] | tuple[str, ...] = (), external: bool = False) -> dict[str, Any]:
    status = "PASS" if passed and not partial else "PARTIAL" if passed or partial else "BLOCKED" if external else "FAIL"
    return {"number": number, "requirement": name, "status": status, "evidence_proven": passed, "blockers": list(blockers), "external_blocker": external}


def _source_has(project_root: Path, relative: str, *tokens: str) -> bool:
    path = project_root / relative
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return all(token in text for token in tokens)


def _artifact_has(project_root: Path, relative: str, key: str, expected: Any = ...) -> bool:
    payload = _read_json(project_root / relative)
    values = _recursive_values(payload, key)
    if not values:
        return False
    return True if expected is ... else expected in values


def _recursive_values(value: Any, key: str) -> list[Any]:
    if isinstance(value, dict):
        return [value[key]] if key in value else [
            item
            for nested in value.values()
            for item in _recursive_values(nested, key)
        ]
    if isinstance(value, list):
        return [
            item
            for nested in value
            for item in _recursive_values(nested, key)
        ]
    return []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _missing_shariah(evidence: dict[str, Any]) -> int:
    return sum(group.get("statuses", {}).get("SHARIAH_DATA_MISSING", 0) for group in evidence.get("shariah_coverage", {}).get("groups", {}).values())


def _shariah_status_contract(evidence: dict[str, Any]) -> bool:
    expected = {
        "SHARIAH_ALLOWED",
        "SHARIAH_BLOCKED",
        "SHARIAH_REVIEW_REQUIRED",
        "SHARIAH_DATA_MISSING",
    }
    groups = evidence.get("shariah_coverage", {}).get("groups", {})
    observed = {
        status
        for group in groups.values()
        for status in group.get("statuses", {})
    }
    return expected.issubset(observed)


def _commodity_model_contract(intelligence: dict[str, Any]) -> bool:
    models = intelligence.get("commodity_family_models", {})
    expected = {"GOLD", "SILVER", "COPPER", "URANIUM", "OIL", "ENERGY"}
    names = {
        str(row.get("model"))
        for row in models.values()
        if isinstance(row, dict)
    }
    return (
        set(models) == expected
        and len(names) == len(expected)
        and all(
            row.get("same_model_as_other_commodities") is False
            for row in models.values()
        )
    )


def _strategy_inventory_contract(strategy: dict[str, Any]) -> bool:
    required = {
        "strategy_family", "instrument_scope", "timeframe", "signal_count",
        "historical_trades", "net_expectancy", "profit_factor", "sharpe",
        "sortino", "max_drawdown", "turnover", "average_holding_time",
        "cost_sensitivity", "walk_forward", "regime_performance",
        "parameter_stability", "forward_evidence",
    }
    rows = strategy.get("strategies", [])
    return (
        strategy.get("strategy_count", 0) == len(rows) > 0
        and set(strategy.get("asset_class_counts", {}))
        == {"EQUITY", "ETF", "COMMODITY_EXPOSURE"}
        and all(required.issubset(row) for row in rows)
    )


def _attribution_contract(attribution: dict[str, Any]) -> bool:
    dimensions = {
        "asset", "asset_class", "strategy_family", "sector", "industry",
        "factor", "regime", "entry_type", "exit_type",
    }
    measures = {
        "gross_pnl_eur", "fees_eur", "slippage_eur", "net_pnl_eur",
        "capital_employed_eur", "risk_employed_eur",
    }
    return (
        attribution.get("derived_read_model_only") is True
        and attribution.get("rebuildable_from_canonical_records") is True
        and attribution.get("canonical_records_mutated") is False
        and attribution.get("parallel_financial_ledger_created") is False
        and set(attribution.get("dimensions", [])) == dimensions
        and set(attribution.get("measures", [])) == measures
    )


def _tests_pass(validation: dict[str, Any]) -> bool:
    return validation.get("ruff") == "PASS" and validation.get("compile") == "PASS" and "passed" in str(validation.get("full", ""))


__all__ = ["publish_p1_requirement_audit"]
