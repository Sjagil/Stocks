from __future__ import annotations

from collections import Counter
from typing import Any

from stocks.research.autopilot.components import component_registry
from stocks.research.autopilot.contracts import (
    ALLOWED_SWING_TIMEFRAMES,
    FORBIDDEN_TIMEFRAMES,
    StrategyFamily,
    stable_hash,
)


CANONICAL_SECTIONS = (
    "technical_indicators",
    "momentum_indicators",
    "mean_reversion_indicators",
    "volatility_indicators",
    "volume_and_liquidity_indicators",
    "market_structure_and_price_action",
    "relative_strength_indicators",
    "breadth_indicators",
    "statistical_and_quantitative_indicators",
    "stock_fundamentals",
    "etf_fundamentals",
    "commodity_fundamentals",
    "macro_indicators",
    "stock_strategies",
    "etf_strategies",
    "commodity_and_commodity_etf_strategies",
    "multi_factor_strategies",
    "entries",
    "exits",
    "position_sizing",
    "portfolio_indicators",
    "version_one_core",
    "priority_strategy_families",
    "component_architecture",
)

PRIORITY_STRATEGY_FAMILIES = (
    "quality_momentum_stocks",
    "trend_pullback_stocks",
    "breakout_with_volume_confirmation",
    "volatility_contraction_breakout",
    "etf_relative_strength_rotation",
    "sector_rotation",
    "regional_etf_rotation",
    "quality_value_trend",
    "earnings_momentum",
    "dividend_quality",
    "commodity_etf_trend",
    "gold_defensive_sleeve",
    "macro_filtered_exposure",
    "breadth_filtered_market_regime",
    "multi_timeframe_trend",
    "confirmed_fractal_breakout_retest",
    "regime_filtered_mean_reversion",
    "risk_adjusted_momentum",
    "hierarchical_sector_to_stock_selection",
    "equal_weight_benchmark_portfolio",
)

FAMILY_IMPLEMENTATION_MAP = {
    "quality_momentum_stocks": StrategyFamily.QUALITY_MOMENTUM.value,
    "trend_pullback_stocks": StrategyFamily.TREND_PULLBACK.value,
    "volatility_contraction_breakout": (
        StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT.value
    ),
    "etf_relative_strength_rotation": StrategyFamily.ETF_ROTATION.value,
    "commodity_etf_trend": StrategyFamily.COMMODITY_ETF_TREND.value,
}

REQUIRED_COMPONENT_METADATA = (
    "name",
    "version",
    "category",
    "formula",
    "required_fields",
    "supported_assets",
    "supported_timeframes",
    "lookback",
    "warmup",
    "available_at_rules",
    "missing_data_policy",
    "causality_status",
    "unit",
    "output_range",
    "test_status",
)

REQUIRED_STRATEGY_BLOCKS = (
    "universe_filter",
    "fundamental_eligibility",
    "ranking",
    "entry",
    "confirmation",
    "regime_filter",
    "position_sizing",
    "exit",
    "portfolio_constraints",
    "cost_model",
    "benchmark",
)


def taxonomy_coverage_report() -> dict[str, Any]:
    registry = component_registry()
    categories = Counter(item.category for item in registry.values())
    rows = []
    metadata_failures: dict[str, list[str]] = {}
    for item in sorted(registry.values(), key=lambda value: value.name):
        canonical = {
            "name": item.name,
            "version": item.version,
            "category": item.category,
            "formula": item.formula,
            "required_fields": list(item.required_columns),
            "supported_assets": list(item.asset_compatibility),
            "supported_timeframes": list(item.supported_timeframes),
            "lookback": item.lookback,
            "warmup": item.warmup,
            "available_at_rules": item.causality_rule,
            "missing_data_policy": item.missing_data_policy,
            "causality_status": item.causality_status,
            "unit": item.unit,
            "output_range": item.output_range,
            "test_status": item.test_status,
        }
        missing = [
            field
            for field in REQUIRED_COMPONENT_METADATA
            if canonical.get(field) is None
            or canonical.get(field) == ""
            or canonical.get(field) == []
            or canonical.get(field) == ()
        ]
        if missing:
            metadata_failures[item.name] = missing
        rows.append(canonical)
    strategy_family_coverage = [
        {
            "canonical_family": family,
            "implementation": FAMILY_IMPLEMENTATION_MAP.get(family),
            "status": (
                "IMPLEMENTED"
                if family in FAMILY_IMPLEMENTATION_MAP
                else "REGISTERED_RESEARCH_BACKLOG"
            ),
            "automatic_authority": False,
        }
        for family in PRIORITY_STRATEGY_FAMILIES
    ]
    payload = {
        "schema": "canonical_strategy_taxonomy_coverage_v1",
        "status": "GO" if not metadata_failures else "BLOCKED",
        "source_contract": "CANONICAL_SWING_RESEARCH_TAXONOMY_2026_07",
        "scope": {
            "assets": ["STOCK", "ETF", "COMMODITY_ETF", "ETC"],
            "long_only": True,
            "minimum_timeframe": "1h",
            "allowed_timeframes": list(ALLOWED_SWING_TIMEFRAMES),
            "forbidden_timeframes": sorted(FORBIDDEN_TIMEFRAMES),
            "ai_or_machine_learning": False,
        },
        "canonical_sections": list(CANONICAL_SECTIONS),
        "canonical_section_count": len(CANONICAL_SECTIONS),
        "component_count": len(rows),
        "component_category_counts": dict(sorted(categories.items())),
        "required_component_metadata": list(REQUIRED_COMPONENT_METADATA),
        "component_metadata_failures": metadata_failures,
        "components": rows,
        "required_strategy_blocks": list(REQUIRED_STRATEGY_BLOCKS),
        "priority_strategy_family_count": len(PRIORITY_STRATEGY_FAMILIES),
        "priority_strategy_family_coverage": strategy_family_coverage,
        "implemented_priority_family_count": len(FAMILY_IMPLEMENTATION_MAP),
        "backlog_is_financial_evidence": False,
        "equal_weight_benchmark_required": True,
        "cost_model_required": True,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
    }
    payload["taxonomy_hash"] = stable_hash(payload)
    return payload
