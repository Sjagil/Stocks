from __future__ import annotations

from typing import Any, Mapping, Sequence


REVERSAL_TOKENS = (
    "contrarian",
    "failed_breakdown",
    "mean_reversion",
    "oversold",
    "pullback_reversal",
    "rsi2",
)


def opportunity_class(
    metadata: Mapping[str, Any],
    strategy_families: Sequence[str],
) -> str:
    exposure = str(
        metadata.get("commodity_exposure_type") or "NONE"
    ).upper()
    if (
        str(metadata.get("sleeve")) == "commodity_security"
        or exposure not in {"", "NONE", "UNSPECIFIED"}
    ):
        return "REAL_ASSET"
    normalized = " ".join(strategy_families).lower()
    if any(token in normalized for token in REVERSAL_TOKENS):
        return "REVERSAL"
    return "TREND"


def active_swing_timeframe_context(
    timeframes: Sequence[str],
) -> dict[str, Any]:
    weights = {
        "1w": 0.05,
        "1d": 0.15,
        "4h": 0.30,
        "2h": 0.20,
        "1h": 0.30,
    }
    observed = sorted({str(value).lower() for value in timeframes})
    score = min(sum(weights.get(value, 0.0) for value in observed), 1.0)
    if {"1h", "4h"}.issubset(observed):
        alignment = "ACTIVE_SWING_ALIGNED"
        higher_timeframe_risk_multiplier = 1.0 if "1d" in observed else 0.9
    elif "1h" in observed or "4h" in observed:
        alignment = "ACTIVE_SWING_PARTIAL"
        higher_timeframe_risk_multiplier = 0.75
    else:
        alignment = "HIGHER_TIMEFRAME_CONTEXT_ONLY"
        higher_timeframe_risk_multiplier = 0.65
    return {
        "schema": "active_swing_timeframe_context_v1",
        "score": round(score, 6),
        "classification": alignment,
        "information_weights": weights,
        "higher_timeframe_policy": "MULTIPLIER_NOT_AUTOMATIC_VETO",
        "higher_timeframe_risk_multiplier": (
            higher_timeframe_risk_multiplier
        ),
        "timing_15m_status": (
            "UNAVAILABLE_RESEARCH_CONTRACT_FORBIDS_15M"
        ),
        "standalone_entry_allowed": False,
    }


def real_asset_context(
    metadata: Mapping[str, Any],
    components: Mapping[str, float],
) -> dict[str, Any]:
    exposure = str(
        metadata.get("commodity_exposure_type") or "NONE"
    ).upper()
    if (
        str(metadata.get("sleeve")) != "commodity_security"
        and exposure in {"", "NONE", "UNSPECIFIED"}
    ):
        return {
            "schema": "real_asset_context_v1",
            "status": "NOT_APPLICABLE",
            "standalone_entry_allowed": False,
        }
    structure = str(
        metadata.get("product_structure") or "UNCLASSIFIED"
    ).upper()
    structure_quality = {
        "PHYSICAL_BACKED_GRANTOR_TRUST": 0.85,
        "PHYSICAL_CLOSED_END_TRUST": 0.80,
        "FUTURES_COMMODITY_POOL": 0.65,
        "PRODUCER_EQUITY_FUND": 0.75,
        "RESOURCE_EQUITY_FUND": 0.70,
        "OPERATING_COMPANY_EQUITY": 0.65,
    }.get(structure, 0.50)
    trend = _bounded(
        0.35 * float(components.get("signal_quality", 0.0))
        + 0.25 * float(components.get("timeframe_confirmation", 0.0))
        + 0.20 * float(components.get("relative_strength", 0.0))
        + 0.20 * float(components.get("setup_quality", 0.0))
    )
    macro = _bounded(float(components.get("regime_fit", 0.5)))
    liquidity = _bounded(float(components.get("liquidity", 0.0)))
    score = (
        0.40 * trend
        + 0.20 * macro
        + 0.20 * liquidity
        + 0.20 * structure_quality
    )
    return {
        "schema": "real_asset_context_v1",
        "status": "PARTIAL_AVAILABLE_COMPONENTS_ONLY",
        "score": round(score, 6),
        "underlying_commodity": str(
            metadata.get("underlying_commodity") or "UNKNOWN"
        ),
        "commodity_exposure_type": exposure,
        "product_structure": structure,
        "components": {
            "trend": round(trend, 6),
            "macro": round(macro, 6),
            "liquidity": round(liquidity, 6),
            "structure_quality": round(structure_quality, 6),
        },
        "unavailable_components": {
            "carry": "UNAVAILABLE_NO_CAUSAL_TERM_STRUCTURE_INPUT",
            "inventory": "UNAVAILABLE_NO_CAUSAL_INVENTORY_INPUT",
            "nav_premium_discount": (
                "UNAVAILABLE_NO_POINT_IN_TIME_NAV_INPUT"
            ),
            "producer_margin": (
                "UNAVAILABLE_NO_NORMALIZED_PRODUCER_MARGIN_INPUT"
            ),
        },
        "ranking_influence": "OBSERVATION_ONLY_PENDING_ABLATION",
        "standalone_entry_allowed": False,
        "execution_authority": "NONE",
    }


def _bounded(value: float) -> float:
    return min(max(value, 0.0), 1.0)
