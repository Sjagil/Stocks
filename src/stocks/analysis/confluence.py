from __future__ import annotations

import math
from typing import Any


DEFAULT_WEIGHTS = {
    "technical": 0.50,
    "fundamental": 0.30,
    "macro": 0.20,
}
DEFAULT_THRESHOLDS = {
    "supportive": 0.60,
    "adverse": 0.40,
    "minimum_fundamental": 0.35,
    "severe_macro_headwind": 0.30,
    "minimum_macro_confidence_for_headwind": 0.50,
}


def evaluate_multilayer_confluence(
    *,
    technical_score: float,
    fundamental_score: float | None,
    fundamental_required: bool,
    macro_score: float | None,
    macro_confidence: float,
    macro_status: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = config or {}
    weights = {**DEFAULT_WEIGHTS, **policy.get("weights", {})}
    thresholds = {
        **DEFAULT_THRESHOLDS,
        **policy.get("thresholds", {}),
    }
    technical = _clamp(technical_score)
    fundamental_available = fundamental_score is not None
    fundamental = (
        _clamp(fundamental_score)
        if fundamental_available
        else None
    )
    normalized_macro_status = str(macro_status or "UNAVAILABLE").upper()
    macro_available = (
        macro_score is not None
        and normalized_macro_status
        in {"AVAILABLE", "GENERIC_REGIME_FALLBACK"}
        and _clamp(macro_confidence) > 0
    )
    macro = _clamp(macro_score) if macro_available else None
    missing_layers: list[str] = []
    if fundamental_required and not fundamental_available:
        missing_layers.append("FUNDAMENTAL")
    if not macro_available:
        missing_layers.append("MACRO")

    layer_scores: dict[str, float] = {"technical": technical}
    active_weights: dict[str, float] = {
        "technical": float(weights["technical"])
    }
    if fundamental is not None:
        layer_scores["fundamental"] = fundamental
        active_weights["fundamental"] = float(weights["fundamental"])
    elif not fundamental_required:
        layer_scores["fundamental"] = 0.50
        active_weights["fundamental"] = float(weights["fundamental"])
    if macro is not None:
        layer_scores["macro"] = macro
        active_weights["macro"] = float(weights["macro"])

    confluence_score = _weighted_geometric_mean(
        layer_scores, active_weights
    )
    blockers: list[str] = []
    if "FUNDAMENTAL" in missing_layers:
        blockers.append("MULTILAYER_FUNDAMENTAL_DATA_REQUIRED")
    if "MACRO" in missing_layers:
        blockers.append("MULTILAYER_MACRO_CONTEXT_REQUIRED")
    if technical < float(thresholds["adverse"]):
        blockers.append("MULTILAYER_TECHNICAL_SETUP_ADVERSE")
    if (
        fundamental_required
        and fundamental is not None
        and fundamental < float(thresholds["minimum_fundamental"])
    ):
        blockers.append("MULTILAYER_FUNDAMENTAL_QUALITY_ADVERSE")

    severe_macro_headwind = bool(
        macro is not None
        and macro < float(thresholds["severe_macro_headwind"])
        and _clamp(macro_confidence)
        >= float(thresholds["minimum_macro_confidence_for_headwind"])
    )
    supportive = all(
        score >= float(thresholds["supportive"])
        for name, score in layer_scores.items()
        if name != "fundamental" or fundamental_required
    )
    if missing_layers:
        status = "BLOCKED_MISSING_REQUIRED_LAYER"
        multiplier = 0.75
    elif blockers:
        status = "BLOCKED_ADVERSE_REQUIRED_LAYER"
        multiplier = 0.80
    elif severe_macro_headwind:
        status = "MACRO_HEADWIND_RISK_REDUCTION"
        multiplier = 0.85
    elif supportive:
        status = "THREE_LAYER_CONFIRMED"
        multiplier = 1.05
    else:
        status = "MIXED_CONFLUENCE"
        multiplier = 1.00

    return {
        "schema": "technical_fundamental_macro_confluence_v1",
        "status": status,
        "confluence_score": round(confluence_score, 8),
        "ranking_multiplier": multiplier,
        "allocation_allowed": not blockers and not missing_layers,
        "layers": {
            "technical": {
                "status": _layer_status(technical, thresholds),
                "score": round(technical, 8),
                "required": True,
                "standalone_entry_allowed": False,
            },
            "fundamental": {
                "status": (
                    _layer_status(float(fundamental), thresholds)
                    if fundamental is not None
                    else "NOT_REQUIRED"
                    if not fundamental_required
                    else "UNAVAILABLE"
                ),
                "score": (
                    None if fundamental is None else round(fundamental, 8)
                ),
                "required": fundamental_required,
                "standalone_entry_allowed": False,
            },
            "macro": {
                "status": (
                    _layer_status(float(macro), thresholds)
                    if macro is not None
                    else "UNAVAILABLE"
                ),
                "source_status": normalized_macro_status,
                "score": None if macro is None else round(macro, 8),
                "confidence": round(_clamp(macro_confidence), 8),
                "severe_headwind": severe_macro_headwind,
                "required": True,
                "standalone_entry_allowed": False,
            },
        },
        "missing_required_layers": missing_layers,
        "allocation_blockers": blockers,
        "combination_method": "WEIGHTED_GEOMETRIC_MEAN",
        "weights": {
            name: round(float(weight), 8)
            for name, weight in active_weights.items()
        },
        "technical_signal_required": True,
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }


def _weighted_geometric_mean(
    scores: dict[str, float], weights: dict[str, float]
) -> float:
    denominator = sum(max(0.0, weights.get(name, 0.0)) for name in scores)
    if denominator <= 0:
        return 0.0
    log_sum = sum(
        max(0.0, weights.get(name, 0.0))
        * math.log(max(0.01, _clamp(score)))
        for name, score in scores.items()
    )
    return _clamp(math.exp(log_sum / denominator))


def _layer_status(score: float, thresholds: dict[str, Any]) -> str:
    if score >= float(thresholds["supportive"]):
        return "SUPPORTIVE"
    if score < float(thresholds["adverse"]):
        return "ADVERSE"
    return "NEUTRAL_OR_MIXED"


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(max(number, 0.0), 1.0)


__all__ = ["evaluate_multilayer_confluence"]
