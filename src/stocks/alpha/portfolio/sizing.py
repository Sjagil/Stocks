from __future__ import annotations


def size_position(
    *,
    alpha_confidence: float,
    regime_multiplier: float,
    volatility: float,
    maximum_weight: float = 0.10,
    base_weight: float = 0.08,
) -> float:
    volatility_penalty = min(1.0, 0.20 / max(volatility, 0.01))
    raw = base_weight * alpha_confidence * regime_multiplier * volatility_penalty
    return round(max(0.0, min(maximum_weight, raw)), 6)


def capital_preservation_rotation(scores: dict[str, float], hedge_budget: float) -> dict[str, float]:
    eligible = [(name, score) for name, score in scores.items() if score >= 0.55]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if not eligible:
        return {"OPERATIONAL_CASH": min(0.10, hedge_budget)}
    selected = eligible[:2]
    if len(selected) == 1:
        return {selected[0][0]: round(hedge_budget, 6)}
    return {
        selected[0][0]: round(hedge_budget * 0.60, 6),
        selected[1][0]: round(hedge_budget * 0.40, 6),
    }
