from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.macro.config import MacroConfig
from stocks.macro.contracts import MacroScore, RegimeLabel, ScoreStatus, stable_hash
from stocks.macro.transforms import build_feature_snapshot


CORE_SCORES = (
    "growth",
    "inflation",
    "labor",
    "liquidity",
    "monetary",
    "credit",
    "financial_stress",
    "housing",
    "consumer",
    "currency",
    "commodity",
    "breadth",
    "valuation",
    "earnings_cycle",
    "risk_appetite",
)


def compute_macro_snapshot(
    observations: list[dict[str, Any]],
    config: MacroConfig,
    *,
    as_of: datetime,
    regime_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cutoff = _utc(as_of)
    features = build_feature_snapshot(observations, config, as_of=cutoff)
    scores = {
        score_name: compute_score(
            score_name,
            config.score_weights[score_name],
            features,
            config,
        )
        for score_name in CORE_SCORES
    }
    regime = classify_regime(
        scores,
        config,
        as_of=cutoff,
        history=regime_history or [],
    )
    implications = implication_map(scores, config)
    data_quality = macro_data_quality(features, scores)
    payload = {
        "schema": "macro_snapshot_v1",
        "as_of": cutoff.isoformat(),
        "config_version": config.version,
        "config_hash": config.config_hash,
        "features": features,
        "scores": {name: score.__dict__ for name, score in scores.items()},
        "regime": regime,
        "cycle_clock": cycle_clock(scores, regime),
        "implications": implications,
        "data_quality": data_quality,
        "fixture_evidence": False,
        "financial_evidence": False,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def compute_score(
    name: str,
    weights: Mapping[str, float],
    features: Mapping[str, Mapping[str, Any]],
    config: MacroConfig,
) -> MacroScore:
    available: list[tuple[str, float, float, float]] = []
    missing: list[str] = []
    stale: list[str] = []
    for series_id, weight in weights.items():
        feature = features.get(series_id)
        if not feature or feature.get("normalized_score") is None:
            missing.append(series_id)
            continue
        if feature.get("stale"):
            stale.append(series_id)
            continue
        available.append(
            (
                series_id,
                float(weight),
                float(feature["normalized_score"]),
                float(feature.get("quality_confidence_multiplier", 1.0)),
            )
        )
    coverage = sum(weight for _, weight, _, _ in available)
    if coverage < config.minimum_score_coverage:
        status = (
            ScoreStatus.UNAVAILABLE
            if not available
            else ScoreStatus.DATA_INCOMPLETE
        )
        value = None
    else:
        value = float(
            np.clip(
                sum(
                    weight * score
                    for _, weight, score, _ in available
                )
                / coverage,
                -100.0,
                100.0,
            )
        )
        status = (
            ScoreStatus.VALID
            if coverage >= config.partial_score_coverage and not stale
            else ScoreStatus.PARTIAL
        )
    contributions: list[dict[str, Any]] = [
        {
            "series_id": series_id,
            "weighted_contribution": weight * score,
            "normalized_score": score,
            "configured_weight": weight,
        }
        for series_id, weight, score, _ in available
    ]
    positive = tuple(
        sorted(
            (
                row
                for row in contributions
                if float(row["weighted_contribution"]) > 0
            ),
            key=lambda row: float(row["weighted_contribution"]),
            reverse=True,
        )[:5]
    )
    negative = tuple(
        sorted(
            (
                row
                for row in contributions
                if float(row["weighted_contribution"]) < 0
            ),
            key=lambda row: float(row["weighted_contribution"]),
        )[:5]
    )
    quality_confidence = (
        sum(weight * quality for _, weight, _, quality in available) / coverage
        if coverage > 0
        else 0.0
    )
    confidence = (
        min(1.0, coverage)
        * (1.0 if status == ScoreStatus.VALID else 0.75)
        * quality_confidence
    )
    return MacroScore(
        name=name,
        value=value,
        confidence=confidence,
        coverage=coverage,
        status=status.value,
        positive_contributions=positive,
        negative_contributions=negative,
        missing_inputs=tuple(sorted(missing)),
        stale_inputs=tuple(sorted(stale)),
    )


def classify_regime(
    scores: Mapping[str, MacroScore],
    config: MacroConfig,
    *,
    as_of: datetime,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    growth = _value(scores, "growth")
    inflation = _value(scores, "inflation")
    liquidity = _value(scores, "liquidity")
    monetary = _value(scores, "monetary")
    credit = _value(scores, "credit")
    risk = _value(scores, "risk_appetite")
    commodity = _value(scores, "commodity")
    currency = _value(scores, "currency")
    required = (growth, inflation, liquidity, credit, risk)
    if any(value is None for value in required):
        candidate = RegimeLabel.UNKNOWN.value
        reasons = ["CRITICAL_SCORE_UNAVAILABLE"]
    else:
        assert growth is not None
        assert inflation is not None
        assert liquidity is not None
        assert credit is not None
        assert risk is not None
        candidate, reasons = _overall_candidate(
            float(growth),
            float(inflation),
            float(liquidity),
            float(credit),
            float(risk),
            float(config.hysteresis["transition_band"]),
        )
    accepted, hysteresis_status, confirmations = apply_hysteresis(
        candidate,
        as_of=as_of,
        history=history,
        minimum_confirmations=int(
            config.hysteresis["minimum_confirmations"]
        ),
        minimum_regime_days=int(config.hysteresis["minimum_regime_days"]),
    )
    confidence_values = [
        scores[name].confidence
        for name in ("growth", "inflation", "liquidity", "credit", "risk_appetite")
    ]
    confidence = float(np.mean(confidence_values))
    return {
        "growth_regime": _axis(growth, "ACCELERATING", "SLOWING"),
        "inflation_regime": _axis(inflation, "FALLING", "RISING"),
        "liquidity_regime": _axis(liquidity, "EXPANDING", "CONTRACTING"),
        "monetary_regime": _axis(monetary, "SUPPORTIVE", "RESTRICTIVE"),
        "credit_regime": _axis(credit, "IMPROVING", "DETERIORATING"),
        "market_regime": _axis(risk, "RISK_ON", "RISK_OFF"),
        "currency_regime": _axis(currency, "SUPPORTIVE", "HEADWIND"),
        "commodity_regime": _axis(commodity, "STRENGTHENING", "WEAKENING"),
        "candidate_regime": candidate,
        "overall_macro_regime": accepted,
        "hysteresis_status": hysteresis_status,
        "candidate_confirmations": confirmations,
        "confidence": confidence,
        "reasons": reasons,
        "available_score_count": sum(
            score.value is not None for score in scores.values()
        ),
        "score_count": len(scores),
    }


def apply_hysteresis(
    candidate: str,
    *,
    as_of: datetime,
    history: list[dict[str, Any]],
    minimum_confirmations: int,
    minimum_regime_days: int,
) -> tuple[str, str, int]:
    if not history:
        return candidate, "INITIAL_REGIME", 1
    ordered = sorted(history, key=lambda row: str(row["as_of"]))
    accepted_history = [
        row
        for row in ordered
        if row["regime"].get("overall_macro_regime")
        not in {RegimeLabel.TRANSITION.value, RegimeLabel.UNKNOWN.value}
    ]
    previous = (
        accepted_history[-1]["regime"]
        if accepted_history
        else ordered[-1]["regime"]
    )
    previous_accepted = str(previous["overall_macro_regime"])
    if candidate == previous_accepted:
        return candidate, "REGIME_STABLE", minimum_confirmations
    consecutive = 1
    for row in reversed(ordered):
        if row["regime"].get("candidate_regime") != candidate:
            break
        consecutive += 1
    accepted_dates = [
        pd.Timestamp(row["as_of"])
        for row in ordered
        if row["regime"].get("overall_macro_regime") == previous_accepted
    ]
    last_transition = (
        min(accepted_dates) if accepted_dates else pd.Timestamp(as_of)
    )
    elapsed_days = (pd.Timestamp(as_of) - last_transition).days
    if consecutive < minimum_confirmations or elapsed_days < minimum_regime_days:
        return RegimeLabel.TRANSITION.value, "PENDING_CONFIRMATION", consecutive
    return candidate, "REGIME_CHANGE_CONFIRMED", consecutive


def cycle_clock(
    scores: Mapping[str, MacroScore],
    regime: Mapping[str, Any],
) -> dict[str, Any]:
    growth = _value(scores, "growth")
    inflation = _value(scores, "inflation")
    if growth is None or inflation is None:
        quadrant = "UNKNOWN"
    elif growth >= 0 and inflation >= 0:
        quadrant = "GROWTH_ACCELERATING_INFLATION_FALLING"
    elif growth >= 0 and inflation < 0:
        quadrant = "GROWTH_ACCELERATING_INFLATION_RISING"
    elif growth < 0 and inflation >= 0:
        quadrant = "GROWTH_SLOWING_INFLATION_FALLING"
    else:
        quadrant = "GROWTH_SLOWING_INFLATION_RISING"
    return {
        "quadrant": quadrant,
        "confidence": regime["confidence"],
        "liquidity_overlay": regime["liquidity_regime"],
        "credit_overlay": regime["credit_regime"],
        "predictive_claim": False,
    }


def implication_map(
    scores: Mapping[str, MacroScore],
    config: MacroConfig,
) -> dict[str, Any]:
    sectors = {
        name: _mapped_implication(name, mapping, scores)
        for name, mapping in config.sector_mappings.items()
    }
    regions = {
        name: _mapped_implication(name, mapping, scores)
        for name, mapping in config.regional_mappings.items()
    }
    return {
        "sectors_and_asset_classes": sectors,
        "regions": regions,
        "mapping_is_order_signal": False,
        "technical_confirmation_required": True,
        "fundamental_confirmation_required": True,
    }


def exposure_multiplier(
    regime: Mapping[str, Any],
    config: MacroConfig,
) -> float:
    overall = regime.get("overall_macro_regime")
    market = regime.get("market_regime")
    if overall in {RegimeLabel.UNKNOWN.value, RegimeLabel.TRANSITION.value}:
        raw = config.portfolio["transition_multiplier"]
    elif market == "RISK_OFF":
        raw = config.portfolio["risk_off_multiplier"]
    elif market == "RISK_ON":
        raw = config.portfolio["risk_on_multiplier"]
    else:
        raw = config.portfolio["neutral_multiplier"]
    return float(
        np.clip(
            raw,
            config.portfolio["minimum_multiplier"],
            config.portfolio["maximum_multiplier"],
        )
    )


def apply_macro_exposure(
    weights: pd.DataFrame,
    multiplier: float | pd.Series,
    *,
    minimum: float = 0.5,
    maximum: float = 1.1,
) -> pd.DataFrame:
    if isinstance(multiplier, pd.Series):
        bounded = multiplier.reindex(weights.index).ffill().fillna(1.0).clip(
            lower=minimum, upper=maximum
        )
        adjusted = weights.mul(bounded, axis=0)
    else:
        adjusted = weights * float(np.clip(multiplier, minimum, maximum))
    total = adjusted.sum(axis=1)
    scale = (1.0 / total.where(total > 1.0)).fillna(1.0).clip(upper=1.0)
    return adjusted.mul(scale, axis=0).clip(lower=0.0)


def compare_macro_variant(
    baseline_returns: pd.Series,
    macro_returns: pd.Series,
) -> dict[str, Any]:
    baseline = baseline_returns.align(macro_returns, join="inner")[0].fillna(0.0)
    macro = macro_returns.reindex(baseline.index).fillna(0.0)
    baseline_nav = (1.0 + baseline).cumprod()
    macro_nav = (1.0 + macro).cumprod()
    baseline_dd = float((baseline_nav / baseline_nav.cummax() - 1.0).min())
    macro_dd = float((macro_nav / macro_nav.cummax() - 1.0).min())
    baseline_total = float(baseline_nav.iloc[-1] - 1.0) if len(baseline_nav) else 0.0
    macro_total = float(macro_nav.iloc[-1] - 1.0) if len(macro_nav) else 0.0
    accepted = macro_dd > baseline_dd or macro_total > baseline_total
    return {
        "status": "VALUE_ADDED" if accepted else "NO_VALUE_ADDED",
        "baseline_total_return": baseline_total,
        "macro_total_return": macro_total,
        "baseline_maximum_drawdown": baseline_dd,
        "macro_maximum_drawdown": macro_dd,
        "drawdown_improvement": macro_dd - baseline_dd,
        "return_difference": macro_total - baseline_total,
        "baseline_comparison_required": True,
    }


def macro_data_quality(
    features: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, MacroScore],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for feature in features.values():
        status = str(feature["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    critical = ("growth", "inflation", "liquidity", "credit", "breadth", "currency", "commodity")
    critical_complete = all(scores[name].value is not None for name in critical)
    return {
        "status": "GO" if critical_complete else "DATA_INCOMPLETE",
        "feature_status_counts": dict(sorted(status_counts.items())),
        "critical_scores_complete": critical_complete,
        "vintage_history_unavailable_count": sum(
            feature.get("vintage_history_status")
            == "VINTAGE_HISTORY_UNAVAILABLE"
            for feature in features.values()
        ),
        "fixture_results_are_macro_evidence": False,
    }


def _overall_candidate(
    growth: float,
    inflation: float,
    liquidity: float,
    credit: float,
    risk: float,
    transition_band: float,
) -> tuple[str, list[str]]:
    if max(abs(growth), abs(inflation), abs(liquidity), abs(credit), abs(risk)) < transition_band:
        return RegimeLabel.TRANSITION.value, ["SCORES_WITHIN_TRANSITION_BAND"]
    if growth < -50 and credit < -25:
        return RegimeLabel.CONTRACTION.value, ["GROWTH_WEAK", "CREDIT_DETERIORATING"]
    if growth < -25 and inflation < -25:
        return RegimeLabel.STAGFLATION.value, ["GROWTH_SLOWING", "INFLATION_RISING"]
    if growth >= 0 and inflation >= 0:
        label = RegimeLabel.EXPANSION_DISINFLATION.value
    elif growth >= 0 and inflation < 0:
        label = RegimeLabel.EXPANSION_REFLATION.value
    elif growth < 0 and inflation >= 0:
        label = RegimeLabel.SLOWDOWN_DISINFLATION.value
    else:
        label = RegimeLabel.SLOWDOWN_INFLATION.value
    reasons = [
        f"GROWTH_SCORE_{growth:.1f}",
        f"INFLATION_SCORE_{inflation:.1f}",
        f"LIQUIDITY_SCORE_{liquidity:.1f}",
        f"CREDIT_SCORE_{credit:.1f}",
        f"RISK_SCORE_{risk:.1f}",
    ]
    return label, reasons


def _mapped_implication(
    name: str,
    mapping: Mapping[str, float],
    scores: Mapping[str, MacroScore],
) -> dict[str, Any]:
    available: list[tuple[str, float, MacroScore, float]] = []
    for score_name, weight in mapping.items():
        score = scores.get(score_name)
        if score is None or score.value is None:
            continue
        available.append((score_name, weight, score, score.value))
    coverage = sum(abs(weight) for _, weight, _, _ in available) / max(
        sum(abs(weight) for weight in mapping.values()),
        1e-12,
    )
    raw = (
        sum(weight * score_value for _, weight, _, score_value in available)
        / max(sum(abs(weight) for _, weight, _, _ in available), 1e-12)
        if available
        else 0.0
    )
    status = "POSITIVE" if raw > 15 else "NEGATIVE" if raw < -15 else "NEUTRAL"
    return {
        "name": name,
        "macro_support": status,
        "score": float(raw),
        "confidence": float(
            coverage
            * np.mean([score.confidence for _, _, score, _ in available])
        )
        if available
        else 0.0,
        "drivers": [
            {
                "score": score_name,
                "weight": weight,
                "value": score_value,
            }
            for score_name, weight, _, score_value in available
        ],
        "technical_confirmation": "REQUIRED",
        "fundamental_confirmation": "REQUIRED",
        "final_status": "WATCHLIST",
    }


def _value(
    scores: Mapping[str, MacroScore],
    name: str,
) -> float | None:
    score = scores.get(name)
    return None if score is None else score.value


def _axis(
    value: float | None,
    positive: str,
    negative: str,
) -> str:
    if value is None:
        return "UNKNOWN"
    if value > 15:
        return positive
    if value < -15:
        return negative
    return "NEUTRAL"


def _utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=UTC)
        if value.tzinfo is None
        else value.astimezone(UTC)
    )
