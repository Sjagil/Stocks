from __future__ import annotations

import math
from typing import Any

from scipy.stats import beta as beta_distribution


STRATEGY_SCORE_WEIGHTS = {
    "profit_factor": 0.20,
    "sharpe": 0.15,
    "sortino": 0.15,
    "expectancy": 0.15,
    "win_rate": 0.10,
    "regime_fit": 0.10,
    "stability": 0.10,
    "recency": 0.05,
}

FORMULA_FAMILY_MAP = {
    "asymmetric_ma": "asymmetric_ma_crossover",
    "asymmetric_ma_crossover": "asymmetric_ma_crossover",
    "bollinger_breakout": "bollinger_breakout",
    "choppiness_breakout": "volatility_contraction_breakout",
    "ma_channel": "ma_channel",
    "ma_crossover": "ma_crossover",
    "nr7_breakout": "volatility_contraction_breakout",
    "ppo_trend": "time_series_momentum",
    "rsi14_trend_pullback": "short_term_mean_reversion",
    "trend_pullback_consensus": "cross_sectional_momentum",
    "trend_quality_52w": "quality_momentum",
}


def infer_strategy_family(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in FORMULA_FAMILY_MAP:
        return FORMULA_FAMILY_MAP[normalized]
    if "bollinger" in normalized:
        return "bollinger_breakout"
    if "quality" in normalized:
        return "quality_momentum"
    if "momentum" in normalized or "trend" in normalized:
        return "time_series_momentum"
    if "pullback" in normalized or "rsi" in normalized:
        return "short_term_mean_reversion"
    if "breakout" in normalized or "contraction" in normalized:
        return "volatility_contraction_breakout"
    return normalized or "unknown"


def bayesian_positive_probability(
    wins: int,
    losses: int,
    *,
    break_even: float = 0.5,
    prior_alpha: float = 5.0,
    prior_beta: float = 5.0,
) -> dict[str, float | int | str]:
    wins = max(int(wins), 0)
    losses = max(int(losses), 0)
    break_even = min(max(float(break_even), 0.0), 1.0)
    alpha = prior_alpha + wins
    beta = prior_beta + losses
    return {
        "status": "GO" if wins + losses else "UNAVAILABLE",
        "evidence_unit": "POSITIVE_OOS_FOLD_OR_EVALUATION_PERIOD",
        "wins": wins,
        "losses": losses,
        "prior_alpha": prior_alpha,
        "prior_beta": prior_beta,
        "posterior_alpha": alpha,
        "posterior_beta": beta,
        "posterior_mean": alpha / (alpha + beta),
        "break_even_probability": break_even,
        "probability_above_break_even": float(
            beta_distribution.sf(break_even, alpha, beta)
        ),
    }


def score_strategy_evidence(
    candidate: dict[str, Any],
    *,
    regime_fit: float,
    data_quality: float = 1.0,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    pf = _number(
        candidate.get("profit_factor"),
        candidate.get("combined_period_profit_factor"),
    )
    sharpe = _number(
        candidate.get("sharpe"),
        candidate.get("combined_oos_Sharpe"),
    )
    sortino = _number(
        candidate.get("sortino"),
        candidate.get("combined_oos_Sortino"),
    )
    expectancy = _number(
        candidate.get("net_expectancy"),
        candidate.get("expectancy"),
    )
    wins, trials = _positive_evidence(candidate)
    bayesian = bayesian_positive_probability(wins, max(trials - wins, 0))
    plateau = _number(candidate.get("parameter_plateau_ratio"))
    stability = plateau
    if stability is None and trials:
        stability = wins / trials
    recency = _number(candidate.get("recent_forward_score"))

    _metric(metrics, "profit_factor", pf, _pf_score(pf))
    _metric(metrics, "sharpe", sharpe, _sharpe_score(sharpe))
    _metric(metrics, "sortino", sortino, _sharpe_score(sortino))
    _metric(metrics, "expectancy", expectancy, _expectancy_score(expectancy))
    _metric(
        metrics,
        "win_rate",
        (wins / trials if trials else None),
        (
            float(bayesian["posterior_mean"])
            if bayesian["status"] == "GO"
            else None
        ),
    )
    _metric(metrics, "regime_fit", regime_fit, _bounded(regime_fit))
    _metric(
        metrics,
        "stability",
        stability,
        _bounded(stability) if stability is not None else None,
    )
    _metric(
        metrics,
        "recency",
        recency,
        _bounded(recency) if recency is not None else None,
    )

    available_weight = sum(
        STRATEGY_SCORE_WEIGHTS[name]
        for name, metric in metrics.items()
        if metric["status"] == "GO"
    )
    weighted_total = sum(
        STRATEGY_SCORE_WEIGHTS[name] * float(metric["normalized"])
        for name, metric in metrics.items()
        if metric["status"] == "GO"
    )
    coverage = available_weight / sum(STRATEGY_SCORE_WEIGHTS.values())
    base_score = weighted_total / available_weight if available_weight else 0.0

    sample_count = int(
        _number(
            candidate.get("sample_count"),
            candidate.get("normal_cost_fill_count"),
        )
        or 0
    )
    sample_confidence = min(1.0, math.sqrt(sample_count / 50.0))
    evidence_penalty, limitations = _evidence_penalty(candidate)
    coverage_multiplier = 0.60 + 0.40 * coverage
    sample_multiplier = 0.75 + 0.25 * sample_confidence
    score = (
        base_score
        * coverage_multiplier
        * sample_multiplier
        * evidence_penalty
        * _bounded(data_quality)
    )
    missing = [name for name, row in metrics.items() if row["status"] != "GO"]
    evidence_status = (
        "INSUFFICIENT_EVIDENCE"
        if available_weight < 0.40 or sample_count < 10
        else "PARTIAL_EVIDENCE"
        if missing or limitations
        else "GO"
    )
    return {
        "score": round(_bounded(score), 6),
        "base_score": round(_bounded(base_score), 6),
        "evidence_status": evidence_status,
        "metric_coverage": round(coverage, 6),
        "sample_count": sample_count,
        "sample_confidence": round(sample_confidence, 6),
        "metrics": metrics,
        "bayesian_positive_probability": bayesian,
        "trade_break_even_win_rate": {
            "status": "UNAVAILABLE",
            "reason": "AVERAGE_WIN_AND_AVERAGE_LOSS_NOT_AVAILABLE",
        },
        "evidence_penalty": round(evidence_penalty, 6),
        "limitations": limitations,
        "missing_metrics": missing,
        "data_quality": round(_bounded(data_quality), 6),
    }


def _positive_evidence(candidate: dict[str, Any]) -> tuple[int, int]:
    wins = int(
        _number(
            candidate.get("positive_periods"),
            candidate.get("positive_fold_count"),
        )
        or 0
    )
    trials = int(
        _number(
            candidate.get("total_periods"),
            candidate.get("fold_count"),
        )
        or 0
    )
    return min(wins, trials), trials


def _evidence_penalty(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    penalty = 1.0
    limitations: list[str] = []
    statistical = candidate.get("statistical_evidence")
    if isinstance(statistical, dict):
        failed = sorted(key for key, value in statistical.items() if value is False)
        if failed:
            penalty *= 0.75
            limitations.extend(f"{key.upper()}_FAIL" for key in failed)
    if candidate.get("evidence_scope") == "SELECTION_CONDITIONED_REUSED_HISTORY":
        penalty *= 0.85
        limitations.append("INDEPENDENT_FORWARD_EVIDENCE_MISSING")
    if candidate.get("financial_finalist") is False:
        limitations.append("FINANCIAL_FINALIST_FALSE")
    return penalty, limitations


def _metric(
    metrics: dict[str, dict[str, Any]],
    name: str,
    raw: float | None,
    normalized: float | None,
) -> None:
    metrics[name] = {
        "status": "GO" if raw is not None and normalized is not None else "UNAVAILABLE",
        "raw": raw,
        "normalized": round(normalized, 6) if normalized is not None else None,
    }


def _pf_score(value: float | None) -> float | None:
    if value is None:
        return None
    return _bounded((value - 0.80) / 0.80)


def _sharpe_score(value: float | None) -> float | None:
    if value is None:
        return None
    return _bounded((value + 0.50) / 2.50)


def _expectancy_score(value: float | None) -> float | None:
    if value is None:
        return None
    return _bounded(0.5 + 0.5 * math.tanh(value / 0.01))


def _bounded(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None
