from __future__ import annotations

import math

from stocks.features.normalization import positive_normalize
from stocks.features.technical import effective_volatility


def region_score(
    *,
    technical_score: float,
    fundamental_score: float,
    earnings_revision_score: float,
    macro_score: float,
    valuation_score: float,
    currency_score: float,
    liquidity_score: float,
) -> float:
    return (
        0.25 * technical_score
        + 0.25 * fundamental_score
        + 0.15 * earnings_revision_score
        + 0.10 * macro_score
        + 0.10 * valuation_score
        + 0.10 * currency_score
        + 0.05 * liquidity_score
    )


def emerging_market_region_score(
    *,
    base_region_score: float,
    fx_volatility: float,
    political_risk: float,
    external_vulnerability: float,
    liquidity_penalty: float,
    lambda_fx: float = 1.0,
    lambda_political: float = 1.0,
    lambda_external: float = 1.0,
    lambda_liquidity: float = 1.0,
) -> float:
    return (
        base_region_score
        - lambda_fx * fx_volatility
        - lambda_political * political_risk
        - lambda_external * external_vulnerability
        - lambda_liquidity * liquidity_penalty
    )


def score_to_volatility_adjusted_weights(
    scores: dict[str, float],
    forecast_volatilities: dict[str, float],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for key, score in scores.items():
        volatility = forecast_volatilities[key]
        if volatility <= 0:
            raise ValueError(f"forecast volatility must be positive for {key}")
        raw[key] = max(score, 0.0) / volatility
    return positive_normalize(raw)


def score_to_softmax_weights(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return {}
    scaled = {key: value / temperature for key, value in scores.items()}
    offset = max(scaled.values())
    exp_values = {key: math.exp(value - offset) for key, value in scaled.items()}
    total = sum(exp_values.values())
    return {key: value / total for key, value in exp_values.items()}


def stock_weight_within_region(
    *,
    stock_scores: dict[str, float],
    effective_volatilities: dict[str, float],
    threshold: float = 0.0,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for symbol, score in stock_scores.items():
        volatility = effective_volatilities[symbol]
        if volatility <= 0:
            raise ValueError(f"effective volatility must be positive for {symbol}")
        raw[symbol] = max(score - threshold, 0.0) / volatility
    return positive_normalize(raw)


def effective_volatility_with_fx(
    equity_volatility: float,
    fx_volatility: float,
    equity_fx_correlation: float,
) -> float:
    return effective_volatility(
        equity_volatility=equity_volatility,
        fx_volatility=fx_volatility,
        equity_fx_correlation=equity_fx_correlation,
    )


def fx_return(current_eur_value: float, previous_eur_value: float) -> float:
    if previous_eur_value <= 0:
        raise ValueError("previous_eur_value must be positive")
    return current_eur_value / previous_eur_value - 1.0


def external_risk_score(
    *,
    current_account_deficit_to_gdp: float,
    short_term_external_debt_to_fx_reserves: float,
    fx_reserves_to_monthly_imports: float,
) -> float:
    return (
        current_account_deficit_to_gdp
        + short_term_external_debt_to_fx_reserves
        - fx_reserves_to_monthly_imports
    )


def commodity_beta(
    covariance_country_commodity: float,
    commodity_variance: float,
) -> float:
    if commodity_variance <= 0:
        raise ValueError("commodity_variance must be positive")
    return covariance_country_commodity / commodity_variance


def usd_stress_beta(covariance_country_usd: float, usd_variance: float) -> float:
    if usd_variance <= 0:
        raise ValueError("usd_variance must be positive")
    return covariance_country_usd / usd_variance


def country_adjusted_momentum(momentum: float, country_median_momentum: float) -> float:
    return momentum - country_median_momentum


def sector_adjusted_value(raw_value: float, sector_median_value: float) -> float:
    return raw_value - sector_median_value
