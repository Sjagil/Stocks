from __future__ import annotations

import math

import pytest

from stocks.features.fundamental import (
    accruals,
    asset_turnover,
    buyback_yield,
    cagr,
    cash_conversion,
    current_ratio,
    debt_reduction_yield,
    debt_to_equity,
    dilution,
    earnings_yield,
    ev_to_ebit,
    ev_to_ebitda,
    fcf_conversion,
    fcf_payout_ratio,
    enterprise_value,
    earnings_surprise,
    free_float_market_cap,
    free_cash_flow_yield,
    gross_profitability,
    growth_rate,
    interest_coverage,
    inventory_turnover,
    margin,
    market_cap,
    net_dividend_yield,
    net_debt,
    net_debt_to_ebitda,
    payout_ratio,
    pe_ratio,
    price_to_book,
    price_to_sales,
    quick_ratio,
    revision_breadth,
    revision_percentage,
    roa,
    roe,
    roic,
    shareholder_yield,
    standardized_earnings_surprise,
)
from stocks.features.normalization import positive_normalize, robust_z_score, winsorize, z_score
from stocks.features.returns import base_currency_return, log_return, price_return, total_return
from stocks.features.technical import (
    average_true_range,
    bollinger_bandwidth,
    bollinger_bands,
    bollinger_z,
    effective_volatility,
    exponential_moving_average,
    gap_return,
    momentum,
    percent_b,
    relative_strength,
    risk_adjusted_momentum,
    simple_moving_average,
    trend_strength,
    true_range,
    volume_z,
)
from stocks.regions.scoring import (
    commodity_beta,
    country_adjusted_momentum,
    effective_volatility_with_fx,
    emerging_market_region_score,
    external_risk_score,
    fx_return,
    region_score,
    sector_adjusted_value,
    score_to_softmax_weights,
    score_to_volatility_adjusted_weights,
    stock_weight_within_region,
    usd_stress_beta,
)


def test_local_and_base_currency_returns_include_interaction() -> None:
    assert price_return(110.0, 100.0) == pytest.approx(0.10)
    assert total_return(101.0, 100.0, dividend=2.0) == pytest.approx(0.03)
    assert log_return(math.e, 1.0) == pytest.approx(1.0)
    assert base_currency_return(0.10, -0.08) == pytest.approx(0.012)


def test_technical_formulas() -> None:
    assert simple_moving_average([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert exponential_moving_average([10.0, 12.0], period=3) == pytest.approx(11.0)
    assert momentum(120.0, 100.0) == pytest.approx(0.20)
    assert risk_adjusted_momentum(0.20, 0.10) == pytest.approx(2.0)
    assert trend_strength(105.0, 100.0, 2.5) == pytest.approx(2.0)
    assert relative_strength(0.08, 0.03) == pytest.approx(0.05)
    assert gap_return(105.0, 100.0) == pytest.approx(0.05)
    assert true_range(110.0, 100.0, 104.0) == pytest.approx(10.0)
    assert average_true_range([10.0, 12.0], period=3) == pytest.approx(11.0)
    middle, upper, lower = bollinger_bands([98.0, 100.0, 102.0], k=2.0)
    assert middle == pytest.approx(100.0)
    assert upper > middle > lower
    assert bollinger_z(102.0, [98.0, 100.0, 102.0]) > 0
    assert bollinger_bandwidth(upper, lower, middle) == pytest.approx((upper - lower) / middle)
    assert percent_b(middle, lower, upper) == pytest.approx(0.5)
    assert volume_z(130.0, [100.0, 110.0, 120.0]) > 0


def test_effective_volatility_with_fx() -> None:
    value = effective_volatility(0.20, 0.10, 0.50)
    assert value == pytest.approx(math.sqrt(0.04 + 0.01 + 0.02))
    assert effective_volatility_with_fx(0.20, 0.10, 0.50) == pytest.approx(value)


def test_normalization_helpers() -> None:
    population = [1.0, 2.0, 3.0]
    assert z_score(2.0, population) == pytest.approx(0.0)
    assert robust_z_score(2.0, population) == pytest.approx(0.0)
    assert winsorize(10.0, 0.0, 5.0) == pytest.approx(5.0)
    assert positive_normalize({"a": 2.0, "b": -1.0, "c": 2.0}) == {
        "a": 0.5,
        "b": 0.0,
        "c": 0.5,
    }


def test_fundamental_formulas() -> None:
    cap = market_cap(50.0, 1_000_000.0)
    assert cap == pytest.approx(50_000_000.0)
    assert free_float_market_cap(cap, 0.80) == pytest.approx(40_000_000.0)
    ev = enterprise_value(cap, 10_000_000.0, 0.0, 0.0, 5_000_000.0)
    assert ev == pytest.approx(55_000_000.0)
    assert pe_ratio(50.0, 5.0) == pytest.approx(10.0)
    assert earnings_yield(5_000_000.0, cap) == pytest.approx(0.10)
    assert price_to_book(cap, 25_000_000.0) == pytest.approx(2.0)
    assert price_to_sales(cap, 100_000_000.0) == pytest.approx(0.5)
    assert free_cash_flow_yield(5_500_000.0, ev) == pytest.approx(0.10)
    assert ev_to_ebitda(ev, 11_000_000.0) == pytest.approx(5.0)
    assert ev_to_ebit(ev, 5_500_000.0) == pytest.approx(10.0)
    assert net_dividend_yield(0.05, 0.15) == pytest.approx(0.0425)
    assert payout_ratio(2.0, 10.0) == pytest.approx(0.2)
    assert fcf_payout_ratio(2.0, 8.0) == pytest.approx(0.25)
    assert buyback_yield(1_000_000.0, cap) == pytest.approx(0.02)
    assert debt_reduction_yield(12_000_000.0, 10_000_000.0, cap) == pytest.approx(0.04)
    assert shareholder_yield(0.03, 0.02, 0.04) == pytest.approx(0.09)
    assert roe(10.0, 50.0) == pytest.approx(0.2)
    assert roa(10.0, 100.0) == pytest.approx(0.1)
    assert roic(12.0, 100.0) == pytest.approx(0.12)
    assert gross_profitability(40.0, 100.0) == pytest.approx(0.4)
    assert margin(25.0, 100.0) == pytest.approx(0.25)
    assert growth_rate(110.0, 100.0) == pytest.approx(0.10)
    assert cagr(121.0, 100.0, years=2.0) == pytest.approx(0.10)
    assert debt_to_equity(30.0, 60.0) == pytest.approx(0.5)
    assert net_debt(30.0, 5.0) == pytest.approx(25.0)
    assert net_debt_to_ebitda(25.0, 10.0) == pytest.approx(2.5)
    assert interest_coverage(20.0, 5.0) == pytest.approx(4.0)
    assert current_ratio(30.0, 15.0) == pytest.approx(2.0)
    assert quick_ratio(5.0, 10.0, 15.0, 20.0) == pytest.approx(1.5)
    assert asset_turnover(150.0, 100.0) == pytest.approx(1.5)
    assert inventory_turnover(80.0, 20.0) == pytest.approx(4.0)
    assert accruals(10.0, 8.0, 100.0) == pytest.approx(0.02)
    assert cash_conversion(8.0, 10.0) == pytest.approx(0.8)
    assert fcf_conversion(7.0, 10.0) == pytest.approx(0.7)
    assert dilution(110.0, 100.0) == pytest.approx(0.10)
    assert standardized_earnings_surprise(1.10, 1.00) == pytest.approx(0.10)
    assert earnings_surprise(1.10, 1.00) == pytest.approx(0.10)
    assert revision_percentage(1.20, 1.00) == pytest.approx(0.20)
    assert revision_breadth(6, 2, 10) == pytest.approx(0.4)


def test_region_score_and_em_penalties() -> None:
    base = region_score(
        technical_score=1.0,
        fundamental_score=1.0,
        earnings_revision_score=1.0,
        macro_score=1.0,
        valuation_score=1.0,
        currency_score=1.0,
        liquidity_score=1.0,
    )
    assert base == pytest.approx(1.0)
    assert emerging_market_region_score(
        base_region_score=base,
        fx_volatility=0.10,
        political_risk=0.20,
        external_vulnerability=0.05,
        liquidity_penalty=0.15,
    ) == pytest.approx(0.50)
    assert fx_return(1.10, 1.00) == pytest.approx(0.10)
    assert external_risk_score(
        current_account_deficit_to_gdp=0.04,
        short_term_external_debt_to_fx_reserves=0.30,
        fx_reserves_to_monthly_imports=0.20,
    ) == pytest.approx(0.14)
    assert commodity_beta(0.06, 0.03) == pytest.approx(2.0)
    assert usd_stress_beta(0.04, 0.02) == pytest.approx(2.0)
    assert country_adjusted_momentum(0.12, 0.05) == pytest.approx(0.07)
    assert sector_adjusted_value(0.08, 0.03) == pytest.approx(0.05)


def test_score_to_weight_helpers() -> None:
    weights = score_to_volatility_adjusted_weights(
        {"us": 1.0, "eu": 0.5, "jp": -1.0},
        {"us": 0.20, "eu": 0.10, "jp": 0.30},
    )
    assert weights == pytest.approx({"us": 0.5, "eu": 0.5, "jp": 0.0})

    softmax = score_to_softmax_weights({"us": 1.0, "eu": 1.0}, temperature=1.0)
    assert softmax == pytest.approx({"us": 0.5, "eu": 0.5})

    stock_weights = stock_weight_within_region(
        stock_scores={"A": 1.0, "B": 0.5, "C": -1.0},
        effective_volatilities={"A": 0.20, "B": 0.10, "C": 0.10},
    )
    assert stock_weights == pytest.approx({"A": 0.5, "B": 0.5, "C": 0.0})
