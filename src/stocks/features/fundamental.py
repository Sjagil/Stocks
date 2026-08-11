from __future__ import annotations


def market_cap(share_price: float, shares_outstanding: float) -> float:
    return share_price * shares_outstanding


def free_float_market_cap(
    market_capitalization: float,
    free_float_percentage: float,
) -> float:
    if market_capitalization < 0:
        raise ValueError("market_capitalization cannot be negative")
    if not 0.0 <= free_float_percentage <= 1.0:
        raise ValueError("free_float_percentage must be in [0, 1]")
    return market_capitalization * free_float_percentage


def enterprise_value(
    market_capitalization: float,
    total_debt: float,
    preferred_equity: float,
    minority_interest: float,
    cash_and_equivalents: float,
) -> float:
    return (
        market_capitalization
        + total_debt
        + preferred_equity
        + minority_interest
        - cash_and_equivalents
    )


def pe_ratio(share_price: float, earnings_per_share: float) -> float:
    if earnings_per_share <= 0:
        raise ValueError("earnings_per_share must be positive")
    return share_price / earnings_per_share


def earnings_yield(net_income: float, market_capitalization: float) -> float:
    if market_capitalization <= 0:
        raise ValueError("market_capitalization must be positive")
    return net_income / market_capitalization


def price_to_book(market_capitalization: float, book_equity: float) -> float:
    if book_equity <= 0:
        raise ValueError("book_equity must be positive")
    return market_capitalization / book_equity


def price_to_sales(market_capitalization: float, revenue: float) -> float:
    if revenue <= 0:
        raise ValueError("revenue must be positive")
    return market_capitalization / revenue


def free_cash_flow_yield(free_cash_flow: float, enterprise_value_: float) -> float:
    if enterprise_value_ <= 0:
        raise ValueError("enterprise_value must be positive")
    return free_cash_flow / enterprise_value_


def ev_to_ebitda(enterprise_value_: float, ebitda: float) -> float:
    if ebitda <= 0:
        raise ValueError("ebitda must be positive")
    return enterprise_value_ / ebitda


def ev_to_ebit(enterprise_value_: float, ebit: float) -> float:
    if ebit <= 0:
        raise ValueError("ebit must be positive")
    return enterprise_value_ / ebit


def net_dividend_yield(dividend_yield: float, withholding_tax_rate: float) -> float:
    if not 0.0 <= withholding_tax_rate <= 1.0:
        raise ValueError("withholding_tax_rate must be in [0, 1]")
    return dividend_yield * (1.0 - withholding_tax_rate)


def payout_ratio(dividends: float, net_income: float) -> float:
    if net_income <= 0:
        raise ValueError("net_income must be positive")
    return dividends / net_income


def fcf_payout_ratio(dividends: float, free_cash_flow: float) -> float:
    if free_cash_flow <= 0:
        raise ValueError("free_cash_flow must be positive")
    return dividends / free_cash_flow


def buyback_yield(net_share_repurchases: float, market_capitalization: float) -> float:
    if market_capitalization <= 0:
        raise ValueError("market_capitalization must be positive")
    return net_share_repurchases / market_capitalization


def debt_reduction_yield(
    previous_debt: float,
    current_debt: float,
    market_capitalization: float,
) -> float:
    if market_capitalization <= 0:
        raise ValueError("market_capitalization must be positive")
    return (previous_debt - current_debt) / market_capitalization


def shareholder_yield(
    dividend_yield: float,
    buyback_yield_: float,
    debt_reduction_yield_: float,
) -> float:
    return dividend_yield + buyback_yield_ + debt_reduction_yield_


def roe(net_income: float, average_book_equity: float) -> float:
    if average_book_equity <= 0:
        raise ValueError("average_book_equity must be positive")
    return net_income / average_book_equity


def roa(net_income: float, average_total_assets: float) -> float:
    if average_total_assets <= 0:
        raise ValueError("average_total_assets must be positive")
    return net_income / average_total_assets


def roic(nopat: float, invested_capital: float) -> float:
    if invested_capital <= 0:
        raise ValueError("invested_capital must be positive")
    return nopat / invested_capital


def gross_profitability(gross_profit: float, total_assets: float) -> float:
    if total_assets <= 0:
        raise ValueError("total_assets must be positive")
    return gross_profit / total_assets


def margin(numerator: float, revenue: float) -> float:
    if revenue <= 0:
        raise ValueError("revenue must be positive")
    return numerator / revenue


def growth_rate(current_value: float, previous_value: float) -> float:
    if previous_value <= 0:
        raise ValueError("previous_value must be positive")
    return current_value / previous_value - 1.0


def cagr(current_value: float, previous_value: float, years: float) -> float:
    if current_value <= 0 or previous_value <= 0:
        raise ValueError("values must be positive")
    if years <= 0:
        raise ValueError("years must be positive")
    return (current_value / previous_value) ** (1.0 / years) - 1.0


def debt_to_equity(total_debt: float, book_equity: float) -> float:
    if book_equity <= 0:
        raise ValueError("book_equity must be positive")
    return total_debt / book_equity


def net_debt(total_debt: float, cash: float) -> float:
    return total_debt - cash


def net_debt_to_ebitda(net_debt_: float, ebitda: float) -> float:
    if ebitda <= 0:
        raise ValueError("ebitda must be positive")
    return net_debt_ / ebitda


def interest_coverage(ebit: float, interest_expense: float) -> float:
    if interest_expense <= 0:
        raise ValueError("interest_expense must be positive")
    return ebit / interest_expense


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    if current_liabilities <= 0:
        raise ValueError("current_liabilities must be positive")
    return current_assets / current_liabilities


def quick_ratio(
    cash: float,
    marketable_securities: float,
    receivables: float,
    current_liabilities: float,
) -> float:
    if current_liabilities <= 0:
        raise ValueError("current_liabilities must be positive")
    return (cash + marketable_securities + receivables) / current_liabilities


def asset_turnover(revenue: float, average_total_assets: float) -> float:
    if average_total_assets <= 0:
        raise ValueError("average_total_assets must be positive")
    return revenue / average_total_assets


def inventory_turnover(cost_of_goods_sold: float, average_inventory: float) -> float:
    if average_inventory <= 0:
        raise ValueError("average_inventory must be positive")
    return cost_of_goods_sold / average_inventory


def accruals(net_income: float, operating_cash_flow: float, average_total_assets: float) -> float:
    if average_total_assets <= 0:
        raise ValueError("average_total_assets must be positive")
    return (net_income - operating_cash_flow) / average_total_assets


def cash_conversion(operating_cash_flow: float, net_income: float) -> float:
    if net_income <= 0:
        raise ValueError("net_income must be positive")
    return operating_cash_flow / net_income


def fcf_conversion(free_cash_flow: float, net_income: float) -> float:
    if net_income <= 0:
        raise ValueError("net_income must be positive")
    return free_cash_flow / net_income


def dilution(current_shares: float, previous_shares: float) -> float:
    if previous_shares <= 0:
        raise ValueError("previous_shares must be positive")
    return current_shares / previous_shares - 1.0


def earnings_surprise(actual_eps: float, consensus_eps: float) -> float:
    return actual_eps - consensus_eps


def standardized_earnings_surprise(actual_eps: float, consensus_eps: float) -> float:
    if consensus_eps == 0:
        raise ValueError("consensus_eps cannot be zero")
    return (actual_eps - consensus_eps) / abs(consensus_eps)


def revision_percentage(new_consensus_eps: float, old_consensus_eps: float) -> float:
    if old_consensus_eps == 0:
        raise ValueError("old_consensus_eps cannot be zero")
    return new_consensus_eps / old_consensus_eps - 1.0


def revision_breadth(up_revisions: int, down_revisions: int, total_revisions: int) -> float:
    if total_revisions <= 0:
        raise ValueError("total_revisions must be positive")
    if up_revisions < 0 or down_revisions < 0:
        raise ValueError("revision counts cannot be negative")
    if up_revisions + down_revisions > total_revisions:
        raise ValueError("up and down revisions cannot exceed total_revisions")
    return (up_revisions - down_revisions) / total_revisions
