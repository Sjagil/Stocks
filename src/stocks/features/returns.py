from __future__ import annotations

import math


def price_return(current_price: float, previous_price: float) -> float:
    if previous_price <= 0:
        raise ValueError("previous_price must be positive")
    return current_price / previous_price - 1.0


def total_return(current_price: float, previous_price: float, dividend: float = 0.0) -> float:
    if previous_price <= 0:
        raise ValueError("previous_price must be positive")
    return (current_price + dividend - previous_price) / previous_price


def log_return(current_price: float, previous_price: float) -> float:
    if current_price <= 0 or previous_price <= 0:
        raise ValueError("prices must be positive")
    return math.log(current_price / previous_price)


def base_currency_return(local_return: float, fx_return: float) -> float:
    return (1.0 + local_return) * (1.0 + fx_return) - 1.0


def hedged_return(
    local_asset_return: float,
    hedge_carry: float,
    hedge_costs: float,
) -> float:
    return local_asset_return + hedge_carry - hedge_costs
