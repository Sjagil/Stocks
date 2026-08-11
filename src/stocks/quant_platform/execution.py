from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransactionCostModel:
    fixed_commission: float = 0.35
    commission_bps: float = 0.0
    half_spread_bps: float = 1.0
    base_slippage_bps: float = 2.0
    impact_coefficient: float = 10.0

    def estimate(
        self,
        *,
        price: float,
        quantity: int,
        average_daily_volume: float,
        volatility: float,
        time_of_day_multiplier: float = 1.0,
    ) -> dict[str, float]:
        if price <= 0 or quantity <= 0 or average_daily_volume <= 0 or volatility < 0 or time_of_day_multiplier <= 0:
            raise ValueError("invalid transaction cost inputs")
        notional = price * quantity
        participation = quantity / average_daily_volume
        commission = max(self.fixed_commission, notional * self.commission_bps / 10_000)
        spread = notional * self.half_spread_bps / 10_000
        slippage = notional * self.base_slippage_bps * time_of_day_multiplier / 10_000
        impact_bps = self.impact_coefficient * volatility * math.sqrt(participation) * 100
        market_impact = notional * impact_bps / 10_000
        return {
            "notional": notional,
            "commission": commission,
            "half_spread": spread,
            "slippage": slippage,
            "market_impact": market_impact,
            "participation": participation,
            "expected_cost": commission + spread + slippage + market_impact,
            "expected_cost_bps": (commission + spread + slippage + market_impact) / notional * 10_000,
        }

    def economically_executable(self, expected_alpha: float, costs: MappingLike, *, safety_margin: float) -> bool:
        return float(expected_alpha) > float(costs["expected_cost_bps"]) / 10_000 + float(safety_margin)


MappingLike = dict[str, float]


class OptimalExecutionEngine:
    def schedule(
        self,
        quantity: int,
        intervals: int,
        *,
        method: str,
        forecast_volumes: Iterable[float] | None = None,
        participation_rate: float = 0.10,
        risk_aversion: float = 1.0,
        volatility: float = 0.02,
        temporary_impact: float = 1e-6,
    ) -> dict[str, Any]:
        if quantity <= 0 or intervals <= 0:
            raise ValueError("quantity and intervals must be positive")
        method = method.upper().replace("-", "_")
        if method == "TWAP":
            proportions = np.ones(intervals)
        elif method == "VWAP":
            volumes = _volumes(forecast_volumes, intervals)
            proportions = volumes
        elif method == "POV":
            volumes = _volumes(forecast_volumes, intervals)
            if not 0 < participation_rate <= 1:
                raise ValueError("participation_rate must be in (0, 1]")
            proportions = np.minimum(volumes * participation_rate, quantity)
        elif method == "IMPLEMENTATION_SHORTFALL":
            if risk_aversion < 0:
                raise ValueError("risk_aversion cannot be negative")
            proportions = np.exp(-risk_aversion * np.arange(intervals) / max(intervals - 1, 1))
        elif method in {"ALMGREN_CHRISS", "ALMGRENCHRISS"}:
            if risk_aversion < 0 or volatility < 0 or temporary_impact <= 0:
                raise ValueError("invalid Almgren-Chriss parameters")
            kappa = math.sqrt(max(risk_aversion * volatility**2 / temporary_impact, 1e-12))
            remaining = np.sinh(kappa * np.arange(intervals, -1, -1)) / np.sinh(kappa * intervals)
            proportions = remaining[:-1] - remaining[1:]
            method = "ALMGREN_CHRISS"
        else:
            raise ValueError("unsupported execution method")
        schedule = _integer_schedule(quantity, proportions)
        return {
            "method": method,
            "quantity": quantity,
            "intervals": intervals,
            "schedule": [{"interval": index, "quantity": int(value)} for index, value in enumerate(schedule)],
            "scheduled_quantity": int(schedule.sum()),
            "paper_test_only": True,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


def execution_shortfall(
    fills: pd.DataFrame,
    *,
    decision_price: float,
    side: str,
) -> dict[str, float]:
    if not {"price", "quantity"} <= set(fills) or decision_price <= 0:
        raise ValueError("fills and decision_price are invalid")
    quantity = pd.to_numeric(fills["quantity"], errors="coerce")
    price = pd.to_numeric(fills["price"], errors="coerce")
    total = float(quantity.sum())
    if total <= 0:
        raise ValueError("filled quantity must be positive")
    average = float((price * quantity).sum() / total)
    direction = 1 if side.upper() == "BUY" else -1 if side.upper() == "SELL" else 0
    if direction == 0:
        raise ValueError("side must be BUY or SELL")
    shortfall = direction * (average / decision_price - 1)
    return {"average_fill_price": average, "decision_price": decision_price, "implementation_shortfall": shortfall, "implementation_shortfall_bps": shortfall * 10_000}


def _volumes(values: Iterable[float] | None, intervals: int) -> np.ndarray:
    if values is None:
        raise ValueError("forecast_volumes are required")
    array = np.asarray(list(values), dtype=float)
    if len(array) != intervals or not np.isfinite(array).all() or (array < 0).any() or array.sum() <= 0:
        raise ValueError("forecast_volumes must be non-negative and match intervals")
    return array


def _integer_schedule(quantity: int, proportions: np.ndarray) -> np.ndarray:
    values = np.asarray(proportions, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("execution proportions must be finite and positive")
    exact = values / values.sum() * quantity
    schedule = np.floor(exact).astype(int)
    remainder = quantity - int(schedule.sum())
    if remainder:
        order = np.argsort(-(exact - schedule), kind="stable")
        schedule[order[:remainder]] += 1
    return schedule
