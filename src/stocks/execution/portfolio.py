from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class PortfolioState:
    cash: Decimal = Decimal("100000")
    reserved_cash: Decimal = Decimal("0")
    positions: dict[int, Decimal] = field(default_factory=dict)
    average_cost: dict[int, Decimal] = field(default_factory=dict)
    realized_pnl: Decimal = Decimal("0")
    commissions: Decimal = Decimal("0")
    high_water_mark: Decimal = Decimal("100000")
    last_prices: dict[int, Decimal] = field(default_factory=dict)

    def equity(self) -> Decimal:
        position_value = sum(quantity * self.last_prices.get(con_id, self.average_cost.get(con_id, Decimal("0"))) for con_id, quantity in self.positions.items())
        return self.cash + position_value

    def drawdown(self) -> Decimal:
        equity = self.equity()
        if equity > self.high_water_mark:
            self.high_water_mark = equity
        if self.high_water_mark == 0:
            return Decimal("0")
        return equity / self.high_water_mark - Decimal("1")

