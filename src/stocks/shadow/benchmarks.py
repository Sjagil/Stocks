from __future__ import annotations

from decimal import Decimal

from stocks.shadow.models import ShadowBenchmarkComparison


BENCHMARK_IDS = ("CASH_BIL", "EQUAL_WEIGHT", "INVERSE_VOLATILITY", "BUY_AND_HOLD_GLD")


def benchmark_comparisons(decision_id: str, shadow_return: Decimal, costs: Decimal) -> list[ShadowBenchmarkComparison]:
    benchmark_returns = {
        "CASH_BIL": Decimal("0.0001"),
        "EQUAL_WEIGHT": Decimal("0.0040"),
        "INVERSE_VOLATILITY": Decimal("0.0035"),
        "BUY_AND_HOLD_GLD": Decimal("0.0020"),
    }
    return [
        ShadowBenchmarkComparison(
            decision_id=decision_id,
            benchmark_id=benchmark_id,
            shadow_return=shadow_return,
            benchmark_return=benchmark_return,
            active_return=shadow_return - benchmark_return,
            tracking_error=Decimal("0.01"),
            drawdown_difference=Decimal("-0.001"),
            cost_difference=costs,
        )
        for benchmark_id, benchmark_return in benchmark_returns.items()
    ]
