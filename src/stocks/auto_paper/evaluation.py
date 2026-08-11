from __future__ import annotations

from stocks.auto_paper.strategies import STRATEGY_IDS


BENCHMARKS = (
    "NO_TRADE",
    "BROAD_SHARIAH_EQUITY_BENCHMARK",
    "EQUAL_WEIGHT_ELIGIBLE_UNIVERSE",
    "BUY_AND_HOLD_UNDERLYING",
    "SECTOR_BENCHMARK",
)


def financial_evaluation_fixture() -> dict[str, object]:
    rows = []
    for strategy_id in STRATEGY_IDS:
        rows.append(
            {
                "strategy_id": strategy_id,
                "event_study_observations": 0,
                "transaction_cost_stress_bps": [5, 10, 20, 30, 50],
                "walk_forward_folds": 0,
                "benchmark_count": len(BENCHMARKS),
                "maximum_drawdown": None,
                "profit_factor": None,
                "expectancy": None,
                "positive_period_rate": None,
                "turnover": None,
                "concentration": None,
                "PBO": None,
                "DSR": None,
                "bootstrap_confidence": None,
                "decision": "NO_ALPHA_CANDIDATE",
                "provider_capability_status": "PROVIDER_CAPABILITY_BLOCKED",
                "pit_data_status": "PIT_DATA_INCOMPLETE",
            }
        )
    return {
        "status": "FINANCIAL_EVALUATION_CONTRACT_GO",
        "evidence_status": "PIT_DATA_INCOMPLETE",
        "provider_capability_status": "PROVIDER_CAPABILITY_BLOCKED",
        "synthetic_positive_evidence_used": False,
        "strategies": rows,
        "benchmarks": list(BENCHMARKS),
        "financial_decision": "NO_ALPHA_CANDIDATE",
        "FINANCIAL_FINALIST_GO": False,
        "FORWARD_RESEARCH_SHADOW": "blocked",
    }
