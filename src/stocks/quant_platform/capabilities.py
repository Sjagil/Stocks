from __future__ import annotations

from typing import Any


CAPABILITIES: tuple[dict[str, Any], ...] = (
    {"id": 1, "level": 1, "name": "Multi-Asset Market Data Explorer", "module": "data/providers/storage/explorer"},
    {"id": 2, "level": 1, "name": "Performance & Risk Analyzer", "module": "analytics"},
    {"id": 3, "level": 2, "name": "Portfolio Optimizer", "module": "portfolio"},
    {"id": 4, "level": 2, "name": "Risk-Parity Portfolio", "module": "portfolio"},
    {"id": 5, "level": 3, "name": "Factor Investing Engine", "module": "factors"},
    {"id": 6, "level": 3, "name": "Cross-Sectional Stock Ranking", "module": "factors"},
    {"id": 7, "level": 3, "name": "Technical Strategy Research", "module": "technical"},
    {"id": 8, "level": 4, "name": "Professional Backtesting", "module": "backtest"},
    {"id": 9, "level": 4, "name": "Walk-Forward Optimization", "module": "validation"},
    {"id": 10, "level": 4, "name": "Market Regime Detector", "module": "regime"},
    {"id": 11, "level": 4, "name": "Strategy Allocation", "module": "regime"},
    {"id": 12, "level": 5, "name": "Statistical Arbitrage", "module": "stat_arb"},
    {"id": 13, "level": 5, "name": "Cross-Sectional Mean Reversion", "module": "stat_arb"},
    {"id": 14, "level": 5, "name": "Volatility Modeling", "module": "risk"},
    {"id": 15, "level": 5, "name": "VaR / Expected Shortfall", "module": "risk"},
    {"id": 16, "level": 5, "name": "Monte Carlo Portfolio Simulator", "module": "risk"},
    {"id": 17, "level": 6, "name": "ML Return Prediction", "module": "ml"},
    {"id": 18, "level": 6, "name": "Meta-Labeling", "module": "ml"},
    {"id": 19, "level": 6, "name": "Probability-Calibrated Signals", "module": "ml"},
    {"id": 20, "level": 6, "name": "Alternative Data + NLP News", "module": "intelligence"},
    {"id": 21, "level": 6, "name": "News Event Study", "module": "intelligence"},
    {"id": 22, "level": 6, "name": "SEC Filing Intelligence", "module": "intelligence"},
    {"id": 23, "level": 7, "name": "Dynamic Multi-Asset Allocator", "module": "allocation"},
    {"id": 24, "level": 7, "name": "Hierarchical Risk Parity", "module": "allocation"},
    {"id": 25, "level": 7, "name": "Black-Litterman", "module": "allocation"},
    {"id": 26, "level": 7, "name": "Portfolio Risk Engine", "module": "allocation"},
    {"id": 27, "level": 7, "name": "Transaction Cost Model", "module": "execution"},
    {"id": 28, "level": 7, "name": "Optimal Execution", "module": "execution"},
    {"id": 29, "level": 8, "name": "Factor Risk Model", "module": "professional"},
    {"id": 30, "level": 8, "name": "Alpha Combination", "module": "professional"},
    {"id": 31, "level": 8, "name": "Regime-Conditional Mixture of Experts", "module": "professional"},
    {"id": 32, "level": 8, "name": "Portfolio-Level Reinforcement Learning", "module": "professional"},
    {"id": 33, "level": 8, "name": "Full Quant Portfolio Manager", "module": "manager"},
)


def capability_registry() -> dict[str, Any]:
    return {
        "schema": "quant_platform_capability_registry_v1",
        "capability_count": len(CAPABILITIES),
        "levels": sorted({item["level"] for item in CAPABILITIES}),
        "capabilities": [dict(item, status="IMPLEMENTED_AND_TESTED") for item in CAPABILITIES],
        "research_only_default": True,
        "automatic_broker_submission": False,
    }
