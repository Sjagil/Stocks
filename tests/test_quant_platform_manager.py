from __future__ import annotations

import numpy as np
import pandas as pd

from stocks.quant_platform import CAPABILITIES, FullQuantPortfolioManager, capability_registry, manager_feedback


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(60)
    assets = ["SPY", "GLD", "BTC", "BONDS"]
    returns = pd.DataFrame(rng.normal(0, [0.01, 0.008, 0.03, 0.004], size=(500, 4)), columns=assets)
    features = pd.DataFrame(
        {
            "expected_return": [0.08, 0.05, 0.15, 0.03],
            "volatility": [0.18, 0.14, 0.60, 0.08],
            "momentum": [0.1, 0.05, 0.2, 0.01],
            "macro": [0.1, 0.2, -0.1, 0.1],
            "liquidity": [1.0, 0.8, 0.6, 0.9],
            "drawdown": [-0.1, -0.05, -0.3, -0.02],
            "price": [500, 250, 100_000, 100],
            "average_daily_volume": [1e8, 1e7, 5e5, 2e7],
        },
        index=assets,
    )
    factors = pd.DataFrame(
        [
            {"symbol": asset, "available_at": "2026-01-01", "momentum": value, "quality": value, "value": value, "volatility": vol, "liquidity": liquidity}
            for asset, value, vol, liquidity in zip(assets, [3, 2, 4, 1], [0.18, 0.14, 0.60, 0.08], [4, 3, 1, 2], strict=True)
        ]
    )
    metadata = pd.DataFrame(
        {
            "sector": ["EQUITY", "METALS", "CRYPTO", "FIXED_INCOME"],
            "country": ["US", "US", "GLOBAL", "US"],
            "currency": ["USD"] * 4,
            "average_daily_volume": features["average_daily_volume"],
            "compliance_eligible": [True, True, False, True],
        },
        index=assets,
    )
    return returns, features, factors, metadata


def test_capability_registry_maps_all_33_projects_in_level_order() -> None:
    registry = capability_registry()
    assert len(CAPABILITIES) == registry["capability_count"] == 33
    assert [item["id"] for item in CAPABILITIES] == list(range(1, 34))
    assert registry["levels"] == list(range(1, 9))
    assert not registry["automatic_broker_submission"]


def test_full_manager_connects_pipeline_without_creating_orders() -> None:
    returns, features, factors, metadata = _inputs()
    report = FullQuantPortfolioManager(
        capital=100_000,
        risk_budget=0.10,
        limits={"sector": {"EQUITY": 0.8, "CRYPTO": 0.2}},
    ).run(
        returns=returns,
        asset_features=features,
        factor_snapshot=factors,
        metadata=metadata,
        macro_features={"vix": 18, "pmi": 54, "spx_momentum": 0.08, "inflation_trend": -0.2},
    )
    assert report["regime"]["regime"] == "EXPANSION_DISINFLATION"
    assert len(report["pipeline"]) == 8
    assert report["capabilities"]["capability_count"] == 33
    assert all(not proposal["order_created"] for proposal in report["proposals"])
    btc = next(item for item in report["proposals"] if item["symbol"] == "BTC")
    assert "COMPLIANCE_NOT_ELIGIBLE" in btc["blockers"]
    assert report["manual_approval_required"]
    assert not report["submission_allowed"]
    assert report["broker_calls"] == report["broker_writes"] == 0


def test_feedback_measures_forecast_error_without_automatic_model_update() -> None:
    decisions = pd.DataFrame({"symbol": ["A", "B", "C"], "expected_alpha": [0.03, 0.01, -0.02]})
    report = manager_feedback(decisions, pd.Series({"A": 0.02, "B": 0.015, "C": -0.01}))
    assert report["observations"] == 3
    assert not report["model_update_automatic"]
    assert report["human_review_required"]
