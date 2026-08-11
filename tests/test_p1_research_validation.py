from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stocks.research.performance_validation import independent_performance_metrics
from stocks.portfolio.intelligence import build_cross_asset_intelligence
from stocks.research.stage0 import vectorized_stage0_score
from stocks.research.strategy_inventory import publish_strategy_inventory
from stocks.research.walk_forward import (
    build_walk_forward_manifest,
    lookahead_protection_audit,
)


def test_stage0_is_fast_approximation_not_promotion() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.date_range("2025-01-01", periods=100),
            "close": [100 + index * 0.5 for index in range(100)],
        }
    )
    score = vectorized_stage0_score(frame)
    assert 0 <= score["score"] <= 1
    assert score["data_timestamp"].startswith("2025-")
    assert set(score["components"]) == {
        "trend", "medium_momentum", "short_momentum", "breakout", "volatility_quality"
    }


def test_standard_walk_forward_manifest_is_causal_and_immutable() -> None:
    manifest = build_walk_forward_manifest(
        dataset_id="DATASET",
        strategy_id="STRATEGY",
        asset_universe=["BBB", "AAA", "AAA"],
        asset_class="EQUITY",
        timeframe="1d",
        fold_id="F01",
        train_start=date(2018, 1, 1),
        train_end=date(2022, 12, 31),
        validation_start=date(2023, 1, 6),
        validation_end=date(2023, 12, 31),
        test_start=date(2024, 1, 6),
        test_end=date(2024, 12, 31),
        purge_days=5,
        embargo_days=5,
        selected_parameters={"lookback": 50},
        cost_assumptions={"model": "SHARED_TRANSACTION_COST_MODEL_V1"},
        slippage_assumptions={"model": "SHARED_TRANSACTION_COST_MODEL_V1"},
        universe_assumptions={"point_in_time": True},
    )
    assert manifest["asset_universe"] == ["AAA", "BBB"]
    assert manifest["train_end"] < manifest["validation_start"] < manifest["test_start"]
    assert len(manifest["content_hash"]) == 64
    with pytest.raises(ValueError, match="NON_CAUSAL"):
        build_walk_forward_manifest(
            dataset_id="D",
            strategy_id="S",
            asset_universe=["A"],
            asset_class="EQUITY",
            timeframe="1d",
            fold_id="F",
            train_start=date(2024, 1, 1),
            train_end=date(2024, 12, 31),
            validation_start=date(2024, 6, 1),
            validation_end=date(2025, 1, 1),
            test_start=date(2025, 2, 1),
            test_end=date(2025, 3, 1),
            purge_days=0,
            embargo_days=0,
            selected_parameters={},
            cost_assumptions={},
            slippage_assumptions={},
            universe_assumptions={},
        )


def test_lookahead_and_pit_timestamp_contracts_fail_noisy() -> None:
    audit = lookahead_protection_audit()
    assert audit["status"] == "GO"
    assert audit["checks"]["fundamentals_use_publicly_available_date_not_period_end"]
    assert not audit["period_end_date_is_public_availability"]
    assert not audit["complete_pit_coverage_claimed"]


def test_independent_performance_metrics_are_deterministic() -> None:
    returns = pd.Series([0.01, -0.005, 0.012, -0.004, 0.006] * 60)
    first = independent_performance_metrics(
        returns, bootstrap_samples=100, seed=7
    )
    second = independent_performance_metrics(
        returns, bootstrap_samples=100, seed=7
    )
    assert first == second
    assert first["status"] == "GO"
    assert first["maximum_drawdown"] < 0
    assert first["profit_factor"] > 1
    assert first["bootstrap"]["samples"] == 100


def test_cross_asset_intelligence_uses_real_breadth_and_pairwise_returns(
    tmp_path: Path,
) -> None:
    stage0 = {
        "breadth_by_asset_class": {
            "EQUITY": {
                "above_sma_50_ratio": 0.65,
                "medium_momentum_positive_ratio": 0.62,
                "median_annualized_volatility": 0.25,
            },
            "ETF": {"above_sma_50_ratio": 0.55},
            "COMMODITY_EXPOSURE": {
                "medium_momentum_positive_ratio": 0.45
            },
        },
        "cross_asset_leadership": [
            {"asset_class": "ETF", "leadership_rank": 1},
            {"asset_class": "EQUITY", "leadership_rank": 2},
            {"asset_class": "COMMODITY_EXPOSURE", "leadership_rank": 3},
        ],
        "context_benchmarks": [
            {
                "symbol": "GLD",
                "data_timestamp": "2026-08-08T00:00:00+00:00",
                "momentum_20": 0.05,
                "momentum_63": 0.10,
            },
            {
                "symbol": "SPY",
                "data_timestamp": "2026-08-08T00:00:00+00:00",
                "momentum_20": 0.02,
                "momentum_63": 0.03,
            },
        ],
    }
    macro = {
        "regime": {
            "market_regime": "RISK_ON",
            "liquidity_regime": "EXPANDING",
        }
    }
    report = build_cross_asset_intelligence(
        tmp_path, stage0=stage0, macro=macro
    )
    gold = report["relative_strength_comparisons"][0]
    assert report["regime_dimensions"]["EQUITY_TREND"] == "STRONG"
    assert gold["status"] == "AVAILABLE"
    assert gold["leader"] == "GLD"
    assert not report["breadth_is_automatic_hard_veto"]
    models = report["commodity_family_models"]
    assert set(models) == {
        "GOLD", "SILVER", "COPPER", "URANIUM", "OIL", "ENERGY"
    }
    assert len({row["model"] for row in models.values()}) == 6
    assert all(
        not row["same_model_as_other_commodities"]
        for row in models.values()
    )
    assert models["GOLD"]["status"] == "DATA_AVAILABLE"
    assert models["URANIUM"]["status"] == "DATA_UNAVAILABLE"


def test_strategy_inventory_computes_oos_pnl_correlation(tmp_path: Path) -> None:
    result_path = tmp_path / "output/research/results/portfolio_results.csv"
    result_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy_id": "S1",
                "asset_class": "EQUITY",
                "combined_oos_CAGR": 0.10,
                "robust_pass": True,
                "research_pass": True,
                "deployment_blockers": "INDEPENDENT_FORWARD_SESSION_MISSING",
            },
            {
                "strategy_id": "S2",
                "asset_class": "ETF",
                "combined_oos_CAGR": 0.05,
                "robust_pass": True,
                "research_pass": True,
                "deployment_blockers": "INDEPENDENT_FORWARD_SESSION_MISSING",
            },
        ]
    ).to_csv(result_path, index=False)
    returns_path = (
        tmp_path / "output/research/phase11_13/daily-oos-returns.parquet"
    )
    returns_path.parent.mkdir(parents=True)
    dates = pd.date_range("2025-01-01", periods=60)
    rows = [
        {
            "strategy_id": strategy,
            "fold_id": "F1",
            "cost_bps": 10.0,
            "date": timestamp,
            "daily_return": (index % 7 - 3) / divisor,
        }
        for strategy, divisor in (("S1", 1000), ("S2", 1500))
        for index, timestamp in enumerate(dates)
    ]
    pd.DataFrame(rows).to_parquet(returns_path, index=False)
    phase14_path = (
        tmp_path / "output/research/phase11_14/oos-returns.parquet"
    )
    phase14_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy_id": "S3",
                "fold_id": "F1",
                "cost_bps": 10.0,
                "date": timestamp,
                "daily_return": (index % 5 - 2) / 2000,
            }
            for index, timestamp in enumerate(dates)
        ]
    ).to_parquet(phase14_path, index=False)
    report = publish_strategy_inventory(tmp_path)
    assert report["strategy_pnl_series_count"] == 3
    assert all(
        row["strategy_pnl_correlation"]["status"] == "GO"
        for row in report["strategies"]
    )
    assert (
        tmp_path / "output/research/p1/strategy-pnl-correlation.parquet"
    ).is_file()
