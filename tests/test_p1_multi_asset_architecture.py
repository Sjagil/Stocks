from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pandas as pd

from stocks.costs import (
    SharedTransactionCostModel,
    estimate_transaction_cost,
    whole_share_economics,
)
from stocks.portfolio.coverage import (
    build_coverage_waterfall,
    normalize_asset_class,
    normalize_asset_subclass,
    normalize_shariah_status,
)
from stocks.portfolio.attribution import publish_performance_attribution
from stocks.portfolio.etf_holdings import holdings_overlap_report
from stocks.portfolio.overlap import (
    build_overlap_report,
    evaluate_strategy_overlap_promotion,
)
from stocks.portfolio.p1 import capital_size_coherence, recovery_cohort
from stocks.portfolio.targets import evaluate_rotations


def test_asset_class_subclass_and_shariah_identity() -> None:
    assert normalize_asset_class({"instrument_type": "STOCK"}) == "EQUITY"
    assert normalize_asset_class({"instrument_type": "ETF"}) == "ETF"
    commodity = {
        "instrument_type": "COMMODITY_EXPOSURE",
        "product_structure": "PHYSICAL_BACKED",
    }
    assert normalize_asset_class(commodity) == "COMMODITY_EXPOSURE"
    assert normalize_asset_subclass(commodity) == "PHYSICAL_BACKED_ETF"
    assert normalize_shariah_status("SHARIAH_ELIGIBLE_PIT") == "SHARIAH_ALLOWED"
    assert normalize_shariah_status("SHARIAH_INELIGIBLE") == "SHARIAH_BLOCKED"
    assert normalize_shariah_status("SHARIAH_DATA_INCOMPLETE") == "SHARIAH_REVIEW_REQUIRED"
    assert normalize_shariah_status(None) == "SHARIAH_DATA_MISSING"


def test_coverage_waterfall_separates_research_from_execution(tmp_path: Path) -> None:
    universe = pd.DataFrame(
        [
            _instrument("AAA", "STOCK", "SHARIAH_ELIGIBLE_PIT"),
            _instrument("ETF1", "ETF", None),
            _instrument("GOLD", "COMMODITY_EXPOSURE", None),
            {**_instrument("OLD", "STOCK", "SHARIAH_BLOCKED"), "active_listing": False},
        ]
    )
    _parquet(tmp_path / "output/universe/instruments.parquet", universe)
    contracts = pd.DataFrame([{"symbol": "AAA", "con_id": 1}])
    _parquet(tmp_path / "output/ibkr/contracts/stocks.parquet", contracts)
    for symbol in ("AAA", "ETF1", "GOLD"):
        _parquet(
            tmp_path / f"data/research/critical_trading/yfinance/{symbol}.parquet",
            pd.DataFrame(
                {
                    "session_date": pd.date_range("2025-01-01", periods=80),
                    "close": range(80),
                }
            ),
        )
    report = build_coverage_waterfall(
        tmp_path,
        signals=[{"ticker": "AAA"}, {"ticker": "ETF1"}],
        ranked=[
            {
                "ticker": "AAA",
                "research_allocation_eligible": True,
                "deployment_eligible": False,
            }
        ],
        portfolio_symbols=["AAA"],
        whole_share_symbols=["AAA"],
    )
    assert report["status"] == "GO"
    all_stages = report["funnels"]["ALL"]["stages"]
    assert all_stages["data_analyzable"] == 3
    assert all_stages["broker_resolvable"] == 1
    assert all_stages["execution_preflight_eligible"] == 0
    detail = pd.read_parquet(tmp_path / "output/portfolio/coverage-waterfall.parquet")
    etf = detail.loc[detail["symbol"] == "ETF1"].iloc[0]
    assert bool(etf["research_eligible"])
    assert not bool(etf["shariah_eligible"])
    assert etf["shariah_status"] == "SHARIAH_DATA_MISSING"


def test_shared_costs_and_whole_share_cash_competition() -> None:
    model = SharedTransactionCostModel()
    one_way = estimate_transaction_cost(
        Decimal("1000"), model=model, round_trip=False
    )
    round_trip = estimate_transaction_cost(Decimal("1000"), model=model)
    assert Decimal(round_trip["total_cost_eur"]) > Decimal(one_way["total_cost_eur"])
    feasible = whole_share_economics(
        desired_notional_eur=Decimal("300"),
        price_eur=Decimal("25"),
        risk_budget_eur=Decimal("20"),
        risk_per_share_eur=Decimal("2"),
        available_cash_eur=Decimal("300"),
        expected_gross_return=Decimal("0.10"),
        currency="EUR",
        model=model,
    )
    infeasible = whole_share_economics(
        desired_notional_eur=Decimal("300"),
        price_eur=Decimal("1000"),
        risk_budget_eur=Decimal("20"),
        risk_per_share_eur=Decimal("80"),
        available_cash_eur=Decimal("300"),
        expected_gross_return=Decimal("0.10"),
        currency="EUR",
        model=model,
    )
    assert feasible["execution_candidate_status"] == "EXECUTABLE_WHOLE_SHARE"
    assert infeasible["execution_candidate_status"] == "NON_EXECUTABLE_WHOLE_SHARE"


def test_correlation_clusters_and_strategy_incremental_gate() -> None:
    opportunities = [
        {"symbol": "NVDA", "asset_class": "EQUITY", "correlation_cluster": "SEMIS"},
        {"symbol": "SOXX", "asset_class": "ETF", "correlation_cluster": "SEMIS"},
        {"symbol": "IAU", "asset_class": "COMMODITY_EXPOSURE", "correlation_cluster": "GOLD"},
    ]
    correlation = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]], index=["NVDA", "SOXX"], columns=["NVDA", "SOXX"]
    )
    report = build_overlap_report(opportunities, correlation, threshold=0.75)
    assert report["status"] == "GO"
    assert report["major_clusters"][0]["cluster"] == "SEMIS"
    assert report["high_correlation_pairs"][0]["correlation"] == 0.9
    candidate = pd.Series([0.01, -0.01, 0.02, -0.02])
    existing = pd.DataFrame({"duplicate": candidate})
    gate = evaluate_strategy_overlap_promotion(
        candidate,
        existing,
        standalone_expectancy=0.01,
        maximum_correlation=0.75,
        minimum_incremental_expectancy=0.001,
    )
    assert gate["status"] == "NO_GO"
    assert "STRATEGY_OVERLAP_EXCEEDS_INCREMENTAL_VALUE" in gate["blockers"]


def test_etf_top_holdings_look_through_reports_fund_and_direct_overlap() -> None:
    holdings = pd.DataFrame(
        [
            {"etf_symbol": "ETF1", "holding_symbol": "AAA", "weight": 0.3},
            {"etf_symbol": "ETF1", "holding_symbol": "BBB", "weight": 0.2},
            {"etf_symbol": "ETF2", "holding_symbol": "AAA", "weight": 0.25},
            {"etf_symbol": "ETF2", "holding_symbol": "CCC", "weight": 0.25},
        ]
    )
    report = holdings_overlap_report(
        [
            {"symbol": "AAA", "asset_class": "EQUITY"},
            {"symbol": "ETF1", "asset_class": "ETF"},
        ],
        holdings,
        threshold=0.2,
    )
    assert report["status"] == "TOP_HOLDINGS_LOOK_THROUGH_AVAILABLE"
    assert report["fund_pairs"][0]["left"] == "ETF1"
    assert report["direct_stock_overlaps"] == [
        {"etf": "ETF1", "equity": "AAA", "holding_weight": 0.3},
        {"etf": "ETF2", "equity": "AAA", "holding_weight": 0.25},
    ]
    assert not report["complete_look_through_claimed"]


def test_rotation_requires_material_net_and_cluster_improvement() -> None:
    policy = {
        "rotation": {
            "minimum_expected_net_return_improvement": 0.02,
            "minimum_score_improvement": 0.10,
        },
        "ranking": {"replacement_improvement": 0.10},
    }
    held = {
        "symbol": "AAA",
        "asset_class": "EQUITY",
        "research_eligible": True,
        "expected_net_return": 0.03,
        "confidence": 0.50,
        "correlation_cluster": "A",
    }
    tiny = {
        "symbol": "BBB",
        "asset_class": "ETF",
        "research_eligible": True,
        "expected_net_return": 0.031,
        "confidence": 0.51,
        "correlation_cluster": "B",
    }
    strong = {
        "symbol": "CCC",
        "asset_class": "COMMODITY_EXPOSURE",
        "research_eligible": True,
        "expected_net_return": 0.08,
        "confidence": 0.70,
        "correlation_cluster": "C",
    }
    keep = evaluate_rotations(
        current_positions=[{"symbol": "AAA"}],
        opportunities=[held, tiny],
        policy=policy,
    )
    rotate = evaluate_rotations(
        current_positions=[{"symbol": "AAA"}],
        opportunities=[held, strong],
        policy=policy,
    )
    assert keep[0]["action"] == "KEEP_CURRENT_POSITION"
    assert rotate[0]["action"] == "ROTATE"
    assert rotate[0]["transaction_costs_included_in_expected_net"]


def test_capital_coherence_and_recovery_cohort_filters(tmp_path: Path) -> None:
    (tmp_path / "config/costs").mkdir(parents=True)
    (tmp_path / "config/costs/shared_transaction_cost_v1.json").write_text(
        json.dumps(
            {
                "base_currency": "EUR",
                "commission": {"minimum_per_order_eur": 0.35, "variable_bps": 0},
                "exchange_fees_bps": 0,
                "half_spread_bps": 1,
                "slippage_bps": 5,
                "market_impact_bps": 1,
                "fx_conversion_bps": 10,
                "minimum_practical_trade_eur": 5,
                "round_trip_for_opportunity_economics": True,
            }
        ),
        encoding="utf-8",
    )
    report = capital_size_coherence(tmp_path)
    assert [row["capital_eur"] for row in report["scenarios"]] == [
        1000, 1870, 2500, 5000, 10000, 25000, 50000
    ]
    universe = pd.DataFrame(
        [
            _security_master("GOOD", "Technology", "Software", "Good Corp"),
            _security_master("SHELL", "Financial Services", "Shell Companies", "Shell"),
            _security_master("UNITU", "Technology", "Software", "Example Units"),
            {
                **_security_master("GLD", "GOLD", "Commodity", "GLD"),
                "instrument_type": "COMMODITY_EXPOSURE",
                "discovery_source": "BROAD_MULTI_ASSET_V1",
            },
        ]
    )
    _parquet(tmp_path / "output/universe/instruments.parquet", universe)
    cohort = recovery_cohort(tmp_path, per_sector=35)
    assert "GOOD" in cohort["symbols"]
    assert "GLD" in cohort["symbols"]
    assert "SHELL" not in cohort["symbols"]
    assert "UNITU" not in cohort["symbols"]
    assert not cohort["tradeable_assumed"]


def test_performance_attribution_is_rebuildable_read_model(
    tmp_path: Path,
) -> None:
    _parquet(
        tmp_path / "output/universe/instruments.parquet",
        pd.DataFrame([_instrument("ON", "STOCK", "SHARIAH_ELIGIBLE_PIT")]),
    )
    database = (
        tmp_path / "data/execution/phase9/private/paper_execution.sqlite3"
    )
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE executions (
              exec_identity TEXT PRIMARY KEY, intent_id TEXT,
              payload_hash TEXT, payload_json TEXT, created_at TEXT
            );
            CREATE TABLE commissions (
              commission_identity TEXT PRIMARY KEY, exec_identity TEXT,
              payload_hash TEXT, payload_json TEXT, created_at TEXT
            );
            CREATE TABLE intents (
              intent_id TEXT PRIMARY KEY, economic_order_key TEXT,
              payload_hash TEXT, payload_json TEXT, created_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO intents VALUES (?,?,?,?,?)",
            (
                "I1", "E1", "H1",
                json.dumps(
                    {
                        "symbol": "ON", "intent_source": "RESEARCH",
                        "side": "BUY",
                    }
                ),
                "2026-08-01T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO executions VALUES (?,?,?,?,?)",
            (
                "X1", "I1", "H2",
                json.dumps(
                    {
                        "symbol": "ON", "quantity": "1", "price": "80",
                        "fx_rate": "0.9", "side": "BUY",
                    }
                ),
                "2026-08-01T01:00:00+00:00",
            ),
        )
    report = publish_performance_attribution(tmp_path)
    assert report["fact_count"] == 1
    assert report["derived_read_model_only"]
    assert report["rebuildable_from_canonical_records"]
    assert not report["parallel_financial_ledger_created"]
    assert report["status"] == "PARTIAL_PENDING_REALIZED_ROUND_TRIP"


def _instrument(symbol: str, instrument_type: str, compliance: str | None) -> dict[str, object]:
    return {
        "instrument_id": symbol,
        "symbol": symbol,
        "instrument_type": instrument_type,
        "asset_type": "ETF" if instrument_type == "ETF" else "STOCK",
        "active_listing": True,
        "currency": "USD",
        "compliance_status": compliance,
        "product_structure": "PHYSICAL_BACKED" if symbol == "GOLD" else None,
    }


def _security_master(symbol: str, sector: str, industry: str, name: str) -> dict[str, object]:
    return {
        "instrument_id": symbol,
        "symbol": symbol,
        "name": name,
        "instrument_type": "STOCK",
        "asset_type": "STOCK",
        "active_listing": True,
        "currency": "USD",
        "primary_exchange": "NASDAQ",
        "sector": sector,
        "industry": industry,
        "discovery_source": "PHASE11_4_PIT_SECURITY_MASTER",
    }


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
