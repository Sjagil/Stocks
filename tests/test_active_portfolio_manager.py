from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocks.screener.config import ScreenerConfig
from stocks.execution.idempotency import stable_hash
from stocks.portfolio.manager import (
    _apply_portfolio_risk_overlay,
    _build_position_management,
    _current_positions,
    _latest_portfolio_broker_snapshot,
    _latest_price_and_stop,
    _latest_screener_records,
    _phase11_13_signals,
    _phase11_14_signals,
    _public_opportunity,
    _private_account_state,
    _record_action_cycle,
    _resolved_shariah_status,
    _runtime_screener_expiration,
    allocate_research_portfolio,
    build_active_portfolio_report,
    build_capital_decisions,
    build_opportunity_funnel,
    build_private_sizing,
    position_actions,
    rank_opportunities,
)


def test_only_runtime_verified_signal_can_supply_shariah_status() -> None:
    incomplete = {"shariah_status": "SHARIAH_DATA_INCOMPLETE"}

    assert _resolved_shariah_status(
        incomplete,
        [{"shariah_status": "SHARIAH_ELIGIBLE_PIT"}],
    ) == "SHARIAH_DATA_INCOMPLETE"
    assert _resolved_shariah_status(
        incomplete,
        [
            {
                "shariah_status": "SHARIAH_ELIGIBLE_PIT",
                "_shariah_runtime_verified_source": True,
            }
        ],
    ) == "SHARIAH_ELIGIBLE_PIT"


def test_new_runtime_attestation_supersedes_only_stale_dynamic_shariah_blocker(
) -> None:
    signal = {
        **_signal("BROAD-SCREENER-V1", "broad_screening_rank", "1d"),
        "shariah_status": "SHARIAH_ELIGIBLE_PIT",
        "_shariah_runtime_verified_source": True,
    }
    report = rank_opportunities(
        [signal],
        policy=_policy(),
        contracts={"AAPL": {"con_id": 1, "currency": "USD"}},
        fundamentals={
            "AAPL": {
                "fundamental_score": 80,
                "liquidity_score": 100,
                "shariah_status": "SHARIAH_DATA_INCOMPLETE",
            }
        },
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={
            "AAPL": {
                "action": "AVOID",
                "risk_blockers": [
                    "SHARIAH_ATTESTATION_REQUIRED",
                    "MATERIAL_NEWS_RISK_REVIEW_REQUIRED",
                ],
            }
        },
    )

    candidate = report[0]
    assert candidate["shariah_status"] == "SHARIAH_ELIGIBLE_PIT"
    assert "SHARIAH_ATTESTATION_REQUIRED" not in candidate[
        "research_allocation_blockers"
    ]
    assert "MATERIAL_NEWS_RISK_REVIEW_REQUIRED" in candidate[
        "research_allocation_blockers"
    ]


def test_runtime_screener_expiration_is_bounded_by_shariah_expiry() -> None:
    item = {
        "decision_time": "2026-08-08T20:00:00+00:00",
        "data_timestamps": {
            "shariah_expires_at": "2026-08-09T12:00:00+00:00"
        },
    }

    assert _runtime_screener_expiration(item) == (
        "2026-08-09T12:00:00+00:00"
    )


def test_screened_signal_price_uses_validated_multitimeframe_daily_cache(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=SPUS/interval=1d/source_interval=1d"
    )
    path.mkdir(parents=True)
    timestamps = pd.date_range("2026-07-01", periods=30, freq="B", tz="UTC")
    close = np.linspace(50.0, 60.0, len(timestamps))
    pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adjusted_close": close,
            "quality_status": "VALIDATED_OHLC",
            "is_partial": False,
        }
    ).to_parquet(path / "bars.parquet", index=False)

    price, stop = _latest_price_and_stop(tmp_path, "SPUS")

    assert price == 60.0
    assert 0 < stop < price


def test_screener_preview_requires_current_config_hash(tmp_path: Path) -> None:
    config_root = Path(__file__).resolve().parents[1] / "config"
    shutil.copytree(config_root, tmp_path / "config")
    current_hash = ScreenerConfig.load(tmp_path).config_hash
    preview = tmp_path / "output/screener/latest-preview.json"
    preview.parent.mkdir(parents=True)
    preview.write_text(
        json.dumps(
            {
                "status": "GO",
                "screening_date": "2026-08-07",
                "config_hash": "STALE",
                "canonical_research_evidence": False,
                "append_only_history_mutated": False,
                "records": [{"symbol": "STALE"}],
            }
        ),
        encoding="utf-8",
    )
    canonical = tmp_path / "output/screener/2026-08-06/screening-results.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps(
            {
                "screening_date": "2026-08-06",
                "records": [{"symbol": "CANONICAL"}],
            }
        ),
        encoding="utf-8",
    )

    records, source = _latest_screener_records(tmp_path)

    assert current_hash != "STALE"
    assert [row["symbol"] for row in records] == ["CANONICAL"]
    assert source is not None
    assert Path(source).name == "screening-results.json"


def _write_broker_snapshot(
    path: Path,
    *,
    snapshot_hash: str,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE snapshots("
            "snapshot_id TEXT, snapshot_hash TEXT, payload_json TEXT, "
            "created_at TEXT)"
        )
        connection.execute(
            "INSERT INTO snapshots VALUES(?,?,?,?)",
            (
                "S1",
                snapshot_hash,
                json.dumps(payload),
                "2026-07-31T08:00:00+00:00",
            ),
        )


def _complete_live_snapshot() -> dict[str, object]:
    snapshot = {
        "account": {
            "status": "COMPLETE",
            "values": [
                {
                    "tag": "NetLiquidation",
                    "value": "50000",
                    "currency": "EUR",
                },
                {
                    "tag": "AvailableFunds",
                    "value": "10000",
                    "currency": "EUR",
                },
                {
                    "tag": "BuyingPower",
                    "value": "10000",
                    "currency": "EUR",
                },
                {
                    "tag": "TotalCashValue",
                    "value": "10000",
                    "currency": "EUR",
                },
                {
                    "tag": "GrossPositionValue",
                    "value": "0",
                    "currency": "EUR",
                },
            ],
        },
        "positions": {"status": "EMPTY_COMPLETE", "positions": []},
    }
    for item in snapshot["account"]["values"]:
        item["account_fingerprint"] = "HASHED-ACCOUNT"
        item["source"] = "IBKR_ACCOUNT_SUMMARY"
        item["observed_at"] = "2026-07-31T08:00:00+00:00"
    return snapshot


def test_hash_verified_live_snapshot_is_portfolio_source(
    tmp_path: Path,
) -> None:
    live_hash = "LIVE-HASH"
    live_db = (
        tmp_path
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    _write_broker_snapshot(
        live_db,
        snapshot_hash=live_hash,
        payload=_complete_live_snapshot(),
    )
    public = tmp_path / "output/ibkr/live"
    public.mkdir(parents=True)
    (public / "reconciliation.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "reconciliation_status": "LIVE_RECONCILED_EMPTY",
                "private_snapshot_hash": live_hash,
            }
        ),
        encoding="utf-8",
    )

    snapshot = _latest_portfolio_broker_snapshot(tmp_path)
    account = _private_account_state(tmp_path)
    positions, position_status = _current_positions(tmp_path)

    assert snapshot is not None
    assert snapshot["observation_environment"] == "LIVE_READ_ONLY"
    assert snapshot["content_hash"] == live_hash
    assert account["status"] == "GO"
    assert account["snapshot_hash"] == live_hash
    assert positions == []
    assert position_status == "PRIVATE_BROKER_POSITION_SNAPSHOT_COMPLETE"


def test_live_snapshot_hash_mismatch_falls_back_fail_closed(
    tmp_path: Path,
) -> None:
    live_db = (
        tmp_path
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    _write_broker_snapshot(
        live_db,
        snapshot_hash="ACTUAL-HASH",
        payload=_complete_live_snapshot(),
    )
    public = tmp_path / "output/ibkr/live"
    public.mkdir(parents=True)
    (public / "reconciliation.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "reconciliation_status": "LIVE_RECONCILED_EMPTY",
                "private_snapshot_hash": "DIFFERENT-HASH",
            }
        ),
        encoding="utf-8",
    )

    assert _latest_portfolio_broker_snapshot(tmp_path) is None
    assert (
        _private_account_state(tmp_path)["status"]
        == "PRIVATE_BROKER_SNAPSHOT_UNAVAILABLE"
    )
    positions, status = _current_positions(tmp_path)
    assert positions == []
    assert status == "PRIVATE_BROKER_SNAPSHOT_UNAVAILABLE"


def test_recent_complete_live_account_can_size_research_only(
    tmp_path: Path,
) -> None:
    payload = _complete_live_snapshot()
    for item in payload["account"]["values"]:
        item["account_fingerprint"] = "HASHED-ACCOUNT"
        item["source"] = "IBKR_ACCOUNT_SUMMARY"
    payload["snapshot_completed_at"] = datetime.now(UTC).isoformat()
    live_db = (
        tmp_path
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    _write_broker_snapshot(
        live_db,
        snapshot_hash=stable_hash(payload),
        payload=payload,
    )

    account = _private_account_state(tmp_path)
    positions, position_status = _current_positions(tmp_path)

    assert account["status"] == "GO"
    assert account["net_liquidation_eur"] == "50000"
    assert account["account_authority"] == "RESEARCH_ONLY_LAST_OBSERVED"
    assert account["snapshot_source"] == "LIVE_RESEARCH_ONLY"
    assert account["fresh_reconciliation"] is False
    assert account["execution_eligible"] is False
    assert positions == []
    assert position_status == "PRIVATE_BROKER_SNAPSHOT_UNAVAILABLE"


def test_research_account_fallback_rejects_payload_hash_tampering(
    tmp_path: Path,
) -> None:
    payload = _complete_live_snapshot()
    for item in payload["account"]["values"]:
        item["account_fingerprint"] = "HASHED-ACCOUNT"
        item["source"] = "IBKR_ACCOUNT_SUMMARY"
    payload["snapshot_completed_at"] = datetime.now(UTC).isoformat()
    live_db = (
        tmp_path
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    _write_broker_snapshot(
        live_db,
        snapshot_hash="TAMPERED-HASH",
        payload=payload,
    )

    assert (
        _private_account_state(tmp_path)["status"]
        == "PRIVATE_BROKER_SNAPSHOT_UNAVAILABLE"
    )


def test_incomplete_live_positions_never_become_reconciled_empty(
    tmp_path: Path,
) -> None:
    payload = _complete_live_snapshot()
    payload["positions"] = {"status": "TIMEOUT", "positions": []}
    live_db = (
        tmp_path
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    _write_broker_snapshot(
        live_db,
        snapshot_hash="LIVE-HASH",
        payload=payload,
    )
    public = tmp_path / "output/ibkr/live"
    public.mkdir(parents=True)
    (public / "reconciliation.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "reconciliation_status": "LIVE_RECONCILED_EMPTY",
                "private_snapshot_hash": "LIVE-HASH",
            }
        ),
        encoding="utf-8",
    )

    positions, status = _current_positions(tmp_path)
    assert positions == []
    assert status == "PRIVATE_BROKER_POSITION_SNAPSHOT_INCOMPLETE"


def _policy() -> dict[str, object]:
    path = (
        Path(__file__).parents[1]
        / "config"
        / "portfolio"
        / "active_manager_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _signal(
    strategy_id: str,
    entry_strategy: str,
    timeframe: str,
    *,
    ticker: str = "AAPL",
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "strategy_id": strategy_id,
        "entry_strategy": entry_strategy,
        "timeframe": timeframe,
        "action": "BUY",
        "confidence_score": "0.80",
        "data_freshness": "FRESH",
        "data_timestamp": "2026-07-29T00:00:00+00:00",
        "preferred_entry": "100",
        "stop_loss": "95",
        "take_profit_1": "110",
        "take_profit_2": "115",
        "stop_distance_pct": "0.05",
        "currency": "USD",
        "reasons": [
            "HIGHER_TIMEFRAME_GATE_CONFIRMED",
            "POSITIVE_RELATIVE_MOMENTUM",
        ],
        "_portfolio_eligible_source": True,
    }


def _ranked(ticker: str, score: float, stop: float = 0.05) -> dict[str, object]:
    return {
        "ticker": ticker,
        "asset_type": "STOCK",
        "sleeve": "stock",
        "sector": "Technology",
        "region": "UNITED_STATES",
        "opportunity_score": score,
        "execution_blockers": [],
        "stop_risk_pct": stop,
    }


def test_ranking_recognizes_distinct_indicator_families_and_timeframes() -> None:
    signals = [
        _signal("S1", "rsi_pullback", "1h"),
        _signal("S2", "vwap_deviation_reversion", "4h"),
        _signal("S3", "donchian_breakout", "1d"),
        _signal("S4", "quality_momentum", "1w"),
    ]
    report = rank_opportunities(
        signals,
        policy=_policy(),
        contracts={
            "AAPL": {
                "con_id": 1,
                "currency": "USD",
                "primary_exchange": "NASDAQ",
            }
        },
        fundamentals={
            "AAPL": {
                "fundamental_score": 80,
                "liquidity_score": 100,
                "sector": "Technology",
                "shariah_status": "SHARIAH_ELIGIBLE_PIT",
            }
        },
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
    )

    candidate = report[0]
    assert candidate["strategy_ids"] == ["S1", "S2", "S3", "S4"]
    assert candidate["strategy_families"] == [
        "donchian_breakout",
        "quality_momentum",
        "rsi_pullback",
        "vwap_deviation_reversion",
    ]
    assert candidate["timeframes"] == ["1d", "1h", "1w", "4h"]
    assert candidate["opportunity_class"] == "TREND"
    assert candidate["active_swing_timeframe_context"]["score"] == 0.8
    assert candidate["active_swing_timeframe_context"][
        "higher_timeframe_policy"
    ] == "MULTIPLIER_NOT_AUTOMATIC_VETO"
    assert candidate["execution_blockers"] == []


def test_dynamic_strategy_weight_affects_opportunity_ranking() -> None:
    signals = [
        _signal("HIGH_WEIGHT", "quality_momentum", "4h", ticker="AAA"),
        _signal("LOW_WEIGHT", "quality_momentum", "4h", ticker="BBB"),
    ]
    contracts = {
        ticker: {"con_id": index, "currency": "USD"}
        for index, ticker in enumerate(("AAA", "BBB"), 1)
    }
    fundamentals = {
        ticker: {
            "fundamental_score": 80,
            "liquidity_score": 100,
            "shariah_status": "SHARIAH_ELIGIBLE_PIT",
        }
        for ticker in ("AAA", "BBB")
    }

    report = rank_opportunities(
        signals,
        policy=_policy(),
        contracts=contracts,
        fundamentals=fundamentals,
        family_map={
            "HIGH_WEIGHT": "quality_momentum",
            "LOW_WEIGHT": "quality_momentum",
        },
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
        strategy_weights={"HIGH_WEIGHT": 0.20, "LOW_WEIGHT": 0.02},
    )

    by_ticker = {row["ticker"]: row for row in report}
    assert by_ticker["AAA"]["opportunity_score"] > by_ticker["BBB"][
        "opportunity_score"
    ]
    assert by_ticker["AAA"]["strategy_allocation"]["status"] == (
        "DYNAMIC_WEIGHTED"
    )
    assert by_ticker["AAA"]["strategy_allocation"][
        "participating_weight"
    ] == 0.2


def test_market_context_is_low_weight_context_not_authority() -> None:
    signals = [
        _signal("S1", "quality_momentum", "4h", ticker="AAA"),
        _signal("S2", "quality_momentum", "4h", ticker="BBB"),
    ]
    contracts = {
        ticker: {"con_id": index, "currency": "USD"}
        for index, ticker in enumerate(("AAA", "BBB"), 1)
    }
    fundamentals = {
        ticker: {
            "fundamental_score": 80,
            "liquidity_score": 100,
            "shariah_status": "SHARIAH_ELIGIBLE_PIT",
        }
        for ticker in ("AAA", "BBB")
    }
    report = rank_opportunities(
        signals,
        policy=_policy(),
        contracts=contracts,
        fundamentals=fundamentals,
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
        market_context={
            "AAA": {
                "status": "CONTEXT_AVAILABLE",
                "ranking_components": {
                    "orderflow_context": 0.65,
                    "gex_context": 0.60,
                },
                "standalone_entry_authority": False,
                "execution_authority": "NONE",
            },
            "BBB": {
                "status": "CONTEXT_AVAILABLE",
                "ranking_components": {
                    "orderflow_context": 0.35,
                    "gex_context": 0.40,
                },
                "standalone_entry_authority": False,
                "execution_authority": "NONE",
            },
        },
    )
    by_ticker = {row["ticker"]: row for row in report}
    assert by_ticker["AAA"]["opportunity_score"] > by_ticker["BBB"][
        "opportunity_score"
    ]
    assert by_ticker["AAA"]["components"]["orderflow_context"] == 0.65
    assert by_ticker["AAA"]["market_context"]["execution_authority"] == "NONE"
    assert "EXECUTION_AUTHORITY_NONE" in by_ticker["AAA"][
        "deployment_blockers"
    ]


def test_asset_specific_macro_transmission_changes_confluence_ranking() -> None:
    signals = [
        _signal("S1", "quality_momentum", "4h", ticker="AAA"),
        _signal("S2", "quality_momentum", "4h", ticker="BBB"),
    ]
    contracts = {
        ticker: {"con_id": index, "currency": "USD"}
        for index, ticker in enumerate(("AAA", "BBB"), 1)
    }
    fundamentals = {
        ticker: {
            "fundamental_score": 80,
            "liquidity_score": 100,
            "shariah_status": "SHARIAH_ELIGIBLE_PIT",
        }
        for ticker in ("AAA", "BBB")
    }
    asset_context = {
        "AAA": {
            "transmission_group": "technology_equity",
            "components": {
                "macro": {
                    "status": "AVAILABLE",
                    "score": 0.80,
                    "confidence": 0.80,
                }
            },
        },
        "BBB": {
            "transmission_group": "technology_equity",
            "components": {
                "macro": {
                    "status": "AVAILABLE",
                    "score": -0.80,
                    "confidence": 0.80,
                }
            },
        },
    }

    report = rank_opportunities(
        signals,
        policy=_policy(),
        contracts=contracts,
        fundamentals=fundamentals,
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
        asset_context=asset_context,
    )

    by_ticker = {row["ticker"]: row for row in report}
    assert by_ticker["AAA"]["opportunity_score"] > by_ticker["BBB"][
        "opportunity_score"
    ]
    assert by_ticker["AAA"]["multilayer_confluence"]["macro_source"] == (
        "ASSET_SPECIFIC_MACRO_TRANSMISSION"
    )
    assert by_ticker["BBB"]["multilayer_confluence"]["status"] == (
        "MACRO_HEADWIND_RISK_REDUCTION"
    )
    assert by_ticker["AAA"]["multilayer_confluence"][
        "standalone_entry_allowed"
    ] is False


def test_missing_stock_fundamentals_block_multilayer_allocation() -> None:
    report = rank_opportunities(
        [_signal("S1", "quality_momentum", "4h", ticker="MISS")],
        policy=_policy(),
        contracts={"MISS": {"con_id": 1, "currency": "USD"}},
        fundamentals={},
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
    )

    candidate = report[0]
    assert candidate["research_allocation_eligible"] is False
    assert "MULTILAYER_FUNDAMENTAL_DATA_REQUIRED" in candidate[
        "research_allocation_blockers"
    ]
    assert candidate["multilayer_confluence"]["allocation_allowed"] is False


def test_signal_contract_currency_mismatch_is_fail_closed() -> None:
    signal = _signal("S1", "quality_momentum", "4h", ticker="ASML")
    report = rank_opportunities(
        [signal],
        policy=_policy(),
        contracts={
            "ASML": {
                "con_id": 1,
                "currency": "EUR",
                "primary_exchange": "AEB",
            }
        },
        fundamentals={
            "ASML": {
                "fundamental_score": 80,
                "liquidity_score": 100,
                "shariah_status": "SHARIAH_ELIGIBLE_PIT",
            }
        },
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
    )

    candidate = report[0]
    assert candidate["signal_currency"] == "USD"
    assert candidate["contract_currency"] == "EUR"
    assert candidate["signal_contract_currency_status"] == (
        "SIGNAL_CONTRACT_CURRENCY_MISMATCH"
    )
    assert "SIGNAL_CONTRACT_CURRENCY_MISMATCH" in candidate[
        "position_management_blockers"
    ]
    assert candidate["research_allocation_eligible"] is False
    assert candidate["deployment_eligible"] is False
    public = _public_opportunity(candidate)
    assert public["signal_contract_currency_status"] == (
        "SIGNAL_CONTRACT_CURRENCY_MISMATCH"
    )


def test_configured_provider_currency_overrides_signal_default() -> None:
    policy = _policy()
    policy["asset_metadata"]["U-UN.TO"] = {
        "asset_type": "COMMODITY_CLOSED_END_TRUST",
        "sleeve": "commodity_security",
        "currency": "CAD",
    }
    report = rank_opportunities(
        [_signal("S1", "trend", "1w", ticker="U-UN.TO")],
        policy=policy,
        contracts={
            "U-UN.TO": {
                "con_id": 503299503,
                "currency": "CAD",
                "primary_exchange": "TSE",
            }
        },
        fundamentals={},
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
    )

    candidate = report[0]
    assert candidate["signal_currency"] == "CAD"
    assert candidate["contract_currency"] == "CAD"
    assert candidate["signal_contract_currency_status"] == (
        "SIGNAL_CONTRACT_CURRENCY_MATCH"
    )
    assert "SIGNAL_CONTRACT_CURRENCY_MISMATCH" not in candidate[
        "execution_blockers"
    ]


def test_research_only_architectures_cannot_allocate() -> None:
    signal = _signal("S1", "atr_breakout", "1h")
    signal["_portfolio_eligible_source"] = False
    report = rank_opportunities(
        [signal],
        policy=_policy(),
        contracts={
            "AAPL": {
                "con_id": 1,
                "currency": "USD",
                "primary_exchange": "NASDAQ",
            }
        },
        fundamentals={
            "AAPL": {
                "fundamental_score": 80,
                "liquidity_score": 100,
                "shariah_status": "SHARIAH_ELIGIBLE_PIT",
            }
        },
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
    )
    assert (
        "RESEARCH_OBSERVER_NOT_PORTFOLIO_ELIGIBLE"
        in report[0]["execution_blockers"]
    )


def test_robustness_survivor_can_inform_research_not_deployment() -> None:
    signal = _signal("ROBUST-1", "cross_sectional_momentum", "1w")
    signal["_portfolio_eligible_source"] = False
    signal["_research_allocation_eligible_source"] = True
    signal["_deployment_eligible_source"] = False
    signal["_evidence_tier"] = "ROBUSTNESS_SURVIVOR"
    report = rank_opportunities(
        [signal],
        policy=_policy(),
        contracts={
            "AAPL": {
                "con_id": 1,
                "currency": "USD",
                "primary_exchange": "NASDAQ",
            }
        },
        fundamentals={
            "AAPL": {
                "fundamental_score": 80,
                "liquidity_score": 100,
                "shariah_status": "SHARIAH_ELIGIBLE_PIT",
            }
        },
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
    )
    candidate = report[0]
    assert candidate["research_allocation_blockers"] == []
    assert candidate["research_allocation_eligible"] is True
    assert candidate["deployment_eligible"] is False
    assert "EXECUTION_AUTHORITY_NONE" in candidate[
        "deployment_blockers"
    ]
    assert "ROBUSTNESS_SURVIVOR" in candidate["evidence_tiers"]


def test_deployment_evidence_does_not_force_research_position_exit() -> None:
    candidate = {
        **_ranked("AAPL", 0.9),
        "preferred_entry": "100",
        "stop_loss": "95",
        "take_profit_1": "120",
        "research_allocation_blockers": [],
        "position_management_blockers": [],
        "deployment_blockers": [
            "STRATEGY_DEPLOYMENT_EVIDENCE_REQUIRED"
        ],
    }
    actions = position_actions(
        [
            {
                "con_id": 1,
                "symbol": "AAPL",
                "quantity": 1.0,
                "average_cost": 90.0,
                "currency": "USD",
                "security_type": "STK",
            }
        ],
        [candidate],
        {"allocations": [candidate]},
        policy=_policy(),
        dynamic_overrides={},
        daily_target={},
    )
    assert actions[0]["advisory_action"] in {"HOLD", "ADD"}
    assert (
        "STRATEGY_DEPLOYMENT_EVIDENCE_REQUIRED"
        not in actions[0]["reason_codes"]
    )


def test_capital_decisions_make_open_and_cash_explicit() -> None:
    candidate = {
        **_ranked("AAPL", 0.8),
        "deployment_blockers": ["EXECUTION_AUTHORITY_NONE"],
    }
    report = build_capital_decisions(
        [],
        [candidate],
        {
            "allocations": [{**candidate, "target_weight": 0.08}],
            "cash_weight": 0.92,
        },
        [],
        policy=_policy(),
    )

    actions = {row["formal_action"]: row for row in report["actions"]}
    assert report["status"] == "GO"
    assert actions["OPEN"]["ticker"] == "AAPL"
    assert actions["OPEN"]["approved_target_weight"] == 0.0
    assert actions["OPEN"]["implementation_status"] == (
        "BLOCKED_DEPLOYMENT_GATES"
    )
    assert actions["CASH"]["approved_target_weight"] == 1.0
    assert "NO_EXECUTION_READY_ALTERNATIVE" in actions["CASH"][
        "reason_codes"
    ]
    assert report["opportunity_cost_is_first_class"] is True
    assert report["execution_authority"] == "NONE"


def test_phase11_13_robust_attested_weights_become_research_signals(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output/research/phase11_13"
    root.mkdir(parents=True)
    strategy_id = "WEEKLY_CROSS_SECTIONAL_MOMENTUM"
    status = {
        "status": "GO",
        "qualification_boundary": {
            "status": "FROZEN",
            "qualification_hash": "Q1",
            "robust_strategy_ids": [strategy_id],
        },
        "qualification": {
            "strategies": [
                {
                    "strategy_id": strategy_id,
                    "robust_pass": True,
                    "portfolio_invariants_go": True,
                    "combined_period_profit_factor": 1.3,
                    "combined_oos_Sharpe": 0.7,
                }
            ]
        },
    }
    observation = {
        "status": "GO",
        "observations": [
            {
                "strategy_id": strategy_id,
                "timeframe": "1w",
                "closed_bar_timestamp": datetime.now(UTC).isoformat(),
                "current_attested_target_weights": {"AAPL": 0.25},
            }
        ],
    }
    (root / "status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    (root / "latest-forward-observation.json").write_text(
        json.dumps(observation),
        encoding="utf-8",
    )
    prices = tmp_path / "data/research/critical_trading/yfinance"
    prices.mkdir(parents=True)
    close = np.linspace(80.0, 100.0, 40)
    pd.DataFrame(
        {
            "session_date": pd.date_range(
                "2026-01-01",
                periods=40,
                freq="B",
            ),
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
        }
    ).to_parquet(prices / "AAPL.parquet", index=False)

    signals = _phase11_13_signals(tmp_path)

    assert len(signals) == 1
    assert signals[0]["ticker"] == "AAPL"
    assert signals[0]["_research_allocation_eligible_source"] is True
    assert signals[0]["_deployment_eligible_source"] is False
    assert signals[0]["_evidence_tier"] == "ROBUSTNESS_SURVIVOR"


def test_phase11_14_only_robust_forward_candidates_become_signals(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output/research/phase11_14"
    root.mkdir(parents=True)
    accepted = "P1114-ACCEPTED"
    rejected = "P1114-NOT-FORWARD"
    status = {
        "status": "GO",
        "qualification_boundary": {
            "status": "FROZEN",
            "qualification_hash": "Q14",
            "robust_strategy_ids": [accepted, rejected],
        },
        "qualification": {
            "strategies": [
                {
                    "strategy_id": accepted,
                    "robust_pass": True,
                    "portfolio_invariants_go": True,
                    "forward_observer_candidate": True,
                    "combined_period_profit_factor": 1.25,
                    "combined_oos_Sharpe": 0.8,
                },
                {
                    "strategy_id": rejected,
                    "robust_pass": True,
                    "portfolio_invariants_go": True,
                    "forward_observer_candidate": False,
                    "combined_period_profit_factor": 1.5,
                    "combined_oos_Sharpe": 1.0,
                },
            ]
        },
    }
    timestamp = datetime.now(UTC).isoformat()
    observation = {
        "status": "GO",
        "observations": [
            {
                "strategy_id": strategy_id,
                "timeframe": "4h",
                "closed_bar_timestamp": timestamp,
                "current_attested_target_weights": {"AAPL": 0.25},
            }
            for strategy_id in (accepted, rejected)
        ],
    }
    (root / "status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    (root / "latest-forward-observation.json").write_text(
        json.dumps(observation),
        encoding="utf-8",
    )
    prices = tmp_path / "data/research/critical_trading/yfinance"
    prices.mkdir(parents=True)
    close = np.linspace(80.0, 100.0, 40)
    pd.DataFrame(
        {
            "session_date": pd.date_range(
                "2026-01-01",
                periods=40,
                freq="B",
            ),
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
        }
    ).to_parquet(prices / "AAPL.parquet", index=False)

    signals = _phase11_14_signals(tmp_path)

    assert len(signals) == 1
    assert signals[0]["strategy_id"] == accepted
    assert signals[0]["_portfolio_eligible_source"] is True
    assert signals[0]["_deployment_eligible_source"] is False
    assert "PHASE11_14_NESTED_QUALIFIED" in signals[0]["reasons"]


def test_allocator_caps_heat_and_penalizes_correlation() -> None:
    matrix = pd.DataFrame(
        [[1.0, 0.95], [0.95, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    matrix.attrs["annualized_volatility"] = {
        "AAPL": 0.25,
        "MSFT": 0.25,
    }
    report = allocate_research_portfolio(
        [_ranked("AAPL", 0.9), _ranked("MSFT", 0.85)],
        policy=_policy(),
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        macro={"regime": {"market_regime": "RISK_ON"}},
        correlation=matrix,
        daily_target={},
    )

    assert report["portfolio_heat"] <= 0.04
    assert report["research_target_exposure"] <= 0.5
    second = next(
        row for row in report["allocations"] if row["ticker"] == "MSFT"
    )
    assert second["correlation_penalty"] > 0


def test_allocator_applies_dynamic_position_and_exposure_budget() -> None:
    matrix = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    matrix.attrs["annualized_volatility"] = {
        "AAPL": 0.20,
        "MSFT": 0.20,
    }
    report = allocate_research_portfolio(
        [_ranked("AAPL", 0.9), _ranked("MSFT", 0.85)],
        policy=_policy(),
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        macro={"regime": {"market_regime": "RISK_ON"}},
        correlation=matrix,
        daily_target={},
        dynamic_risk={
            "dynamic_research_maximum_positions": 1,
            "signal_scarcity_multiplier": 0.5,
            "multipliers": {"combined": 0.5},
        },
    )

    assert report["position_limit"] == 1
    assert len(report["allocations"]) == 1
    assert report["dynamic_risk_applied"] is True
    assert report["dynamic_risk_multiplier"] == 0.5
    assert report["signal_scarcity_multiplier"] == 0.5
    assert report["research_target_exposure"] <= 0.125


def test_micro_account_allocator_uses_meaningful_target_weight() -> None:
    matrix = pd.DataFrame([[1.0]], index=["ON"], columns=["ON"])
    matrix.attrs["annualized_volatility"] = {"ON": 0.25}
    report = allocate_research_portfolio(
        [_ranked("ON", 0.7)],
        policy=_policy(),
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        macro={"regime": {"market_regime": "RISK_ON"}},
        correlation=matrix,
        daily_target={},
        dynamic_risk={
            "dynamic_research_maximum_positions": 1,
            "signal_scarcity_multiplier": 0.6,
            "minimum_meaningful_target_weight": 0.10,
            "maximum_position_weight": 0.35,
            "maximum_portfolio_heat": 0.06,
            "multipliers": {"combined": 0.85},
        },
    )

    assert report["allocations"][0]["target_weight"] == 0.10
    assert report["maximum_portfolio_heat"] == 0.06
    assert report["portfolio_heat"] <= 0.06


def test_allocator_prefers_whole_share_feasible_candidate() -> None:
    matrix = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=["AAPL", "ON"],
        columns=["AAPL", "ON"],
    )
    matrix.attrs["annualized_volatility"] = {
        "AAPL": 0.20,
        "ON": 0.25,
    }
    report = allocate_research_portfolio(
        [_ranked("AAPL", 0.9), _ranked("ON", 0.7)],
        policy=_policy(),
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        macro={"regime": {"market_regime": "RISK_ON"}},
        correlation=matrix,
        daily_target={},
        dynamic_risk={
            "dynamic_research_maximum_positions": 1,
            "signal_scarcity_multiplier": 0.5,
            "multipliers": {"combined": 0.5},
        },
        whole_share_feasible_tickers={"ON"},
    )

    assert report["whole_share_preselection_applied"] is True
    assert [row["ticker"] for row in report["allocations"]] == ["ON"]


def test_position_action_is_advisory_and_never_executes() -> None:
    ranked = [
        {
            **_ranked("ON", 0.75),
            "preferred_entry": "80",
            "stop_loss": "70",
            "take_profit_1": "100",
        }
    ]
    actions = position_actions(
        [
            {
                "con_id": 1,
                "symbol": "ON",
                "quantity": 1.0,
                "average_cost": 90.0,
                "currency": "USD",
                "security_type": "STK",
            }
        ],
        ranked,
        {"allocations": []},
        policy=_policy(),
        dynamic_overrides={
            "ON": {
                "action": "AVOID",
                "risk_blockers": ["SHARIAH_ATTESTATION_REQUIRED"],
            }
        },
        daily_target={},
    )

    assert actions[0]["advisory_action"] == "EXIT"
    assert actions[0]["executable_action"] == "NO_ACTION"
    assert actions[0]["automatic_execution_allowed"] is False
    assert actions[0]["execution_authority"] == "NONE"


def test_position_lifecycle_reduction_precedes_score_based_add() -> None:
    candidate = {
        **_ranked("ON", 0.9),
        "preferred_entry": "100",
        "stop_loss": "95",
        "take_profit_1": "120",
    }
    actions = position_actions(
        [
            {
                "con_id": 1,
                "symbol": "ON",
                "quantity": 1.0,
                "average_cost": 90.0,
                "currency": "USD",
                "security_type": "STK",
            }
        ],
        [candidate],
        {"allocations": [candidate]},
        policy=_policy(),
        dynamic_overrides={},
        daily_target={},
        management_states={
            "ON": {
                "status": "GO",
                "action": "REDUCE_50",
                "reason_codes": ["PROFIT_GIVEBACK_40_PERCENT"],
            }
        },
    )

    assert actions[0]["advisory_action"] == "REDUCE"
    assert "PROFIT_GIVEBACK_40_PERCENT" in actions[0]["reason_codes"]
    assert actions[0]["executable_action"] == "NO_ACTION"


def test_position_management_store_is_append_only_and_public_audit_is_private(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    market = tmp_path / "data/research/critical_trading/yfinance"
    market.mkdir(parents=True)
    pd.DataFrame(
        {
            "session_date": pd.date_range(
                end=pd.Timestamp(now).normalize(), periods=20, freq="B"
            ),
            "open": np.linspace(100, 109, 20),
            "high": np.linspace(102, 111, 20),
            "low": np.linspace(98, 107, 20),
            "close": np.linspace(100, 110, 20),
        }
    ).to_parquet(market / "ON.parquet", index=False)
    reference = (
        tmp_path
        / "data/research/multitimeframe/private"
        / "provider=YFINANCE"
        / "symbol=ON"
        / "interval=1h"
        / "source_interval=1h"
    )
    reference.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": now,
                "fetched_at": now,
                "symbol": "ON",
                "provider": "YFINANCE",
                "close": 110.0,
            }
        ]
    ).to_parquet(reference / "bars.parquet", index=False)
    positions = [
        {
            "con_id": 123,
            "symbol": "ON",
            "quantity": 1.0,
            "average_cost": 100.0,
            "currency": "USD",
            "security_type": "STK",
        }
    ]
    first = _build_position_management(
        tmp_path,
        positions=positions,
        ranked=[{**_ranked("ON", 0.8), "stop_loss": "95"}],
        technical_regime={"status": "GO"},
    )
    second = _build_position_management(
        tmp_path,
        positions=positions,
        ranked=[{**_ranked("ON", 0.8), "stop_loss": "95"}],
        technical_regime={"status": "GO"},
    )
    third = _build_position_management(
        tmp_path,
        positions=positions,
        ranked=[{**_ranked("ON", 0.8), "stop_loss": "95"}],
        technical_regime={"status": "GO"},
    )

    database = tmp_path / "data/portfolio/private/position_management.sqlite3"
    with sqlite3.connect(database) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM position_events"
        ).fetchone()[0]
    public = first["public_audit"]
    serialized = json.dumps(public)
    assert count == 2
    assert second["public_audit"]["append_only"] is True
    assert third["public_audit"]["append_only"] is True
    assert '"quantity"' not in serialized
    assert '"entry_price"' not in serialized
    assert '"current_price"' not in serialized
    assert public["positions"][0]["market_data_status"] == "GO"
    assert public["positions"][0]["market_data_source"] == "YFINANCE"
    assert public["positions"][0]["atr_source"] == (
        "LEGACY_LOCAL_YFINANCE_DAILY_ATR"
    )
    assert public["execution_authority"] == "NONE"


def test_position_management_blocks_without_fresh_intraday_price(
    tmp_path: Path,
) -> None:
    market = tmp_path / "data/research/critical_trading/yfinance"
    market.mkdir(parents=True)
    pd.DataFrame(
        {
            "session_date": pd.date_range(
                "2026-07-01", periods=20, freq="B"
            ),
            "high": np.linspace(102, 111, 20),
            "low": np.linspace(98, 107, 20),
            "close": np.linspace(100, 110, 20),
        }
    ).to_parquet(market / "ON.parquet", index=False)

    report = _build_position_management(
        tmp_path,
        positions=[
            {
                "con_id": 123,
                "symbol": "ON",
                "quantity": 1.0,
                "average_cost": 100.0,
                "currency": "USD",
                "security_type": "STK",
            }
        ],
        ranked=[{**_ranked("ON", 0.8), "stop_loss": "95"}],
        technical_regime={"status": "GO"},
    )

    public = report["public_audit"]
    assert public["positions"][0]["status"] == "DATA_BLOCKED"
    assert public["positions"][0]["market_data_status"] == (
        "DATA_BLOCKED"
    )
    assert public["positions"][0]["market_data_reason"] == (
        "QUALIFIED_INTRADAY_REFERENCE_UNAVAILABLE"
    )
    assert public["broker_write_calls"] == 0
    assert public["execution_authority"] == "NONE"


def test_daily_target_blocks_risk_increase_but_not_exit() -> None:
    policy = _policy()
    ranked = [
        {
            **_ranked("AAPL", 0.9),
            "preferred_entry": "100",
            "stop_loss": "95",
            "take_profit_1": "120",
        }
    ]
    actions = position_actions(
        [
            {
                "con_id": 1,
                "symbol": "AAPL",
                "quantity": 1.0,
                "average_cost": 90.0,
                "currency": "USD",
                "security_type": "STK",
            }
        ],
        ranked,
        {"allocations": ranked},
        policy=policy,
        dynamic_overrides={},
        daily_target={
            "target_reached": True,
            "enforcement_active": True,
        },
    )
    assert actions[0]["advisory_action"] == "BLOCK_NEW_ENTRY"
    assert actions[0]["executable_action"] == "NO_ACTION"


def test_public_plan_masks_private_position_values_and_has_no_authority(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config/portfolio"
    config.mkdir(parents=True)
    source = (
        Path(__file__).parents[1]
        / "config/portfolio/active_manager_v1.json"
    )
    (config / "active_manager_v1.json").write_text(
        source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    signals = tmp_path / "output/signals"
    signals.mkdir(parents=True)
    signal = _signal("S1", "quality_momentum", "1d")
    signal["contract_identity"] = {"con_id": 1}
    (signals / "latest_signals.json").write_text(
        json.dumps({"signals": [signal]}),
        encoding="utf-8",
    )
    contracts = tmp_path / "output/ibkr/contracts"
    contracts.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "con_id": 1,
                "currency": "USD",
                "primary_exchange": "NASDAQ",
                "industry": "Technology",
            }
        ]
    ).to_parquet(contracts / "stocks.parquet", index=False)
    screener = tmp_path / "output/screener"
    screener.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "decision_time": "2026-07-29T23:59:00+00:00",
                "fundamental_score": 80.0,
                "liquidity_score": 100.0,
                "sector": "Technology",
                "shariah_status": "SHARIAH_ELIGIBLE_PIT",
            }
        ]
    ).to_parquet(screener / "candidate-history.parquet", index=False)
    price_root = (
        tmp_path / "data/research/critical_trading/yfinance"
    )
    price_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "session_date": pd.date_range(
                "2026-01-01", periods=130, freq="B"
            ),
            "close": np.linspace(80.0, 100.0, 130),
        }
    ).to_parquet(price_root / "AAPL.parquet", index=False)
    private = tmp_path / "data/broker/phase8/private"
    private.mkdir(parents=True)
    db = private / "broker_observation.sqlite3"
    snapshot = {
        "positions": {
            "status": "COMPLETE",
            "positions": [
                {
                    "con_id": 1,
                    "symbol": "AAPL",
                    "position_quantity": "3",
                    "average_cost": "92.50",
                    "currency": "USD",
                    "security_type": "STK",
                }
            ],
        }
    }
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE snapshots("
            "snapshot_id TEXT, snapshot_hash TEXT, payload_json TEXT, "
            "created_at TEXT)"
        )
        connection.execute(
            "INSERT INTO snapshots VALUES(?,?,?,?)",
            (
                "S1",
                "HASH",
                json.dumps(snapshot),
                "2026-07-30T00:00:00+00:00",
            ),
        )
    report = build_active_portfolio_report(tmp_path)
    public_text = (
        tmp_path / "output/portfolio/active_portfolio_plan.json"
    ).read_text(encoding="utf-8")
    private_text = (
        tmp_path / "data/portfolio/private/latest-action-plan.json"
    ).read_text(encoding="utf-8")

    assert report["status"] == "GO"
    assert '"current_quantity"' not in public_text
    assert '"average_cost"' not in public_text
    assert '"current_quantity"' in private_text
    assert report["authority"]["execution_authority"] == "NONE"
    assert report["authority"]["broker_write_calls"] == 0


def _private_market_inputs(root: Path) -> None:
    fx = root / "data/fx"
    fx.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "session_date": "2026-07-29",
                "base_currency": "USD",
                "quote_currency": "EUR",
                "rate": "0.9",
            }
        ]
    ).to_parquet(fx / "fx_daily.parquet", index=False)
    capital = root / "output/capital"
    capital.mkdir(parents=True)
    (capital / "capacity_report.json").write_text(
        json.dumps(
            {
                "instruments": [
                    {
                        "symbol": "AAPL",
                        "maximum_order_value_eur": 100000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_private_sizing_uses_equity_stop_cash_and_security_netting(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    policy = _policy()
    aapl = {
        **_ranked("AAPL", 0.8),
        "currency": "USD",
        "preferred_entry": "100",
        "stop_loss": "95",
        "target_weight": 0.1,
    }
    on = {
        **_ranked("ON", 0.5),
        "currency": "USD",
        "preferred_entry": "80",
        "stop_loss": "70",
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[
            {
                "symbol": "ON",
                "quantity": 1,
                "currency": "USD",
            }
        ],
        ranked=[aapl, on],
        allocation={"allocations": [aapl]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": "50000",
            "available_funds_eur": "10000",
            "total_cash_value_eur": "10000",
        },
        policy=policy,
    )
    rows = {row["ticker"]: row for row in report["positions"]}

    assert report["status"] == "GO"
    assert report["security_netting_status"] == "GO"
    assert report["whole_share_violation_count"] == 0
    assert report["negative_cash_violation_count"] == 0
    assert rows["AAPL"]["target_quantity"] == "29"
    assert rows["ON"]["target_quantity"] == "0"
    assert rows["ON"]["planned_quantity_delta"] == "-1"
    assert report["turnover_gate"] == "GO_WITH_RISK_REDUCING_EXIT"


def test_private_sizing_never_spends_more_than_available_cash(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    candidate = {
        **_ranked("AAPL", 1.0),
        "currency": "USD",
        "preferred_entry": "100",
        "stop_loss": "99",
        "target_weight": 0.5,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": "50000",
            "available_funds_eur": "100",
            "total_cash_value_eur": "100",
        },
        policy=_policy(),
    )

    assert report["positions"][0]["target_quantity"] == "1"
    assert report["negative_cash_violation_count"] == 0
    assert float(report["cash_remaining_eur"]) >= 0


def test_private_sizing_applies_dynamic_risk_multiplier(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    candidate = {
        **_ranked("AAPL", 0.8),
        "currency": "USD",
        "preferred_entry": "100",
        "stop_loss": "95",
        "target_weight": 0.5,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": "50000",
            "available_funds_eur": "10000",
            "total_cash_value_eur": "10000",
        },
        policy=_policy(),
        dynamic_risk={
            "base_risk_per_trade": 0.0035,
            "multipliers": {"combined": 0.5},
        },
    )

    assert report["status"] == "GO"
    assert report["dynamic_risk_applied"] is True
    assert report["dynamic_risk_multiplier"] == 0.5
    assert report["positions"][0]["target_quantity"] == "14"


def test_small_account_sizing_is_risk_first_not_target_weight_first(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    candidate = {
        **_ranked("AAPL", 0.8),
        "currency": "USD",
        "preferred_entry": "100",
        "stop_loss": "98",
        "target_weight": 0.04,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": "1870",
            "available_funds_eur": "1870",
            "total_cash_value_eur": "1870",
        },
        policy=_policy(),
        dynamic_risk={
            "base_risk_per_trade": 0.007,
            "maximum_position_weight": 0.30,
            "multipliers": {"combined": 0.85},
        },
    )

    row = report["positions"][0]
    assert report["small_account_whole_share_mode"] is True
    assert report["sizing_basis"] == "RISK_FIRST_WHOLE_SHARE_V2"
    assert row["research_target_weight"] == "0.04"
    assert row["desired_qty"] == "1"
    assert row["target_quantity"] == "1"
    assert row["normal_allowed_qty"] == "1"
    assert row["level1_canary_qty"] == "1"
    assert row["whole_share_feasibility_status"] == (
        "WHOLE_SHARE_FEASIBLE_RISK_FIRST"
    )
    assert float(row["minimum_feasible_weight"]) > 0.04


def test_micro_account_risk_profile_can_buy_two_meaningful_whole_shares(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    candidate = {
        **_ranked("AAPL", 0.6),
        "currency": "EUR",
        "preferred_entry": "70",
        "stop_loss": "65.50",
        "target_weight": 0.10,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": "1870",
            "available_funds_eur": "1870",
            "total_cash_value_eur": "1870",
        },
        policy=_policy(),
        dynamic_risk={
            "base_risk_per_trade": 0.01,
            "maximum_position_weight": 0.35,
            "maximum_portfolio_heat": 0.06,
            "multipliers": {"combined": 0.85},
        },
    )

    row = report["positions"][0]
    assert row["target_quantity"] == "2"
    assert row["actual_notional_eur"] == "140"
    assert row["desired_qty"] == "2"
    assert row["normal_allowed_qty"] == "2"
    assert row["level1_canary_qty"] == "1"
    assert row["canary_sizing_reason"] == (
        "CANARY_DOWNSCALED_TO_ONE_SHARE"
    )
    assert float(row["actual_portfolio_weight"]) > 0.07
    assert float(row["actual_risk_eur"]) <= float(row["risk_budget_eur"])
    assert report["maximum_portfolio_heat"] == 0.06
    assert report["negative_cash_violation_count"] == 0


def test_small_account_sizing_explains_infeasible_risk_unit(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    candidate = {
        **_ranked("AAPL", 0.8),
        "currency": "USD",
        "preferred_entry": "300",
        "stop_loss": "260",
        "target_weight": 0.04,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": "1870",
            "available_funds_eur": "1870",
            "total_cash_value_eur": "1870",
        },
        policy=_policy(),
        dynamic_risk={
            "base_risk_per_trade": 0.007,
            "maximum_position_weight": 0.30,
            "multipliers": {"combined": 0.85},
        },
    )

    row = report["positions"][0]
    assert row["target_quantity"] == "0"
    assert row["whole_share_feasibility_status"] == (
        "WHOLE_SHARE_RISK_BUDGET_INSUFFICIENT"
    )
    assert report["whole_share_infeasible_candidate_count"] == 1


def test_whole_share_sizing_exposes_economic_quantity_derivation(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    candidate = {
        **_ranked("AAPL", 1.0),
        "currency": "EUR",
        "preferred_entry": "80",
        "stop_loss": "76",
        "target_weight": 0.04,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": "1870",
            "available_funds_eur": "1870",
            "total_cash_value_eur": "1870",
            "research_sizing_capacity_eur": "1870",
        },
        policy=_policy(),
        dynamic_risk={
            "base_risk_per_trade": 0.007,
            "maximum_position_weight": 0.30,
            "multipliers": {"combined": 0.85},
        },
    )

    row = report["positions"][0]
    assert row["desired_notional_eur"] == "74.80"
    assert row["risk_based_quantity"] == "2"
    assert row["cash_based_quantity"] == "23"
    assert row["desired_qty"] == "1"
    assert row["whole_share_quantity"] == "1"
    assert row["actual_notional_eur"] == "80"
    assert Decimal(row["actual_risk_eur"]) > Decimal("4")
    assert row["remaining_cash_eur"] == "1790"
    assert row["execution_candidate_status"] == "EXECUTABLE_WHOLE_SHARE"


def test_allocation_below_one_share_is_explicitly_non_executable(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    candidate = {
        **_ranked("AAPL", 1.0),
        "currency": "EUR",
        "preferred_entry": "1000",
        "stop_loss": "900",
        "target_weight": 0.04,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": "1870",
            "available_funds_eur": "1870",
            "total_cash_value_eur": "1870",
        },
        policy=_policy(),
        dynamic_risk={
            "base_risk_per_trade": 0.007,
            "maximum_position_weight": 0.30,
            "multipliers": {"combined": 0.85},
        },
    )

    row = report["positions"][0]
    assert row["whole_share_quantity"] == "0"
    assert row["execution_candidate_status"] == "NON_EXECUTABLE_WHOLE_SHARE"
    assert row["whole_share_feasibility_status"] == (
        "WHOLE_SHARE_RISK_BUDGET_INSUFFICIENT"
    )


@pytest.mark.parametrize("equity", [1000, 1870, 2500, 5000, 10000])
@pytest.mark.parametrize("price", [5, 25, 100, 250, 500, 1000, 2000])
def test_small_account_price_capital_matrix_is_deterministic(
    tmp_path: Path,
    equity: int,
    price: int,
) -> None:
    _private_market_inputs(tmp_path)
    candidate = {
        **_ranked("AAPL", 1.0),
        "currency": "EUR",
        "preferred_entry": str(price),
        "stop_loss": str(Decimal(price) * Decimal("0.95")),
        "target_weight": 0.04,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "SNAPSHOT",
            "net_liquidation_eur": str(equity),
            "available_funds_eur": str(equity),
            "total_cash_value_eur": str(equity),
        },
        policy=_policy(),
        dynamic_risk={
            "base_risk_per_trade": 0.007,
            "maximum_position_weight": 0.30,
            "multipliers": {"combined": 0.85},
        },
    )
    row = report["positions"][0]
    whole_quantity = Decimal(row["whole_share_quantity"])
    calculated_limit = min(
        Decimal(row[key])
        for key in (
            "risk_quantity",
            "position_cap_quantity",
            "capacity_quantity",
            "cash_quantity",
        )
    )

    assert whole_quantity <= calculated_limit
    assert whole_quantity == whole_quantity.to_integral_value()
    assert Decimal(row["actual_notional_eur"]) <= Decimal(equity)
    assert Decimal(row["actual_risk_eur"]) <= Decimal(row["risk_budget_eur"])
    assert Decimal(row["remaining_cash_eur"]) >= 0
    assert row["execution_candidate_status"] == (
        "EXECUTABLE_WHOLE_SHARE"
        if whole_quantity >= 1
        else "NON_EXECUTABLE_WHOLE_SHARE"
    )


def test_risk_and_cash_constraints_are_not_interchangeable(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    risk_blocked = {
        **_ranked("AAPL", 1.0),
        "currency": "EUR",
        "preferred_entry": "1000",
        "stop_loss": "900",
        "target_weight": 0.5,
    }
    risk_report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[risk_blocked],
        allocation={"allocations": [risk_blocked]},
        account={
            "status": "GO",
            "snapshot_hash": "S1",
            "net_liquidation_eur": "1870",
            "available_funds_eur": "1870",
            "total_cash_value_eur": "1870",
        },
        policy=_policy(),
        dynamic_risk={
            "base_risk_per_trade": 0.007,
            "maximum_position_weight": 0.30,
            "multipliers": {"combined": 0.85},
        },
    )
    cash_blocked = {**risk_blocked, "preferred_entry": "100", "stop_loss": "99"}
    cash_report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[cash_blocked],
        allocation={"allocations": [cash_blocked]},
        account={
            "status": "GO",
            "snapshot_hash": "S2",
            "net_liquidation_eur": "10000",
            "available_funds_eur": "50",
            "total_cash_value_eur": "50",
        },
        policy=_policy(),
    )

    assert risk_report["positions"][0]["risk_based_quantity"] == "0"
    assert risk_report["positions"][0]["cash_based_quantity"] == "1"
    assert Decimal(
        cash_report["positions"][0]["risk_based_quantity"]
    ) >= 1
    assert cash_report["positions"][0]["cash_based_quantity"] == "0"
    assert cash_report["positions"][0]["whole_share_feasibility_status"] == (
        "WHOLE_SHARE_CASH_INSUFFICIENT"
    )


def test_transaction_cost_can_make_one_share_economically_incoherent(
    tmp_path: Path,
) -> None:
    _private_market_inputs(tmp_path)
    policy = _policy()
    policy["small_account_whole_share"]["transaction_cost_model"].update(
        estimated_entry_commission_eur=10,
        estimated_slippage_bps=100,
    )
    candidate = {
        **_ranked("AAPL", 1.0),
        "currency": "EUR",
        "preferred_entry": "5",
        "stop_loss": "4.90",
        "target_weight": 0.1,
    }
    report = build_private_sizing(
        tmp_path,
        current_positions=[],
        ranked=[candidate],
        allocation={"allocations": [candidate]},
        account={
            "status": "GO",
            "snapshot_hash": "S",
            "net_liquidation_eur": "1870",
            "available_funds_eur": "1870",
            "total_cash_value_eur": "1870",
        },
        policy=policy,
        dynamic_risk={
            "base_risk_per_trade": 0.007,
            "maximum_position_weight": 0.30,
            "multipliers": {"combined": 0.85},
        },
    )

    row = report["positions"][0]
    assert Decimal(row["risk_based_quantity"]) >= 1
    assert row["whole_share_quantity"] == "0"
    assert row["whole_share_feasibility_status"] == (
        "WHOLE_SHARE_ECONOMIC_COST_INCOHERENT"
    )


def test_opportunity_funnel_separates_watch_portfolio_and_execution(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "output/analysis"
    analysis.mkdir(parents=True)
    (analysis / "universe-coverage.json").write_text(
        json.dumps(
            {
                "universe_instrument_count": 18355,
                "analyzable_instrument_count": 280,
            }
        ),
        encoding="utf-8",
    )
    screener = tmp_path / "output/screener/2026-08-08"
    screener.mkdir(parents=True)
    (screener / "screening-results.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "symbol": "AAPL",
                        "asset_type": "STOCK",
                        "sector": "Technology",
                        "industry": "Consumer Electronics",
                        "market_cap": 5_000_000_000,
                        "fundamental_coverage": 0.9,
                        "classification": "WATCHLIST",
                        "total_score": 85,
                        "technical_score": 90,
                        "fundamental_score": 80,
                        "liquidity_score": 100,
                        "risk_score": 75,
                        "rejection_reasons": [],
                    },
                    {
                        "symbol": "CPER",
                        "asset_type": "ETF",
                        "sector": "Copper",
                        "industry": "Commodity Funds",
                        "classification": "REJECTED",
                        "total_score": 78,
                        "technical_score": 86,
                        "fundamental_score": 50,
                        "liquidity_score": 80,
                        "risk_score": 70,
                        "rejection_reasons": [
                            "SHARIAH_ATTESTATION_REQUIRED"
                        ],
                    },
                    {
                        "symbol": "MSFT",
                        "asset_type": "STOCK",
                        "sector": "Technology",
                        "industry": "Software - Infrastructure",
                        "market_cap": 2_000_000_000_000,
                        "fundamental_coverage": 1.0,
                        "classification": "HIGH_POTENTIAL",
                        "total_score": 88,
                        "technical_score": 92,
                        "fundamental_score": 84,
                        "liquidity_score": 100,
                        "risk_score": 77,
                        "rejection_reasons": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ranked = [
        {
            **_ranked("AAPL", 0.80),
            "strategy_families": ["quality_momentum"],
            "research_allocation_eligible": True,
            "research_allocation_blockers": [],
            "deployment_eligible": False,
            "deployment_blockers": ["EXECUTION_AUTHORITY_NONE"],
        },
        {
            **_ranked("MSFT", 0.82),
            "strategy_families": ["breakout"],
            "research_allocation_eligible": True,
            "research_allocation_blockers": [],
            "deployment_eligible": True,
            "deployment_blockers": [],
        },
    ]

    report = build_opportunity_funnel(
        tmp_path,
        signals=[{"ticker": "AAPL"}, {"ticker": "MSFT"}],
        ranked=ranked,
        policy=_policy(),
    )

    stages = {
        row["symbol"]: row["candidate_stage"]
        for row in report["watchlist_candidates"]
    }
    assert report["universe_instrument_count"] == 18355
    assert report["analyzable_instrument_count"] == 280
    assert stages["AAPL"] == "PORTFOLIO_CANDIDATE"
    assert stages["MSFT"] == "EXECUTION_CANDIDATE"
    assert "CPER" not in stages
    assert report["portfolio_candidate_count"] == 2
    assert report["execution_candidate_count"] == 1
    rejected = {
        row["symbol"]: row for row in report["top_screening_rejections"]
    }
    assert "SHARIAH_ATTESTATION_REQUIRED" in rejected["CPER"][
        "rejection_reasons"
    ]
    assert report["execution_authority"] == "NONE"
    assert report["broker_write_calls"] == 0
    assert report["qualified_screener_record_count"] == 2
    assert report["rejected_screener_record_count"] == 1
    assert report["screener_current_classification_required"] is True
    assert report["screened_family_counts"]["INDUSTRY_LEADERS"] == 2
    assert report["screened_family_counts"]["HIDDEN_GEMS"] == 1
    assert report["screening_family_statuses"]["industry_leaders"] == (
        "AVAILABLE"
    )
    assert report["screening_family_statuses"]["hidden_gems"] == "AVAILABLE"
    assert report["shariah_review_queue"][0]["symbol"] == "CPER"
    assert report["shariah_review_queue"][0]["review_status"] == (
        "READY_FOR_EXTERNAL_SHARIAH_REVIEW"
    )
    assert report["shariah_review_queue"][0][
        "automatic_approval_allowed"
    ] is False


def test_action_ledger_is_append_only_and_idempotent(
    tmp_path: Path,
) -> None:
    actions = [
        {
            "ticker": "ON",
            "advisory_action": "EXIT",
            "reason_codes": ["RISK_GATE"],
            "replacement_ticker": None,
        }
    ]
    sizing = {
        "account_snapshot_hash": "SNAPSHOT",
        "positions": [
            {
                "ticker": "ON",
                "current_quantity": "1",
                "target_quantity": "0",
                "planned_quantity_delta": "-1",
            }
        ],
    }
    first = _record_action_cycle(
        tmp_path,
        actions=actions,
        private_sizing=sizing,
        policy=_policy(),
    )
    second = _record_action_cycle(
        tmp_path,
        actions=actions,
        private_sizing=sizing,
        policy=_policy(),
    )

    assert first["decision_inserted"] is True
    assert first["action_events_inserted"] == 1
    assert second["decision_inserted"] is False
    assert second["action_events_inserted"] == 0
    assert second["decision_cycle_count"] == 1
    assert second["action_event_count"] == 1
    assert second["automatic_state_mutations"] == 0
    assert second["automatic_order_submissions"] == 0


def test_portfolio_heat_breach_blocks_add_and_reduces_hold() -> None:
    actions = [
        {
            "ticker": "AAPL",
            "advisory_action": "ADD",
            "reason_codes": [],
            "risk_increasing": True,
        },
        {
            "ticker": "MSFT",
            "advisory_action": "HOLD",
            "reason_codes": [],
            "risk_increasing": False,
        },
    ]
    report = _apply_portfolio_risk_overlay(
        actions,
        private_sizing={
            "status": "GO",
            "current_portfolio_heat": 0.05,
            "current_gross_exposure_pct": 0.4,
        },
        policy=_policy(),
    )

    assert report[0]["advisory_action"] == "BLOCK_NEW_ENTRY"
    assert report[1]["advisory_action"] == "REDUCE"
    assert all(
        "CURRENT_PORTFOLIO_HEAT_LIMIT_BREACHED"
        in row["reason_codes"]
        for row in report
    )
