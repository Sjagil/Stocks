from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from stocks.signals.service import (
    CANONICAL_SIGNAL_FAMILIES,
    _fair_signal_limit,
    _promote_consensus,
    _phase11_14_observer_signals,
    _phase11_14_observer_inventory,
    _publish_signal_outputs,
    _strategy_signal,
    promote_manual_signals,
    signal_mark_closed,
    signal_mark_executed,
    signal_list,
    signal_order_plan,
    signal_scan,
    signal_status,
)
from stocks.signals.storage import SignalStore
from stocks.signals.top5 import publish_top_signals


def _signal(
    ticker: str,
    strategy_id: str,
    confidence: float,
    *,
    action: str = "BUY",
) -> dict[str, object]:
    return {
        "signal_id": f"{ticker}-{strategy_id}",
        "ticker": ticker,
        "strategy_id": strategy_id,
        "confidence_score": confidence,
        "action": action,
        "reasons": [],
    }


def setup_signal_fixture(root: Path, monkeypatch: object) -> str:
    registry = root / "config/research_contracts/stocks_strategy_timeframe_registry_v1.json"
    registry.parent.mkdir(parents=True)
    registry.write_bytes(
        (
            Path(__file__).parents[1]
            / "config/research_contracts/stocks_strategy_timeframe_registry_v1.json"
        ).read_bytes()
    )
    candidate_id = "REC-FROZEN-1"
    output = root / "output" / "research"
    output.mkdir(parents=True)
    output.joinpath("recovered_survivors.json").write_text(
        json.dumps(
            {
                "survivors": [
                    {
                        "candidate_id": candidate_id,
                        "strategy_name": "MA crossover",
                        "family": "ma_crossover",
                        "timeframe": "1d",
                        "parameters": json.dumps({"fast": 20, "slow": 50}),
                        "classification": "FROZEN_SHADOW",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dates = pd.date_range(
        end=pd.Timestamp(datetime.now(UTC).date()),
        periods=300,
        freq="D",
    )
    close = pd.Series(range(100, 400), dtype=float)
    frame = pd.DataFrame(
        {
            "session_date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
        }
    )
    cache = root / "data" / "research" / "critical_trading" / "yfinance"
    cache.mkdir(parents=True)
    frame.to_parquet(cache / "TEST.parquet", index=False)
    intraday = (
        root
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=TEST/interval=1h/source_interval=1h"
    )
    intraday.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "symbol": "TEST",
                "provider": "YFINANCE",
                "close": float(close.iloc[-1]),
                "fetched_at": datetime.now(UTC).isoformat(),
                "quality_status": "VALIDATED_OHLC",
                "is_partial": True,
            }
        ]
    ).to_parquet(intraday / "bars.parquet", index=False)
    contracts = root / "output" / "ibkr" / "contracts"
    contracts.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "con_id": 123,
                "symbol": "TEST",
                "local_symbol": "TEST",
                "security_type": "STK",
                "currency": "EUR",
                "exchange": "SMART",
                "primary_exchange": "TESTEX",
                "trading_class": "TEST",
                "min_tick": 0.01,
                "resolved_at": datetime.now(UTC).isoformat(),
                "contract_hash": "A" * 64,
                "server_version": 225,
            }
        ]
    ).to_parquet(contracts / "stocks.parquet", index=False)
    monkeypatch.setenv("SIGNAL_MANUAL_APPROVAL_PHRASE", "EXACT TEST PHRASE")
    return candidate_id


def test_signal_authority_is_separate_from_execution(tmp_path: Path, monkeypatch) -> None:
    candidate = setup_signal_fixture(tmp_path, monkeypatch)
    promotion = promote_manual_signals(
        tmp_path, strategy_id=candidate, approval="EXACT TEST PHRASE"
    )
    assert promotion["status"] == "GO"
    assert promotion["signal_authority"] == "MANUAL_ACTIONABLE"
    assert promotion["execution_authority"] == "NONE"


def test_scan_generates_stop_targets_and_zero_orders(tmp_path: Path, monkeypatch) -> None:
    candidate = setup_signal_fixture(tmp_path, monkeypatch)
    promote_manual_signals(tmp_path, strategy_id=candidate, approval="EXACT TEST PHRASE")
    report = signal_scan(tmp_path, maximum_signals=10)
    assert report["status"] == "GO"
    assert report["signal_count"] == 1
    signal = report["signals"][0]
    assert Decimal(str(signal["stop_loss"])) < Decimal(str(signal["preferred_entry"]))
    assert Decimal(str(signal["take_profit_1"])) > Decimal(str(signal["preferred_entry"]))
    assert Decimal(str(signal["take_profit_2"])) > Decimal(str(signal["take_profit_1"]))
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0
    assert report["signal_authority"] == "MANUAL_ACTIONABLE"
    assert report["active_signal_asset_count"] == 1
    assert report["asset_without_active_signal_count"] == 0
    assert report["published_asset_count"] == 1
    assert report["published_active_asset_coverage_ratio"] == 1.0
    assert signal["contract_identity"]["cache_status"] == "FRESH"
    assert signal["contract_identity"]["contract_source"] == ("PHASE2_EXACT_STK_CACHE")
    assert signal["contract_identity"]["contract_hash"] == "A" * 64
    assert signal["strategy_family"] == "ma_crossover"
    timeframe_contract = signal["strategy_timeframe_contract"]
    assert timeframe_contract["entry_timeframe"] == "1d"
    assert timeframe_contract["setup_timeframe"] == "1d"
    assert timeframe_contract["context_timeframes"] == []
    assert timeframe_contract["declaration_status"] == "EXPLICIT_RESEARCH_ONLY"
    assert timeframe_contract["multi_timeframe_edge_claimed"] is False
    assert timeframe_contract["execution_authority"] == "NONE"


def test_stale_contract_cache_cannot_authorize_signal(tmp_path: Path, monkeypatch) -> None:
    candidate = setup_signal_fixture(tmp_path, monkeypatch)
    promote_manual_signals(tmp_path, strategy_id=candidate, approval="EXACT TEST PHRASE")
    path = tmp_path / "output/ibkr/contracts/stocks.parquet"
    contracts = pd.read_parquet(path)
    contracts["resolved_at"] = (datetime.now(UTC) - timedelta(days=8)).isoformat()
    contracts.to_parquet(path, index=False)

    report = signal_scan(tmp_path, maximum_signals=10)
    signal = report["signals"][0]

    assert signal["contract_identity"] == {}
    assert signal["action"] == "WATCHLIST"
    assert "CONTRACT_IDENTITY_UNAVAILABLE" in signal["risks"]
    assert signal["automatic_execution_allowed"] is False
    assert report["orders_generated"] == 0
    assert report["broker_calls"] == 0
    stored = pd.read_parquet(tmp_path / "output/signals/signal_history.parquet")
    assert stored.loc[0, "contract_identity"] == "{}"


def test_signal_parquet_normalizes_mixed_collection_types(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "signal_id": "SIG-A",
            "reasons": ["ONE", "TWO"],
            "contract_identity": {},
        },
        {
            "signal_id": "SIG-B",
            "reasons": ("THREE",),
            "contract_identity": {"con_id": 123},
        },
    ]

    _publish_signal_outputs(tmp_path, {"signals": rows}, rows)
    stored = pd.read_parquet(tmp_path / "output/signals/signal_history.parquet")

    assert stored["reasons"].tolist() == [
        '["ONE", "TWO"]',
        '["THREE"]',
    ]
    assert stored["contract_identity"].tolist() == [
        "{}",
        '{"con_id": 123}',
    ]


def test_manual_execution_and_close_lifecycle(tmp_path: Path, monkeypatch) -> None:
    candidate = setup_signal_fixture(tmp_path, monkeypatch)
    promote_manual_signals(tmp_path, strategy_id=candidate, approval="EXACT TEST PHRASE")
    report = signal_scan(tmp_path)
    signal_id = report["signals"][0]["signal_id"]
    executed = signal_mark_executed(
        tmp_path,
        signal_id=signal_id,
        quantity=Decimal("1"),
        fill_price=Decimal("399"),
    )
    closed = signal_mark_closed(
        tmp_path,
        signal_id=signal_id,
        quantity=Decimal("1"),
        fill_price=Decimal("410"),
        reason="manual test close",
    )
    assert executed["status"] == "GO"
    assert closed["status"] == "GO"
    assert closed["execution_authority"] == "NONE"


def test_manual_order_plan_is_whole_share_and_broker_free(tmp_path: Path, monkeypatch) -> None:
    candidate = setup_signal_fixture(tmp_path, monkeypatch)
    promote_manual_signals(tmp_path, strategy_id=candidate, approval="EXACT TEST PHRASE")
    scan = signal_scan(tmp_path)
    plan = signal_order_plan(
        tmp_path,
        signal_id=scan["signals"][0]["signal_id"],
        capital=Decimal("10000"),
        risk=Decimal("0.005"),
    )

    assert plan["status"] == "GO"
    assert Decimal(plan["quantity"]) % 1 == 0
    assert Decimal(plan["maximum_planned_loss_eur"]) <= Decimal("50")
    assert plan["automatic_submission"] is False
    assert plan["execution_authority"] == "NONE"
    assert plan["broker_calls"] == 0
    assert plan["orders_generated"] == 0


def test_signal_status_contract(tmp_path: Path) -> None:
    status = signal_status(tmp_path)
    assert status["SIGNALS_CAN_RUN_WITHOUT_BROKER"] is True
    assert status["SIGNALS_INCLUDE_STOP_LOSS"] is True
    assert status["SIGNALS_INCLUDE_TAKE_PROFIT"] is True
    assert status["SIGNAL_AUTHORITY_SEPARATE_FROM_EXECUTION"] is True
    assert status["unimplemented_frozen_candidate_families"] == []


def test_frozen_shadow_can_publish_watchlist_without_manual_authority(
    tmp_path: Path, monkeypatch
) -> None:
    setup_signal_fixture(tmp_path, monkeypatch)
    report = signal_scan(tmp_path)
    assert report["signal_authority"] == "SHADOW"
    assert report["signal_count"] == 1
    assert report["signals"][0]["action"] == "WATCHLIST"
    assert report["orders_generated"] == 0


def test_exploratory_one_hour_observer_is_watchlist_only_and_zero_quantity(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "config/research_contracts/stocks_strategy_timeframe_registry_v1.json"
    registry.parent.mkdir(parents=True)
    registry.write_bytes(
        (
            Path(__file__).parents[1]
            / "config/research_contracts/stocks_strategy_timeframe_registry_v1.json"
        ).read_bytes()
    )
    root = tmp_path / "output/research/phase11_14"
    root.mkdir(parents=True)
    now = datetime.now(UTC).replace(microsecond=0)
    root.joinpath("latest-forward-observation.json").write_text(
        json.dumps(
            {
                "schema": "phase11_14_forward_observation_v3",
                "status": "GO",
                "qualification_hash": "QUALIFICATION-HASH",
                "observations": [
                    {
                        "strategy_id": "P1114-ONE-HOUR",
                        "formula": "bollinger_breakout",
                        "timeframe": "1h",
                        "asset_class": "COMMODITY_PROXY",
                        "observer_tier": "EXPLORATORY_FORWARD_OBSERVER",
                        "portfolio_eligible": False,
                        "observation_status": "OBSERVATION_COMPLETE",
                        "closed_bar_timestamp": (now - timedelta(hours=1)).isoformat(),
                        "raw_active_signals": [
                            {
                                "symbol": "DBC",
                                "action": "BUY",
                                "confidence_score": 1.0,
                                "currently_attested": False,
                                "execution_envelope_status": "GO",
                                "entry_reference": 25.0,
                                "stop_loss": 23.0,
                                "take_profit_1": 29.0,
                                "take_profit_2": 31.0,
                                "stop_policy": "ATR_STRUCTURE",
                                "exit_policy": "TARGET_TRAIL",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    signals = _phase11_14_observer_signals(tmp_path, now=now)
    inventory = _phase11_14_observer_inventory(tmp_path)

    assert len(signals) == 1
    signal = signals[0]
    assert signal["timeframe"] == "1h"
    assert signal["action"] == "WATCHLIST"
    assert signal["observer_tier"] == "EXPLORATORY_FORWARD_OBSERVER"
    assert signal["portfolio_eligible"] is False
    assert signal["execution_eligible"] is False
    assert signal["automatic_execution_allowed"] is False
    assert signal["suggested_quantity"] == Decimal("0")
    assert signal["strategy_timeframe_contract"]["entry_timeframe"] == "1h"
    assert signal["strategy_timeframe_contract"]["research_only"] is True
    assert signal["strategy_timeframe_contract"]["execution_authority"] == "NONE"
    assert signal["broker_calls"] == 0
    assert signal["orders_generated"] == 0
    assert inventory == {
        "observer_strategy_count": 1,
        "exploratory_strategy_count": 1,
    }


def test_four_hour_signal_uses_closed_four_hour_bars_with_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    candidate_id = setup_signal_fixture(tmp_path, monkeypatch)
    survivor_path = tmp_path / "output" / "research" / "recovered_survivors.json"
    survivors = json.loads(survivor_path.read_text(encoding="utf-8"))
    survivors["survivors"][0]["timeframe"] = "4h"
    survivor_path.write_text(json.dumps(survivors), encoding="utf-8")

    daily_path = tmp_path / "data" / "research" / "critical_trading" / "yfinance" / "TEST.parquet"
    daily = pd.read_parquet(daily_path)
    daily["close"] = list(reversed(range(100, 400)))
    daily["open"] = daily["close"] + 0.5
    daily["high"] = daily["close"] + 1.0
    daily["low"] = daily["close"] - 1.0
    daily.to_parquet(daily_path, index=False)

    start = pd.Timestamp(datetime.now(UTC)).floor("h") - pd.Timedelta(hours=4 * 300)
    timestamps = pd.date_range(start=start, periods=301, freq="4h")
    close = pd.Series(range(100, 401), dtype=float)
    four_hour = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "session_date": timestamps.date.astype(str),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
            "provider": "YFINANCE",
            "source_interval": "1h",
            "bar_origin": "DERIVED",
            "quality_status": "VALIDATED_OHLC",
            "is_partial": False,
        }
    )
    four_hour.loc[four_hour.index[-1], "close"] = 1.0
    four_hour.loc[four_hour.index[-1], "is_partial"] = True
    cache = (
        tmp_path
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=TEST/interval=4h/source_interval=1h"
    )
    cache.mkdir(parents=True)
    four_hour.to_parquet(cache / "bars.parquet", index=False)

    report = signal_scan(tmp_path, maximum_signals=10)

    assert report["signal_count"] == 1
    signal = report["signals"][0]
    assert signal["strategy_id"] == candidate_id
    assert signal["timeframe"] == "4h"
    assert signal["source_provider"] == "YFINANCE"
    assert signal["source_interval"] == "1h"
    assert signal["bar_origin"] == "DERIVED"
    assert signal["bar_closed"] is True
    assert Decimal(str(signal["preferred_entry"])) == Decimal("399.0000")
    data_time = datetime.fromisoformat(str(signal["data_timestamp"]))
    expires = datetime.fromisoformat(str(signal["expiration_timestamp"]))
    assert expires - data_time == timedelta(days=3)
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0


def test_signal_status_does_not_count_expired_history_as_active(
    tmp_path: Path,
) -> None:
    with SignalStore(tmp_path) as store:
        store.append_signal(
            {
                "signal_id": "SIG-EXPIRED",
                "strategy_id": "STRATEGY-1",
                "ticker": "TEST",
                "action": "WATCHLIST",
                "lifecycle_status": "WATCHLIST",
                "expiration_timestamp": "2000-01-01T00:00:00+00:00",
            }
        )

    report = signal_status(tmp_path)

    assert report["active_signals"] == 0


def test_signal_status_counts_only_latest_version_per_strategy_asset_timeframe(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    with SignalStore(tmp_path) as store:
        for index in range(2):
            store.append_signal(
                {
                    "signal_id": f"SIG-VERSION-{index}",
                    "strategy_id": "STRATEGY-1",
                    "ticker": "TEST",
                    "timeframe": "4h",
                    "action": "WATCHLIST",
                    "lifecycle_status": "WATCHLIST",
                    "data_timestamp": (now - timedelta(hours=4 - index)).isoformat(),
                    "expiration_timestamp": (now + timedelta(days=1)).isoformat(),
                }
            )

    report = signal_status(tmp_path)

    assert report["active_signals"] == 1
    assert report["unexpired_open_history_record_count"] == 2
    assert report["superseded_unexpired_signal_count"] == 1
    assert report["active_signal_semantics"] == ("LATEST_UNEXPIRED_PER_TICKER_STRATEGY_TIMEFRAME")


def test_signal_store_refreshes_nonterminal_state_but_never_reopens_terminal(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    base = {
        "signal_id": "SIG-REFRESH",
        "strategy_id": "STRATEGY-1",
        "ticker": "TEST",
        "timeframe": "1d",
        "action": "WATCHLIST",
        "lifecycle_status": "WATCHLIST",
        "data_timestamp": now.isoformat(),
        "expiration_timestamp": (now + timedelta(days=1)).isoformat(),
    }
    with SignalStore(tmp_path) as store:
        assert store.append_signal(base) is True
        invalidated = {
            **base,
            "action": "AVOID",
            "lifecycle_status": "INVALIDATED",
            "price_validity_status": "CURRENT_PRICE_ABOVE_ENTRY_ZONE",
        }
        assert store.append_signal(invalidated) is True
        assert store.signal("SIG-REFRESH")["lifecycle_status"] == "INVALIDATED"
        store.connection.execute(
            "UPDATE signals SET lifecycle_status='CLOSED' WHERE signal_id=?",
            ("SIG-REFRESH",),
        )
        store.connection.commit()
        assert store.append_signal(base) is False
        assert store.signal("SIG-REFRESH")["lifecycle_status"] == "CLOSED"


def test_active_status_and_list_use_latest_scan_not_unexpired_history(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    old = {
        "signal_id": "SIG-OLD",
        "strategy_id": "STRATEGY-1",
        "ticker": "TEST",
        "timeframe": "1d",
        "action": "WATCHLIST",
        "lifecycle_status": "WATCHLIST",
        "data_timestamp": (now - timedelta(days=1)).isoformat(),
        "expiration_timestamp": (now + timedelta(days=5)).isoformat(),
    }
    with SignalStore(tmp_path) as store:
        store.append_signal(old)
    output = tmp_path / "output" / "signals"
    output.mkdir(parents=True)
    output.joinpath("latest_signals.json").write_text(
        json.dumps(
            {
                "signals": [
                    {
                        **old,
                        "signal_id": "SIG-CURRENT-INVALID",
                        "action": "AVOID",
                        "lifecycle_status": "INVALIDATED",
                        "data_timestamp": now.isoformat(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    status = signal_status(tmp_path)
    active = signal_list(tmp_path, "active")
    watchlist = signal_list(tmp_path, "watchlist")

    assert status["active_signals"] == 0
    assert status["active_signal_source"] == "LATEST_SCAN_ARTIFACT"
    assert active["count"] == 0
    assert watchlist["count"] == 0
    assert active["current_state_source"] == "LATEST_SCAN_ARTIFACT"


def test_fair_limit_covers_assets_before_repeating_tickers() -> None:
    plans = [
        _signal("AAPL", "S1", 0.9),
        _signal("AAPL", "S2", 0.9),
        _signal("MSFT", "S1", 0.8),
        _signal("NVDA", "S1", 0.7),
    ]

    selected = _fair_signal_limit(plans, 3)

    assert {row["ticker"] for row in selected} == {"AAPL", "MSFT", "NVDA"}


def test_fair_limit_round_robins_remaining_strategy_capacity() -> None:
    plans = [
        _signal("AAPL", "S1", 0.9),
        _signal("MSFT", "S1", 0.8),
        _signal("AAPL", "S2", 0.7),
        _signal("MSFT", "S3", 0.6),
        _signal("AAPL", "S1", 0.5),
    ]

    selected = _fair_signal_limit(plans, 4)

    assert len(selected) == 4
    assert {row["strategy_id"] for row in selected} == {"S1", "S2", "S3"}


def test_consensus_is_calculated_before_fair_limit() -> None:
    plans = [
        _signal("NVDA", "S1", 0.9),
        _signal("NVDA", "S2", 0.8),
        _signal("AAPL", "S1", 0.7),
    ]

    promoted = _promote_consensus(plans)
    selected = _fair_signal_limit(promoted, 2)
    nvda = next(row for row in selected if row["ticker"] == "NVDA")

    assert nvda["action"] == "STRONG_BUY"
    assert "MULTI_STRATEGY_CONFIRMATION" in nvda["reasons"]


def test_canonical_signal_registry_covers_broad_frozen_families() -> None:
    expected = {
        "adx_trend",
        "atr_breakout",
        "bollinger_breakout",
        "breakout_consensus",
        "channel_consensus",
        "ema_pullback",
        "keltner_breakout",
        "macd_trend",
        "momentum_consensus",
        "pullback_consensus",
        "range_expansion_breakout",
        "risk_adjusted_momentum",
        "roc_trend",
        "rsi2_adx_pullback",
        "stochastic_trend_pullback",
        "trend_consensus",
        "triple_ma_trend",
        "volatility_contraction_breakout",
        "volume_breakout",
    }
    assert expected <= CANONICAL_SIGNAL_FAMILIES


def test_broad_family_signals_use_closed_ohlcv_without_future_rows() -> None:
    length = 320
    close = pd.Series([100.0 + index * 0.25 for index in range(length)])
    close.iloc[-1] += 8.0
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.25
    volume = pd.Series([1_000_000.0] * length)
    volume.iloc[-1] = 2_500_000.0
    parameters = {
        "adx_min": 18,
        "atr_mult": 1.5,
        "channel": 20,
        "fast": 20,
        "roc_threshold": 0.0,
        "rsi14_low": 42,
        "rsi_low": 15,
        "sigma": 2.0,
        "slow": 52,
        "volatility_ratio": 0.75,
        "volume_mult": 1.25,
    }
    families = CANONICAL_SIGNAL_FAMILIES - {
        "commodity_etf_trend",
        "etf_rotation",
        "quality_trend_consensus",
        "time_series_momentum",
    }
    for family in families:
        action, confidence, reasons = _strategy_signal(
            family,
            close,
            high,
            low,
            parameters,
            open_=open_,
            volume=volume,
        )
        assert action in {"BUY", "NO_SIGNAL"}
        assert 0.0 <= confidence <= 0.9
        assert "FAMILY_SIGNAL_NOT_IMPLEMENTED" not in reasons

    quality = _strategy_signal(
        "quality_trend_consensus",
        close,
        high,
        low,
        parameters,
        open_=open_,
        volume=volume,
    )
    assert quality == (
        "NO_SIGNAL",
        0.0,
        ["PIT_FUNDAMENTAL_QUALITY_REQUIRED"],
    )


def _top5_fixture(root: Path) -> None:
    portfolio = root / "output" / "portfolio"
    signals = root / "output" / "signals"
    dynamic = root / "output" / "dynamic"
    portfolio.mkdir(parents=True)
    signals.mkdir(parents=True)
    dynamic.mkdir(parents=True)
    opportunities = []
    signal_rows = []
    for index, (ticker, sector, region, eligible) in enumerate(
        [
            ("AAA", "Technology", "US", True),
            ("BBB", "Technology", "US", False),
            ("CCC", "Healthcare", "EU", False),
            ("DDD", "Industrials", "JP", False),
            ("EEE", "Financials", "US", False),
            ("FFF", "Energy", "GLOBAL", False),
        ]
    ):
        opportunities.append(
            {
                "ticker": ticker,
                "asset_type": "STOCK",
                "currency": "USD",
                "sector": sector,
                "region": region,
                "sleeve": (
                    "stock" if index < 2 else "etf_core" if index < 4 else "commodity_security"
                ),
                "opportunity_score": 0.90 - index * 0.03,
                "components": {
                    "liquidity": 1.0,
                    "signal_quality": 0.8,
                },
                "contract_resolved": True,
                "research_allocation_eligible": eligible,
                "research_allocation_blockers": (
                    [] if eligible else ["SHARIAH_ATTESTATION_REQUIRED"]
                ),
                "execution_blockers": [],
                "deployment_blockers": ["EXECUTION_AUTHORITY_NONE"],
                "deployment_eligible": False,
                "entry_method": "LIMIT",
                "exit_policy": "ATR_TRAIL",
                "expected_holding_period": "5-20 sessions",
                "strategy_families": [f"family_{index}"],
                "strategy_ids": [f"S{index}"],
                "timeframes": ["4h", "1d"],
                "evidence_tiers": ["FROZEN_SHADOW"],
                "shariah_status": (
                    "SHARIAH_ELIGIBLE_PIT" if eligible else "SHARIAH_DATA_UNAVAILABLE"
                ),
            }
        )
        signal_rows.append(
            {
                "signal_id": f"SIG-{ticker}",
                "ticker": ticker,
                "strategy_id": f"S{index}",
                "exchange": "SMART",
                "signal_timestamp": "2026-01-02T00:00:00+00:00",
                "data_timestamp": "2026-01-01T00:00:00+00:00",
                "data_freshness": "FRESH",
                "current_market_price": "100",
                "preferred_entry": "100",
                "entry_zone_low": "99",
                "entry_zone_high": "101",
                "invalidation_level": "95",
                "stop_loss": "95",
                "take_profit_1": "108",
                "take_profit_2": "112",
                "reward_risk_1": "1.6",
                "reward_risk_2": "2.4",
                "confidence_score": "0.8",
                "suggested_quantity": "1",
                "maximum_order_value_eur": "100",
                "expiration_timestamp": "2099-01-10T00:00:00+00:00",
                "reasons": ["CLOSED_BAR_SIGNAL"],
                "risks": [],
                "automatic_execution_allowed": False,
            }
        )
    (portfolio / "opportunity_ranking.json").write_text(
        json.dumps({"opportunities": opportunities}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "ticker": row["ticker"],
                **{
                    other["ticker"]: (1.0 if other["ticker"] == row["ticker"] else 0.2)
                    for other in opportunities
                },
            }
            for row in opportunities
        ]
    ).to_parquet(portfolio / "correlation_matrix.parquet", index=False)
    (signals / "latest_signals.json").write_text(
        json.dumps({"signals": signal_rows}),
        encoding="utf-8",
    )
    (dynamic / "status.json").write_text(
        json.dumps({"current_regime": "BULL_TREND_LOW_VOL"}),
        encoding="utf-8",
    )


def test_top5_publishes_raw_diversified_and_separate_eligibility(
    tmp_path: Path,
) -> None:
    _top5_fixture(tmp_path)

    report = publish_top_signals(tmp_path, mode="diversified", limit=5)

    assert report["status"] == "GO"
    assert report["signal_count"] == 5
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0
    first = report["signals"][0]
    assert first["manual_signal_eligible"] is True
    assert first["automated_execution_eligible"] is False
    assert first["eligibility_status"] == "MANUAL_ACTIONABLE"
    assert (tmp_path / "output" / "signals" / "latest_raw_top_5.json").exists()
    assert (tmp_path / "output" / "signals" / "latest_diversified_top_5.json").exists()
    assert (tmp_path / "output" / "signals" / "latest_manual_order_plans.json").exists()


def test_top5_never_fills_with_weak_opportunities(tmp_path: Path) -> None:
    _top5_fixture(tmp_path)
    path = tmp_path / "output" / "portfolio" / "opportunity_ranking.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["opportunities"][2]["opportunity_score"] = 0.40
    payload["opportunities"][3]["opportunity_score"] = 0.30
    payload["opportunities"][4]["opportunity_score"] = 0.20
    payload["opportunities"][5]["opportunity_score"] = 0.10
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = publish_top_signals(tmp_path, mode="raw", limit=5)

    assert report["signal_count"] == 2
    assert report["selection_policy"]["weak_signals_added_to_fill_limit"] is False


def test_top5_history_suppresses_unchanged_publication(tmp_path: Path) -> None:
    _top5_fixture(tmp_path)

    publish_top_signals(tmp_path, mode="diversified", limit=5)
    publish_top_signals(tmp_path, mode="diversified", limit=5)

    history = (
        (tmp_path / "output" / "signals" / "signal_history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(history) == 1


def test_top5_rotation_marks_persistent_without_changing_semantic_hash(
    tmp_path: Path,
) -> None:
    _top5_fixture(tmp_path)

    first = publish_top_signals(tmp_path, mode="diversified", limit=5)
    second = publish_top_signals(tmp_path, mode="diversified", limit=5)

    assert first["rotation_summary"]["new_count"] == 5
    assert second["rotation_summary"]["persistent_count"] == 5
    assert second["rotation_summary"]["material_change_count"] == 0
    assert first["content_hash"] == second["content_hash"]
    assert all(row["appearance_status"] == "PERSISTENT" for row in second["signals"])


def test_top5_excludes_expired_signal_before_ranking(tmp_path: Path) -> None:
    _top5_fixture(tmp_path)
    path = tmp_path / "output" / "signals" / "latest_signals.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["signals"][0]["expiration_timestamp"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = publish_top_signals(tmp_path, mode="raw", limit=5)

    assert report["expired_or_invalid_signal_count"] == 1
    assert "AAA" not in {row["symbol"] for row in report["signals"]}


def test_top5_clamps_declared_expiry_to_one_hour_freshness_contract(
    tmp_path: Path,
) -> None:
    _top5_fixture(tmp_path)
    path = tmp_path / "output" / "signals" / "latest_signals.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["signals"][0].update(
        {
            "timeframe": "1h",
            "data_timestamp": (datetime.now(UTC) - timedelta(hours=13)).isoformat(),
            "expiration_timestamp": (datetime.now(UTC) + timedelta(days=365)).isoformat(),
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = publish_top_signals(tmp_path, mode="raw", limit=5)

    assert report["expired_or_invalid_signal_count"] == 1
    assert "AAA" not in {row["symbol"] for row in report["signals"]}


def test_top5_separates_signal_timeframe_from_confirmation_timeframes(
    tmp_path: Path,
) -> None:
    _top5_fixture(tmp_path)
    path = tmp_path / "output" / "signals" / "latest_signals.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    payload["signals"][0].update(
        {
            "timeframe": "4h",
            "data_timestamp": now.isoformat(),
            "expiration_timestamp": (now + timedelta(days=30)).isoformat(),
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = publish_top_signals(tmp_path, mode="raw", limit=5)
    first = next(row for row in report["signals"] if row["symbol"] == "AAA")

    assert first["timeframe"] == "4h"
    assert first["signal_timeframe"] == "4h"
    assert first["confirmation_timeframes"] == ["4h", "1d"]
    assert datetime.fromisoformat(first["valid_until"]) == now + timedelta(days=3)


def test_top5_public_artifacts_contain_no_broker_identity_or_order_calls(
    tmp_path: Path,
) -> None:
    _top5_fixture(tmp_path)
    publish_top_signals(tmp_path, mode="diversified", limit=5)

    payload = (tmp_path / "output" / "signals" / "latest_top_5_publication.json").read_text(
        encoding="utf-8"
    )

    assert "account_id" not in payload.lower()
    assert "broker_order_id" not in payload.lower()
    assert '"execution_authority": "NONE"' in payload
    assert '"orders_generated": 0' in payload


def test_top5_specialized_collections_remain_read_only(
    tmp_path: Path,
) -> None:
    _top5_fixture(tmp_path)
    opportunity_path = tmp_path / "output" / "portfolio" / "opportunity_ranking.json"
    opportunities = json.loads(opportunity_path.read_text(encoding="utf-8"))
    opportunities["opportunities"][2]["asset_type"] = "ETF"
    opportunities["opportunities"][3]["asset_type"] = "ETF"
    opportunities["opportunities"][4]["asset_type"] = "COMMODITY_ETF"
    opportunity_path.write_text(json.dumps(opportunities), encoding="utf-8")
    universe_path = tmp_path / "config" / "universes"
    universe_path.mkdir(parents=True)
    (universe_path / "broad_multi_asset_v1.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "group": "commodity",
                        "asset_type": "COMMODITY_ETF",
                        "sleeve": "commodity_security",
                        "region": "GLOBAL",
                        "instruments": {"EEE": "GOLD"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    stocks = publish_top_signals(tmp_path, mode="stocks", limit=5)
    etfs = publish_top_signals(tmp_path, mode="etfs", limit=5)
    commodities = publish_top_signals(tmp_path, mode="commodities", limit=5)
    automatic = publish_top_signals(tmp_path, mode="auto-eligible", limit=5)

    assert {row["symbol"] for row in stocks["signals"]} == {
        "AAA",
        "BBB",
        "FFF",
    }
    assert {row["symbol"] for row in etfs["signals"]} == {"CCC", "DDD"}
    assert [row["symbol"] for row in commodities["signals"]] == [
        "EEE",
        "FFF",
    ]
    assert automatic["signals"] == []
    for report in (stocks, etfs, commodities, automatic):
        assert report["execution_authority"] == "NONE"
        assert report["broker_calls"] == 0
        assert report["orders_generated"] == 0
