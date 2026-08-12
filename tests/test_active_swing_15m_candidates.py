from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from stocks.context.entry_observer import _hierarchy_status
from stocks.signals.active_swing import (
    active_swing_context_gaps,
    generate_active_swing_candidates,
)
from stocks.signals.freshness import evaluate_signal_freshness
from stocks.signals.service import active_swing_scan
from stocks.signals.storage import SignalStore


def _config(root: Path) -> None:
    path = root / "config/active_swing/candidate_hypotheses_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "active_swing_15m_candidate_hypotheses_v1",
                "status": "ACTIVE_RESEARCH_SHADOW_ONLY",
                "version": "test-v1",
                "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
                "entry_timeframe": "15m",
                "setup_timeframe": "1h",
                "required_timeframes": ["15m", "1h", "4h"],
                "optional_timeframes": ["1d"],
                "session": "RTH",
                "maximum_trigger_age_bars": 16,
                "maximum_trigger_age_minutes": 240,
                "near_setup_atr_fraction": 0.25,
                "maximum_near_setup_age_bars": 64,
                "higher_timeframe_policy": ("CAUSAL_CONTEXT_NOT_MANDATORY_CONSENSUS"),
                "negative_sampling_policy": "CANDIDATE_CONDITIONED_ONLY",
                "hypotheses": [
                    {
                        "strategy_id": "ACTIVE-SWING-15M-BREAKOUT-V1",
                        "family": "volatility_contraction_breakout",
                        "trigger": "CLOSE_BREAKS_PRIOR_20_BAR_HIGH",
                        "channel_bars": 20,
                        "atr_bars": 14,
                        "stop_atr": 2.0,
                        "target_1_r": 1.5,
                        "target_2_r": 2.5,
                    }
                ],
                "strategy_authority": "NONE",
                "execution_authority": "NONE",
                "automatic_execution_allowed": False,
            }
        ),
        encoding="utf-8",
    )


def _bars(
    timestamps: pd.DatetimeIndex,
    closes: list[float],
    *,
    interval: str,
    source_interval: str,
    ingested_at: datetime,
) -> pd.DataFrame:
    values = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "session_date": timestamps.strftime("%Y-%m-%d"),
            "open": values - 0.2,
            "high": values + 0.5,
            "low": values - 0.5,
            "close": values,
            "volume": 1_000_000.0,
            "provider": "YFINANCE",
            "interval": interval,
            "source_interval": source_interval,
            "bar_origin": "NATIVE" if interval == source_interval else "DERIVED",
            "quality_status": "VALIDATED_OHLC",
            "is_partial": False,
            "partial_bucket": False,
            "session": "RTH",
            "exchange_timezone": "America/New_York",
            "ingested_at": ingested_at.isoformat(),
        }
    )


def _write_frame(
    root: Path,
    frame: pd.DataFrame,
    *,
    timeframe: str,
    source_interval: str,
) -> None:
    path = (
        root
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=TEST"
        / f"interval={timeframe}"
        / f"source_interval={source_interval}"
    )
    path.mkdir(parents=True)
    frame.to_parquet(path / "bars.parquet", index=False)


def _candidate_fixture(root: Path) -> datetime:
    _config(root)
    trigger_start = pd.Timestamp("2026-08-11T19:45:00Z")
    observed = datetime(2026, 8, 11, 20, 1, tzinfo=UTC)
    fifteen = pd.date_range(end=trigger_start, periods=120, freq="15min")
    closes_15m = [100.0] * 119 + [103.0]
    _write_frame(
        root,
        _bars(
            fifteen,
            closes_15m,
            interval="15m",
            source_interval="15m",
            ingested_at=datetime(2026, 8, 11, 20, 0, 30, tzinfo=UTC),
        ),
        timeframe="15m",
        source_interval="15m",
    )
    hourly = pd.date_range(end="2026-08-11T18:30:00Z", periods=80, freq="1h")
    _write_frame(
        root,
        _bars(
            hourly,
            [80.0 + index * 0.25 for index in range(80)],
            interval="1h",
            source_interval="1h",
            ingested_at=datetime(2026, 8, 11, 19, 31, tzinfo=UTC),
        ),
        timeframe="1h",
        source_interval="1h",
    )
    four_hour = pd.date_range(end="2026-08-11T13:30:00Z", periods=80, freq="4h")
    _write_frame(
        root,
        _bars(
            four_hour,
            [120.0 - index * 0.25 for index in range(80)],
            interval="4h",
            source_interval="1h",
            ingested_at=datetime(2026, 8, 11, 17, 31, tzinfo=UTC),
        ),
        timeframe="4h",
        source_interval="1h",
    )
    return observed


def _near_setup_fixture(root: Path) -> datetime:
    observed = _candidate_fixture(root)
    path = (
        root
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=TEST/interval=15m/source_interval=15m/bars.parquet"
    )
    frame = pd.read_parquet(path)
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [
        100.2,
        100.9,
        99.9,
        100.4,
    ]
    frame.to_parquet(path, index=False)
    return observed


def test_native_15m_candidate_has_stable_setup_and_causal_context(
    tmp_path: Path,
) -> None:
    observed = _candidate_fixture(tmp_path)

    first = generate_active_swing_candidates(tmp_path, ["TEST"], observed_at=observed)
    second = generate_active_swing_candidates(
        tmp_path,
        ["TEST"],
        observed_at=observed + timedelta(seconds=30),
    )

    assert first["status"] == "GO"
    assert first["native_15m_ready_symbol_count"] == 1
    assert first["candidate_count"] == 1
    assert first["declared_strategy_architecture_count"] == 1
    declaration = first["declared_strategy_architectures"][0]
    assert declaration["strategy_id"] == "ACTIVE-SWING-15M-BREAKOUT-V1"
    assert declaration["strategy_timeframe_contract"]["entry_timeframe"] == "15m"
    assert declaration["strategy_timeframe_contract"]["setup_timeframe"] == "1h"
    assert declaration["strategy_timeframe_contract"]["context_timeframes"] == [
        "1h",
        "4h",
        "1d",
    ]
    signal = first["signals"][0]
    assert signal["setup_id"] == second["signals"][0]["setup_id"]
    assert signal["candidate_unit"] == "ONE_NATURAL_STRATEGY_SETUP"
    assert signal["source_interval"] == "15m"
    assert signal["bar_origin"] == "NATIVE"
    assert signal["timeframe"] == "15m"
    assert signal["strict_expiration"] is True
    assert signal["higher_timeframe_consensus_required"] is False
    assert signal["timeframe_evidence"]["1h"]["trend_state"] == "SUPPORTIVE"
    assert signal["timeframe_evidence"]["4h"]["trend_state"] == "ADVERSE"
    assert signal["strategy_authority"] == "NONE"
    assert signal["execution_authority"] == "NONE"
    assert signal["automatic_orders"] == 0
    hierarchy = _hierarchy_status(signal, support={})
    assert hierarchy["status"] == "EXPLICIT_CAUSAL_TIMEFRAME_CONTRACT_READY"
    assert hierarchy["ready"] is True
    assert hierarchy["all_timeframes_need_not_agree"] is True
    assert hierarchy["fifteen_minute_strategy_can_create_candidate"] is True
    assert hierarchy["fifteen_minute_execution_can_create_trade"] is False


def test_near_setup_persists_then_promotes_only_on_exact_next_bar_trigger(
    tmp_path: Path,
) -> None:
    observed = _near_setup_fixture(tmp_path)

    first = generate_active_swing_candidates(tmp_path, ["TEST"], observed_at=observed)
    second = generate_active_swing_candidates(
        tmp_path, ["TEST"], observed_at=observed + timedelta(seconds=30)
    )

    assert first["candidate_count"] == 0
    assert first["near_setup_count"] == 1
    near = first["near_setups"][0]
    assert near["setup_id"] == second["near_setups"][0]["setup_id"]
    assert second["near_setups"][0]["observation_count"] == 2
    assert near["lifecycle_state"] == "NEAR_SETUP"
    assert near["candidate_unit"] == "PRE_TRIGGER_OBSERVATION_NOT_A_CANDIDATE"
    assert near["does_not_create_candidate"] is True
    assert near["does_not_change_strategy_thresholds"] is True
    assert near["execution_authority"] == "NONE"
    assert near["automatic_orders"] == 0

    path = (
        tmp_path
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=TEST/interval=15m/source_interval=15m/bars.parquet"
    )
    frame = pd.read_parquet(path)
    trigger = _bars(
        pd.DatetimeIndex([pd.Timestamp("2026-08-11T20:00:00Z")]),
        [101.6],
        interval="15m",
        source_interval="15m",
        ingested_at=datetime(2026, 8, 11, 20, 15, 30, tzinfo=UTC),
    )
    pd.concat([frame, trigger], ignore_index=True).to_parquet(path, index=False)

    promoted = generate_active_swing_candidates(
        tmp_path,
        ["TEST"],
        observed_at=datetime(2026, 8, 11, 20, 16, tzinfo=UTC),
    )

    assert promoted["candidate_count"] == 1
    assert promoted["near_setup_count"] == 0
    assert promoted["near_setup_promoted_count"] == 1
    candidate = promoted["signals"][0]
    assert candidate["setup_id"] == near["setup_id"]
    assert candidate["promoted_from_near_setup"] is True
    assert candidate["trigger_event_timestamp"] == "2026-08-11T20:15:00+00:00"
    assert candidate["execution_authority"] == "NONE"
    state = json.loads(
        (tmp_path / "data/research/active_swing/private/near-setups-state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["near_setup_count"] == 0
    assert state["promoted_in_scan_count"] == 1
    assert state["order_calls"] == 0


def test_invalid_near_setup_expires_without_becoming_candidate(
    tmp_path: Path,
) -> None:
    observed = _near_setup_fixture(tmp_path)
    first = generate_active_swing_candidates(tmp_path, ["TEST"], observed_at=observed)
    assert first["near_setup_count"] == 1

    path = (
        tmp_path
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=TEST/interval=15m/source_interval=15m/bars.parquet"
    )
    frame = pd.read_parquet(path)
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [
        98.2,
        98.5,
        97.5,
        98.0,
    ]
    frame.to_parquet(path, index=False)

    expired = generate_active_swing_candidates(
        tmp_path, ["TEST"], observed_at=observed + timedelta(seconds=30)
    )

    assert expired["candidate_count"] == 0
    assert expired["near_setup_count"] == 0
    assert expired["near_setup_expired_count"] == 1
    assert expired["near_setup_promoted_count"] == 0


def test_stale_session_data_expires_near_setup_fail_closed(
    tmp_path: Path,
) -> None:
    observed = _near_setup_fixture(tmp_path)
    first = generate_active_swing_candidates(tmp_path, ["TEST"], observed_at=observed)
    assert first["near_setup_count"] == 1

    stale = generate_active_swing_candidates(
        tmp_path,
        ["TEST"],
        observed_at=datetime(2026, 8, 12, 15, 0, tzinfo=UTC),
    )

    assert stale["status"] == "DATA_BLOCKED"
    assert stale["candidate_count"] == 0
    assert stale["near_setup_count"] == 0
    assert stale["near_setup_expired_count"] == 1
    symbol = stale["symbol_status"][0]
    assert symbol["required_timeframes_ready"] is False
    assert symbol["timeframe_status"]["15m"] == "STALE_BAR_BLOCKED"
    assert symbol["timeframe_freshness"]["15m"]["status"] == ("STALE_BAR_BLOCKED")


def test_context_gap_audit_targets_only_missing_or_stale_1h_4h(
    tmp_path: Path,
) -> None:
    observed = _near_setup_fixture(tmp_path)

    ready = active_swing_context_gaps(tmp_path, ["TEST"], observed_at=observed)
    missing = active_swing_context_gaps(tmp_path, ["MISSING"], observed_at=observed)

    assert ready["status"] == "GO"
    assert ready["gap_symbols"] == []
    assert missing["status"] == "CONTEXT_REFRESH_REQUIRED"
    assert missing["gap_symbols"] == ["MISSING"]
    assert missing["gaps"][0]["timeframe_status"] == {
        "1h": "MISSING_CANONICAL_PARTITION",
        "4h": "MISSING_CANONICAL_PARTITION",
    }
    assert missing["refresh_scope"] == "GAPS_ONLY"
    assert missing["execution_authority"] == "NONE"
    assert missing["broker_calls"] == 0


def test_dedicated_fast_scan_preserves_canonical_money_signals(
    tmp_path: Path,
) -> None:
    observed = _candidate_fixture(tmp_path)
    canonical_path = tmp_path / "output/signals/latest_signals.json"
    canonical_path.parent.mkdir(parents=True)
    canonical_payload = {
        "schema": "manual_signal_scan_v1",
        "signals": [{"signal_id": "CANONICAL-SENTINEL"}],
    }
    canonical_path.write_text(json.dumps(canonical_payload), encoding="utf-8")

    report = active_swing_scan(
        tmp_path,
        symbols=["TEST"],
        observed_at=observed,
    )

    assert report["status"] == "GO"
    assert report["candidate_count"] == 1
    assert report["dedicated_fast_path"] is True
    assert report["canonical_money_signals_replaced"] is False
    assert report["canonical_signal_store_appended"] is False
    assert report["manual_execution_eligible"] is False
    assert report["portfolio_eligible"] is False
    assert report["execution_authority"] == "NONE"
    assert json.loads(canonical_path.read_text(encoding="utf-8")) == (canonical_payload)
    tactical = json.loads(
        (tmp_path / "output/signals/active_swing_15m_signals.json").read_text(encoding="utf-8")
    )
    assert tactical["candidate_count"] == 1
    assert tactical["signals"][0]["setup_id"]
    assert tactical["signals"][0]["portfolio_eligible"] is False
    assert tactical["automatic_orders"] == 0
    with SignalStore(tmp_path) as store:
        assert store.signal(tactical["signals"][0]["signal_id"]) is None


def test_expired_trigger_is_not_carried_to_next_session(tmp_path: Path) -> None:
    observed = _candidate_fixture(tmp_path)
    report = generate_active_swing_candidates(
        tmp_path,
        ["TEST"],
        observed_at=observed + timedelta(hours=4),
    )
    assert report["candidate_count"] == 0
    assert report["expired_triggers_are_not_carried_forward"] is True


def test_five_minute_resample_cannot_impersonate_native_15m(
    tmp_path: Path,
) -> None:
    _config(tmp_path)
    observed = datetime(2026, 8, 11, 20, 1, tzinfo=UTC)
    timestamps = pd.date_range(end="2026-08-11T19:45:00Z", periods=120, freq="15min")
    _write_frame(
        tmp_path,
        _bars(
            timestamps,
            [100.0] * 119 + [103.0],
            interval="15m",
            source_interval="5m",
            ingested_at=observed - timedelta(seconds=30),
        ),
        timeframe="15m",
        source_interval="5m",
    )

    report = generate_active_swing_candidates(tmp_path, ["TEST"], observed_at=observed)

    assert report["candidate_count"] == 0
    assert report["native_15m_ready_symbol_count"] == 0
    row = report["symbol_status"][0]
    assert row["timeframe_status"]["15m"] == "MISSING_CANONICAL_PARTITION"


def test_future_ingestion_is_not_backdated_into_candidate_history(
    tmp_path: Path,
) -> None:
    observed = _candidate_fixture(tmp_path)
    path = (
        tmp_path
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=TEST/interval=15m/source_interval=15m/bars.parquet"
    )
    frame = pd.read_parquet(path)
    frame["ingested_at"] = (observed + timedelta(minutes=1)).isoformat()
    frame.to_parquet(path, index=False)

    report = generate_active_swing_candidates(tmp_path, ["TEST"], observed_at=observed)

    assert report["candidate_count"] == 0
    assert report["native_15m_ready_symbol_count"] == 0
    assert report["symbol_status"][0]["timeframe_status"]["15m"] == (
        "NO_CAUSALLY_AVAILABLE_CLOSED_BARS"
    )


def test_strict_15m_expiration_is_not_extended_overnight() -> None:
    result = evaluate_signal_freshness(
        {
            "timeframe": "15m",
            "data_timestamp": "2026-08-11T20:00:00Z",
            "expiration_timestamp": "2026-08-12T00:00:00Z",
            "exchange_timezone": "America/New_York",
            "strict_expiration": True,
        },
        now=datetime(2026, 8, 12, 1, 0, tzinfo=UTC),
    )

    assert result["status"] == "STALE"
    assert result["is_current"] is False
    assert result["effective_expiration"] == datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
