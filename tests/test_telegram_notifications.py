from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from stocks.daily import run_daily
from stocks.notifications.telegram import (
    TelegramNotifier,
    TelegramQueue,
    _phase11_14_survivor_targets,
    format_order_event,
    format_signal_message,
    load_telegram_settings,
    notification_identity,
    signal_filter,
    telegram_command,
    telegram_alert,
    telegram_order_event,
    telegram_preview,
    telegram_send_pit_mtf,
    telegram_send_exit_signals,
    telegram_send_shadow_digest,
    telegram_send_regime_update,
)


def settings(root: Path, **overrides: str):
    env = {
        "TELEGRAM_NOTIFICATIONS_ENABLED": "true",
        "TELEGRAM_BOT_TOKEN": "unit-test-token-never-publish",
        "TELEGRAM_CHAT_ID": "-1001234567890",
        "TELEGRAM_DRY_RUN": "true",
        "TELEGRAM_SEND_SIGNALS": "true",
        "TELEGRAM_SEND_WATCHLIST": "true",
        "TELEGRAM_SEND_EXITS": "true",
        "TELEGRAM_SEND_ORDER_EVENTS": "true",
        "TELEGRAM_MIN_CONFIDENCE": "60",
        "TELEGRAM_MIN_REWARD_RISK": "1.5",
        **overrides,
    }
    return load_telegram_settings(root, env)


def signal(**overrides: Any) -> dict[str, Any]:
    base = {
        "signal_id": "SIG-1",
        "asset": "Apple Inc.",
        "ticker": "AAPL",
        "contract_identity": {"con_id": 265598},
        "asset_class": "STK",
        "exchange": "NASDAQ",
        "currency": "USD",
        "signal_timestamp": "2026-07-27T10:00:00+00:00",
        "data_timestamp": "2026-07-26T20:00:00+00:00",
        "data_freshness": "FRESH",
        "strategy_id": "QUALITY_MOMENTUM_V1",
        "strategy_dna_hash": "dna-1",
        "timeframe": "1d",
        "action": "BUY",
        "current_market_price": "214.40",
        "preferred_entry": "213.60",
        "entry_zone_low": "212.80",
        "entry_zone_high": "215.20",
        "stop_loss": "205.80",
        "stop_method": "2.0 ATR onder entry",
        "stop_distance_pct": "0.0365",
        "take_profit_1": "225.50",
        "take_profit_2": "237.00",
        "take_profit_mode": "PARTIAL_TARGETS_WITH_TRAILING_EXIT",
        "reward_risk_1": "1.5",
        "reward_risk_2": "3.0",
        "suggested_quantity": "1",
        "maximum_order_value_eur": "190.00",
        "maximum_planned_loss_eur": "6.84",
        "expected_holding_period": "2-6 weken",
        "confidence_score": "0.72",
        "reasons": ["POSITIEVE_TREND", "STERKE_RELATIEVE_STERKTE"],
        "risks": ["EARNINGS_OVER_12_DAGEN"],
        "expiration_timestamp": "2099-07-29T13:30:00+00:00",
        "lifecycle_status": "MANUAL_ACTIONABLE",
        "liquidity_status": "GO",
        "spread_status": "GO",
        "event_risk_status": "GO",
        "market_allowed": True,
    }
    base.update(overrides)
    return base


class FakeClient:
    def __init__(
        self,
        *,
        response: httpx.Response | None = None,
        error: Exception | None = None,
        **_: Any,
    ) -> None:
        self.response = response or httpx.Response(200, json={"ok": True})
        self.error = error
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get(self, url: str):
        self.calls.append(("GET", url, None))
        if self.error:
            raise self.error
        return self.response

    def post(self, url: str, json: dict[str, Any]):
        self.calls.append(("POST", url, json))
        if self.error:
            raise self.error
        return self.response


def factory(
    response: httpx.Response | None = None, error: Exception | None = None
):
    return lambda **kwargs: FakeClient(response=response, error=error, **kwargs)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "DISABLED_BY_CONFIG"),
        (
            {"TELEGRAM_NOTIFICATIONS_ENABLED": "true"},
            "DISABLED_MISSING_CONFIG",
        ),
        (
            {
                "TELEGRAM_NOTIFICATIONS_ENABLED": "true",
                "TELEGRAM_BOT_TOKEN": "x",
            },
            "DISABLED_MISSING_CONFIG",
        ),
    ],
)
def test_missing_or_disabled_config(
    tmp_path: Path, env: dict[str, str], expected: str
) -> None:
    assert load_telegram_settings(tmp_path, env).public_status == expected


def test_safe_settings_never_expose_secrets(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    encoded = json.dumps(configured.safe_dict())
    assert configured.token not in encoded
    assert configured.chat_id not in encoded
    assert configured.masked_chat_identity.startswith("chat-sha256:")


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "Type: Aandeel"),
        ({"asset_class": "ETF", "is_etf": True}, "Type: ETF"),
        (
            {"asset_class": "ETF", "instrument_subtype": "COMMODITY_ETF"},
            "Type: Commodity-ETF",
        ),
        (
            {"asset_class": "ETF", "instrument_subtype": "BOND_ETF"},
            "Type: Obligatie-ETF",
        ),
        (
            {"asset_class": "ETF", "instrument_subtype": "SECTOR_ETF"},
            "Type: Sector-ETF",
        ),
    ],
)
def test_stock_and_etf_message_types(
    tmp_path: Path, overrides: dict[str, Any], expected: str
) -> None:
    message = format_signal_message(signal(**overrides), settings(tmp_path))
    assert expected in message
    assert "Entryzone:" in message
    assert "Stop-loss:" in message
    assert "Take-profit 1:" in message
    assert "Take-profit 2:" in message
    assert "Voorgestelde positie:" in message
    assert "Maximaal gepland verlies:" in message
    assert "Automatische execution: uit." in message


def test_nl_currency_and_fx_context(tmp_path: Path) -> None:
    message = format_signal_message(
        signal(currency="EUR", preferred_entry="1234.56"), settings(tmp_path)
    )
    assert "€1.234,56" in message
    assert "FX: reeds verwerkt door signal/risklaag" in message


@pytest.mark.parametrize(
    "risk",
    [
        "EARNINGS_OVER_3_DAGEN",
        "EX_DIVIDEND_MORGEN",
        "MARKT_GESLOTEN",
        "EARLY_CLOSE",
        "DELAYED_DATA",
        "ABNORMAL_GAP",
    ],
)
def test_relevant_signal_warnings_are_visible(tmp_path: Path, risk: str) -> None:
    assert risk in format_signal_message(
        signal(risks=[risk]), settings(tmp_path)
    )


def test_trailing_exit_is_visible(tmp_path: Path) -> None:
    message = format_signal_message(
        signal(take_profit_2=None, take_profit_mode="CHANDELIER_TRAILING"),
        settings(tmp_path),
    )
    assert "CHANDELIER_TRAILING" in message


def test_fractional_position_size_is_visible(tmp_path: Path) -> None:
    message = format_signal_message(
        signal(suggested_quantity="0.125"), settings(tmp_path)
    )
    assert "0,125 aandelen" in message


def test_future_requires_complete_contract(tmp_path: Path) -> None:
    incomplete = signal(
        asset_class="FUT",
        contract_identity={"con_id": 1},
        futures_sleeve_active=True,
        strategy_supports_futures=True,
    )
    assert signal_filter(incomplete, settings(tmp_path)) == (
        "SKIPPED_FILTER",
        "BLOCKED_INCOMPLETE_FUTURES_CONTRACT",
    )
    complete = signal(
        asset_class="FUT",
        contract_identity={
            "con_id": 1,
            "expiry": "202612",
            "multiplier": "100",
            "minimum_tick": "0.1",
            "tick_value": "10",
            "rollover_status": "SAFE_OUTSIDE_ROLL_WINDOW",
            "margin_eur": "12000",
            "first_notice_date": "2026-11-30",
            "last_trade_date": "2026-12-28",
        },
        futures_sleeve_active=True,
        strategy_supports_futures=True,
    )
    assert signal_filter(complete, settings(tmp_path))[0] == "PENDING"
    message = format_signal_message(complete, settings(tmp_path))
    assert "Multiplier: 100" in message
    assert "rolloverrisico" in message
    assert "Geschatte margin" in message


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"action": "NO_SIGNAL"}, "NO_SIGNAL"),
        ({"data_freshness": "STALE"}, "STALE_OR_EXPIRED"),
        ({"confidence_score": "0.59"}, "MINIMUM_CONFIDENCE"),
        ({"reward_risk_1": "1.49"}, "MINIMUM_REWARD_RISK"),
        (
            {"lifecycle_status": "WATCHLIST"},
            "MANUAL_ACTIONABLE_AUTHORITY_REQUIRED",
        ),
        ({"action": "HOLD"}, "HOLD_WITHOUT_MATERIAL_CHANGE"),
    ],
)
def test_signal_filters(
    tmp_path: Path, overrides: dict[str, Any], reason: str
) -> None:
    assert signal_filter(signal(**overrides), settings(tmp_path))[1] == reason


def test_watchlist_filter_can_be_disabled(tmp_path: Path) -> None:
    item = signal(action="WATCHLIST", lifecycle_status="WATCHLIST")
    assert signal_filter(
        item, settings(tmp_path, TELEGRAM_SEND_WATCHLIST="false")
    )[1] == "WATCHLIST_DISABLED"


def test_duplicate_and_material_changes(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    notifier = TelegramNotifier(tmp_path, configured, sleep=lambda _: None)
    first = notifier.enqueue_signal(signal())
    duplicate = notifier.enqueue_signal(signal())
    changed_stop = notifier.enqueue_signal(signal(stop_loss="204.00"))
    changed_target = notifier.enqueue_signal(signal(take_profit_1="226.00"))
    assert first["status"] == "PENDING"
    assert duplicate["status"] == "SKIPPED_DUPLICATE"
    assert changed_stop["status"] == "PENDING"
    assert changed_target["status"] == "PENDING"
    assert notification_identity(signal()) != notification_identity(
        signal(stop_loss="204.00")
    )


def test_exit_message_is_observation_only_and_filterable(
    tmp_path: Path,
) -> None:
    exit_signal = signal(
        action="EXIT",
        lifecycle_status="EXIT",
        material_lifecycle_change=True,
        source="SIGNAL_LIFECYCLE",
        position_scope="MODEL_LIFECYCLE_ONLY",
        reason_codes=["SIGNAL_INVALIDATED"],
        current_r="1.25",
        peak_r="2.00",
        profit_giveback="0.375",
    )

    assert signal_filter(exit_signal, settings(tmp_path)) == ("PENDING", None)
    message = format_signal_message(exit_signal, settings(tmp_path))

    assert "EXIT REVIEW" in message
    assert "SIGNAL_INVALIDATED" in message
    assert "Huidige R: 1,25R" in message
    assert "Profit giveback: 37,5%" in message
    assert "Geen brokerorder gegenereerd." in message
    assert "Executionauthority: NONE" in message


def test_exit_sender_combines_lifecycle_and_position_advisories_without_orders(
    tmp_path: Path,
) -> None:
    signal_root = tmp_path / "output" / "signals"
    signal_root.mkdir(parents=True)
    (signal_root / "latest_signals.json").write_text(
        json.dumps(
            {
                "signals": [
                    signal(
                        signal_id="AAPL-CONTEXT",
                        ticker="AAPL",
                        action="WATCHLIST",
                        lifecycle_status="WATCHLIST",
                    ),
                    signal(
                        signal_id="MSFT-CONTEXT",
                        ticker="MSFT",
                        asset="Microsoft",
                        action="WATCHLIST",
                        lifecycle_status="WATCHLIST",
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )
    lifecycle = tmp_path / "output" / "operations" / "signal-lifecycle.json"
    lifecycle.parent.mkdir(parents=True)
    lifecycle.write_text(
        json.dumps(
            {
                "status": "GO",
                "rows": [
                    {
                        "ticker": "AAPL",
                        "strategy_id": "QUALITY_MOMENTUM_V1",
                        "lifecycle_status": "EXIT",
                        "reason_codes": ["SIGNAL_INVALIDATED"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    management = (
        tmp_path / "output" / "portfolio" / "position-management.json"
    )
    management.parent.mkdir(parents=True)
    management.write_text(
        json.dumps(
            {
                "status": "GO",
                "positions": [
                    {
                        "ticker": "MSFT",
                        "position_identity": "position-fingerprint",
                        "advisory_action": "REDUCE_50",
                        "reason_codes": ["PROFIT_GIVEBACK_40_PERCENT"],
                        "current_r": 1.8,
                        "peak_r": 3.0,
                        "profit_giveback": 0.4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    notifier = TelegramNotifier(
        tmp_path,
        settings(tmp_path),
        sleep=lambda _seconds: None,
    )

    first = telegram_send_exit_signals(tmp_path, notifier)
    duplicate = telegram_send_exit_signals(tmp_path, notifier)
    drained = telegram_send_exit_signals(tmp_path, notifier)

    assert first["status"] == "GO"
    assert first["outcome"] == "EXIT_ADVISORIES_PROCESSED"
    assert first["signals_considered"] == 2
    assert first["enqueue_counts"] == {"PENDING": 2}
    assert first["delivery"]["processed"] == 1
    assert duplicate["enqueue_counts"] == {"SKIPPED_DUPLICATE": 2}
    assert duplicate["delivery"]["processed"] == 1
    assert drained["enqueue_counts"] == {"SKIPPED_DUPLICATE": 2}
    assert drained["delivery"]["processed"] == 0
    assert first["automatic_execution_allowed"] is False
    assert first["execution_authority"] == "NONE"
    assert first["broker_calls"] == 0
    assert first["orders_generated"] == 0
    artifact = json.loads(
        (
            tmp_path
            / "output"
            / "notifications"
            / "latest_exit_delivery.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["orders_generated"] == 0
    with TelegramQueue(tmp_path) as queue:
        rows = queue.connection.execute(
            """
            SELECT message_type, message_text, status FROM notifications
            WHERE message_type LIKE '%_EXIT' ORDER BY message_type
            """
        ).fetchall()
    assert len(rows) == 2
    assert all(row["status"] == "SENT" for row in rows)
    assert all(
        "Geen brokerorder gegenereerd." in row["message_text"]
        for row in rows
    )
    assert all(
        "Executionauthority: NONE" in row["message_text"] for row in rows
    )


def test_exit_sender_no_current_advisories_is_a_safe_noop(
    tmp_path: Path,
) -> None:
    notifier = TelegramNotifier(
        tmp_path,
        settings(tmp_path),
        sleep=lambda _seconds: None,
    )

    report = telegram_send_exit_signals(tmp_path, notifier)

    assert report["status"] == "GO"
    assert report["outcome"] == "NO_CURRENT_EXIT_ADVISORIES"
    assert report["signals_considered"] == 0
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0


def test_pit_mtf_sender_publishes_shadow_watchlist_without_authority(
    tmp_path: Path,
) -> None:
    configured = settings(tmp_path)
    notifier = TelegramNotifier(
        tmp_path, configured, sleep=lambda _seconds: None
    )
    path = tmp_path / "output" / "signals" / "pit_mtf_signals.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                signal(
                    signal_id="MTF-SIG-1",
                    strategy_id="MTF-ALLOWLISTED",
                    timeframe="4h",
                    lifecycle_status="SHADOW",
                    suggested_quantity="0",
                    automatic_execution_allowed=False,
                )
            ]
        ),
        encoding="utf-8",
    )
    research_path = (
        tmp_path
        / "output"
        / "signals"
        / "pit_mtf_research_signals.json"
    )
    research_path.write_text(
        json.dumps(
            [
                signal(
                    signal_id="MTF-SIG-1",
                    strategy_id="MTF-ALLOWLISTED",
                    timeframe="4h",
                ),
                signal(
                    signal_id="MTF-SIG-2",
                    strategy_id="MTF-TEN-BPS-RESEARCH",
                    timeframe="1h",
                ),
            ]
        ),
        encoding="utf-8",
    )

    with TelegramQueue(tmp_path) as queue:
        queue.enqueue(
            identity={"signal_id": "MTF-SIG-OLD"},
            message="obsolete individual MTF signal",
            message_type="AANDEEL_WATCHLIST",
        )
        queue.enqueue(
            identity={"signal_id": "OTHER-SIG-KEEP"},
            message="unrelated pending signal",
            message_type="AANDEEL_WATCHLIST",
        )

    first = telegram_send_pit_mtf(tmp_path, notifier)
    duplicate = telegram_send_pit_mtf(tmp_path, notifier)

    assert first["signals_considered"] == 2
    assert first["source_counts"] == {
        "robust_shortlist": 1,
        "ten_bps_positive_research": 2,
    }
    assert first["deduplicated_signal_count"] == 2
    assert first["selected_signal_count"] == 2
    assert first["notification_count"] == 1
    assert first["superseded_pending_count"] == 1
    assert first["enqueue_counts"] == {"PENDING": 1}
    assert duplicate["enqueue_counts"] == {"SKIPPED_DUPLICATE": 1}
    assert first["failure_is_non_blocking"] is True
    assert first["signal_authority"] == "SHADOW_ONLY"
    assert first["execution_authority"] == "NONE"
    assert first["broker_calls"] == 0
    assert first["orders_generated"] == 0
    with TelegramQueue(tmp_path) as queue:
        mtf_row = queue.connection.execute(
            """
            SELECT status, error_code FROM notifications
            WHERE signal_id='MTF-SIG-OLD'
            """
        ).fetchone()
        unrelated = queue.connection.execute(
            """
            SELECT status FROM notifications
            WHERE signal_id='OTHER-SIG-KEEP'
            """
        ).fetchone()
        digest = queue.connection.execute(
            """
            SELECT message_text FROM notifications
            WHERE message_type='LOWER_TIMEFRAME_SHADOW_DIGEST'
            """
        ).fetchone()
    assert tuple(mtf_row) == (
        "SKIPPED_FILTER",
        "SUPERSEDED_BY_COMPACT_MTF_DIGEST",
    )
    assert unrelated["status"] == "SENT"
    assert "2 gediversifieerde signalen" in digest["message_text"]
    assert "Executionauthority: NONE" in digest["message_text"]


def test_pit_mtf_sender_compacts_large_universe_and_caps_asset_concentration(
    tmp_path: Path,
) -> None:
    configured = settings(tmp_path)
    notifier = TelegramNotifier(
        tmp_path, configured, sleep=lambda _seconds: None
    )
    path = (
        tmp_path
        / "output"
        / "signals"
        / "pit_mtf_research_signals.json"
    )
    path.parent.mkdir(parents=True)
    rows = []
    timeframes = ["1h", "2h", "4h", "1d"]
    for index in range(184):
        rows.append(
            signal(
                signal_id=f"MTF-SIG-{index:03d}",
                ticker=f"SYM{index % 20:02d}",
                asset=f"Asset {index % 20:02d}",
                timeframe=timeframes[index % len(timeframes)],
                strategy_id=f"MTF-STRATEGY-{index:03d}",
                confidence_score=f"{0.90 - index / 1000:.3f}",
                current_pit_attested=False,
                architecture=f"architecture-{index:03d}",
            )
        )
    path.write_text(json.dumps(rows), encoding="utf-8")

    result = telegram_send_pit_mtf(tmp_path, notifier)

    assert result["signals_considered"] == 184
    assert result["selected_signal_count"] == 10
    assert result["suppressed_low_priority_count"] == 174
    assert result["notification_count"] == 1
    assert result["execution_authority"] == "NONE"
    assert result["broker_calls"] == 0
    assert result["orders_generated"] == 0
    with TelegramQueue(tmp_path) as queue:
        digests = queue.connection.execute(
            """
            SELECT message_text FROM notifications
            WHERE message_type='LOWER_TIMEFRAME_SHADOW_DIGEST'
            """
        ).fetchall()
    assert len(digests) == 1
    message = digests[0]["message_text"]
    assert "10 gediversifieerde signalen uit 184" in message
    published_symbols = [
        line.split()[1]
        for line in message.splitlines()
        if line[:1].isdigit() and ". " in line
    ]
    assert max(published_symbols.count(symbol) for symbol in published_symbols) <= 2


def _write_hmm_state(
    root: Path,
    *,
    as_of: str,
    risk_on: float,
    neutral: float,
    stress: float,
    multiplier: float,
) -> None:
    path = root / "output" / "research" / "phase11_11" / "current.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "GO",
                "state": {
                    "as_of": as_of,
                    "model_hash": "MODEL-HASH",
                    "probabilities": {
                        "RISK_ON_TREND": risk_on,
                        "NEUTRAL_CHOPPY": neutral,
                        "STRESS_HIGH_VOL": stress,
                    },
                    "regime_multiplier": multiplier,
                },
            }
        ),
        encoding="utf-8",
    )


def test_hmm_regime_notification_is_material_and_deduplicated(
    tmp_path: Path,
) -> None:
    configured = settings(tmp_path)
    notifier = TelegramNotifier(tmp_path, configured, sleep=lambda _: None)
    _write_hmm_state(
        tmp_path,
        as_of="2026-07-24T00:00:00",
        risk_on=0.61,
        neutral=0.38,
        stress=0.01,
        multiplier=0.84,
    )

    initial = telegram_send_regime_update(tmp_path, notifier)
    unchanged = telegram_send_regime_update(tmp_path, notifier)
    _write_hmm_state(
        tmp_path,
        as_of="2026-07-27T00:00:00",
        risk_on=0.10,
        neutral=0.25,
        stress=0.65,
        multiplier=0.15,
    )
    transition = telegram_send_regime_update(tmp_path, notifier)

    assert initial["outcome"] == "MATERIAL_REGIME_CHANGE_PUBLISHED"
    assert unchanged["outcome"] == "NO_MATERIAL_REGIME_CHANGE"
    assert transition["risk_band"] == "ENTRY_BLOCK"
    assert transition["stress_band"] == "CRITICAL"
    assert "DOMINANT_STATE_CHANGED" in transition["change_reasons"]
    assert transition["execution_authority"] == "NONE"
    assert transition["orders_generated"] == 0
    assert notifier.health(probe_api=False)["sent_type_counts"][
        "HMM_REGIME_ALERT"
    ] == 2


def test_hmm_regime_notification_blocks_invalid_probabilities(
    tmp_path: Path,
) -> None:
    _write_hmm_state(
        tmp_path,
        as_of="2026-07-24T00:00:00",
        risk_on=0.80,
        neutral=0.50,
        stress=0.10,
        multiplier=0.80,
    )

    report = telegram_send_regime_update(tmp_path)

    assert report["status"] == "DATA_BLOCKED"
    assert report["notification_queued"] is False
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0


def test_dry_run_sends_no_http_and_places_no_orders(tmp_path: Path) -> None:
    notifier = TelegramNotifier(
        tmp_path, settings(tmp_path), client_factory=factory(), sleep=lambda _: None
    )
    notifier.enqueue_signal(signal())
    report = notifier.process()
    assert report["counts"]["SENT"] == 1
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0


def test_http_success_and_health_are_sanitized(tmp_path: Path) -> None:
    configured = settings(tmp_path, TELEGRAM_DRY_RUN="false")
    notifier = TelegramNotifier(
        tmp_path, configured, client_factory=factory(), sleep=lambda _: None
    )
    notifier.enqueue_signal(signal())
    assert notifier.process()["counts"]["SENT"] == 1
    health = notifier.health()
    encoded = json.dumps(health)
    assert health["api_reachable"] is True
    assert configured.token not in encoded
    assert configured.chat_id not in encoded


@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        (
            httpx.Response(
                429,
                json={"parameters": {"retry_after": 7}},
            ),
            "TELEGRAM_HTTP_429",
        ),
        (httpx.Response(500), "TELEGRAM_HTTP_500"),
    ],
)
def test_http_retry_and_429(
    tmp_path: Path, response: httpx.Response, error_code: str
) -> None:
    configured = settings(tmp_path, TELEGRAM_DRY_RUN="false")
    notifier = TelegramNotifier(
        tmp_path,
        configured,
        client_factory=factory(response=response),
        sleep=lambda _: None,
    )
    notifier.enqueue_signal(signal())
    report = notifier.process()
    assert report["counts"]["RETRY_PENDING"] == 1
    assert report["last_error"] == error_code


def test_timeout_and_failed_final(tmp_path: Path) -> None:
    configured = settings(
        tmp_path, TELEGRAM_DRY_RUN="false", TELEGRAM_MAX_RETRIES="0"
    )
    notifier = TelegramNotifier(
        tmp_path,
        configured,
        client_factory=factory(error=httpx.ReadTimeout("bounded timeout")),
        sleep=lambda _: None,
    )
    notifier.enqueue_signal(signal())
    report = notifier.process()
    assert report["counts"]["FAILED_FINAL"] == 1
    assert report["last_error"] == "READTIMEOUT"


def test_pending_queue_restart_recovers_sending(tmp_path: Path) -> None:
    with TelegramQueue(tmp_path) as queue:
        notification_id, _ = queue.enqueue(
            identity={"id": 1}, message="safe", message_type="TEST"
        )
        queue.mark_sending(notification_id)
    with TelegramQueue(tmp_path) as recovered:
        assert recovered.counts()["FAILED_FINAL"] == 1
        assert (
            recovered.summary()["last_error"]
            == "DELIVERY_OUTCOME_UNKNOWN_REVIEW_REQUIRED"
        )


def test_pending_signal_quarantine_keeps_only_current_actionable(
    tmp_path: Path,
) -> None:
    with TelegramQueue(tmp_path) as queue:
        for signal_id in ("SIG-KEEP", "SIG-BLOCK"):
            queue.enqueue(
                identity={"signal_id": signal_id},
                message=signal_id,
                message_type="STOCK_BUY",
            )
        assert queue.quarantine_pending_signals_except({"SIG-KEEP"}) == 1
        due = queue.due(10)
        assert [row["signal_id"] for row in due] == ["SIG-KEEP"]


@pytest.mark.parametrize(
    "status",
    [
        "SUBMITTING",
        "SUBMITTED",
        "PARTIAL_FILL",
        "FILLED",
        "CANCELLED",
        "REJECTED",
    ],
)
def test_order_status_messages_mask_order_identity(status: str) -> None:
    event = {
        "status": status,
        "symbol": "SPY",
        "side": "BUY",
        "order_type": "LIMIT",
        "quantity": 2,
        "filled_quantity": 1,
        "remaining_quantity": 1,
        "average_fill_price": 500,
        "environment": "PAPER",
        "order_reference": "raw-order-123",
    }
    message = format_order_event(event)
    assert status.replace("_", " ") in message
    assert "raw-order-123" not in message
    assert "order-sha256:" in message


def test_order_event_adapter_never_places_order(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    notifier = TelegramNotifier(tmp_path, configured, sleep=lambda _: None)
    report = telegram_order_event(
        tmp_path,
        {"status": "SUBMITTED", "symbol": "SPY", "order_reference": "1"},
        notifier=notifier,
    )
    assert report["execution_authority"] == "NONE"
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0


@pytest.mark.parametrize(
    "event_type",
    [
        "IBKR_DISCONNECT",
        "IBKR_RECONNECT",
        "RECONCILIATION_MISMATCH",
        "KILL_SWITCH_ACTIVE",
    ],
)
def test_system_and_risk_alerts_never_change_authority(
    tmp_path: Path, event_type: str
) -> None:
    configured = settings(tmp_path)
    notifier = TelegramNotifier(tmp_path, configured, sleep=lambda _: None)
    report = telegram_alert(
        tmp_path,
        {
            "category": "RISK" if "KILL" in event_type else "SYSTEM",
            "event_type": event_type,
            "status": "ACTIVE",
            "observed_at": "2026-07-27T00:00:00Z",
        },
        notifier=notifier,
    )
    assert report["status"] == "PENDING"
    assert report["execution_authority"] == "NONE"
    assert report["orders_generated"] == 0


def test_preview_is_local_only(tmp_path: Path) -> None:
    path = tmp_path / "output" / "signals"
    path.mkdir(parents=True)
    path.joinpath("latest_signals.json").write_text(
        json.dumps({"signals": [signal()]}), encoding="utf-8"
    )
    report = telegram_preview(tmp_path, settings(tmp_path))
    assert report["sent"] == 0
    assert Path(report["text_path"]).exists()
    assert Path(report["html_path"]).exists()


def test_safe_test_command_has_zero_broker_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    monkeypatch.setenv("TELEGRAM_DRY_RUN", "true")
    report = telegram_command(tmp_path, "test")
    assert report["status"] == "GO"
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0
    assert report["execution_authority"] == "NONE"


def test_status_surfaces_safety_evidence_without_changing_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_NOTIFICATIONS_ENABLED", "false")
    paths = {
        "output/ibkr/phase9/status.json": {
            "status": "NO_GO",
            "open_blockers": ["closing_sell_canary"],
        },
        "output/ibkr/live/writer-integrity-verify.json": {
            "status": "NO_GO",
            "writer_hash_integrity": False,
        },
        "output/ibkr/data-capabilities/capability-matrix.json": {
            "summary": {"tick_by_tick_trades": "UNAVAILABLE_ENTITLEMENT"},
            "missing_subscription_classes": ["TICK_BY_TICK"],
        },
        "output/research/active_swing/selective_ml/status.json": {
            "status": "NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS",
            "closed_trainable_episode_count": 0,
        },
        "output/research/sec_intelligence/status.json": {
            "status": "DEGRADED",
            "structured_event_count": 0,
        },
    }
    for relative, payload in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = telegram_command(tmp_path, "status")

    safety = report["safety_status"]
    assert safety["phase9"] == "NO_GO"
    assert safety["writer_hash_integrity"] is False
    assert safety["canonical_ml_closed_labels"] == 0
    assert safety["sec_structured_event_count"] == 0
    assert safety["execution_authority"] == "NONE"


def test_telegram_failure_does_not_stop_daily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "stocks.daily.telegram_daily_delivery",
        lambda *_args, **_kwargs: {
            "status": "DEGRADED",
            "failure_is_non_blocking": True,
        },
    )
    monkeypatch.setattr(
        "stocks.daily.screener_run", lambda *_args, **_kwargs: {"status": "GO"}
    )
    monkeypatch.setattr(
        "stocks.daily.signal_scan",
        lambda *_args, **_kwargs: {
            "status": "GO",
            "signals": [],
            "broker_calls": 0,
            "orders_generated": 0,
        },
    )
    monkeypatch.setattr(
        "stocks.daily.signal_status", lambda *_args: {"status": "GO"}
    )
    report = run_daily(tmp_path, signals_only=True)
    assert report["status"] == "GO"
    assert report["telegram"]["status"] == "DEGRADED"
    assert report["orders_generated"] == 0


def test_public_artifacts_contain_no_token_or_chat_id(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    notifier = TelegramNotifier(tmp_path, configured, sleep=lambda _: None)
    notifier.enqueue_signal(signal())
    notifier.process()
    public = (tmp_path / "output" / "notifications").read_text if False else None
    del public
    contents = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "output" / "notifications").glob("*")
        if path.is_file()
    )
    assert configured.token not in contents
    assert configured.chat_id not in contents
    assert "IBKR-orders geplaatst" not in contents or "0" in contents


def test_shadow_digest_is_deduplicated_and_never_grants_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = settings(tmp_path)
    monkeypatch.setattr(
        "stocks.notifications.telegram.load_telegram_settings",
        lambda _root: configured,
    )
    monkeypatch.setattr(
        "stocks.notifications.telegram.build_market_intelligence_digest",
        lambda _root: {
            "upcoming_macro_events": [
                {
                    "event_id": "FOMC",
                    "name": "Rate decision",
                    "scheduled_at": "2026-07-29T18:00:00Z",
                    "window_status": "WITHIN_24H",
                }
            ],
            "important_news": [
                {"affected_markets": ["US_EQUITIES"]}
            ],
        },
    )
    root = tmp_path / "output" / "research" / "phase11_12"
    root.mkdir(parents=True)
    root.joinpath("latest-shadow-observation.json").write_text(
        json.dumps(
            {
                "active_signals": [
                    {
                        "strategy_id": "BULK-1",
                        "symbol": "AAPL",
                        "timeframe": "4h",
                        "formula": "flow_consensus",
                        "profile": "balanced",
                        "closed_bar_timestamp": "2026-07-28T16:00:00Z",
                        "score": 1.0,
                        "reference_close": 200,
                        "illustrative_stop": 190,
                        "illustrative_target_1": 210,
                        "illustrative_target_2": 215,
                        "current_shariah_attestation": "CURRENTLY_ATTESTED",
                        "action": "SHADOW_ENTRY_CANDIDATE",
                    }
                ],
                "forward_evidence": {
                    "closed_episode_count": 12,
                    "open_episode_count": 1,
                    "pending_entry_count": 2,
                    "aggregate": {
                        "net_profit_factor": 1.2,
                        "sample_status": "LOW_CONFIDENCE",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    survivor_root = (
        tmp_path / "output" / "research" / "phase11_14"
    )
    survivor_root.mkdir(parents=True)
    survivor_root.joinpath("status.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "qualification_boundary": {
                    "status": "FROZEN",
                    "qualification_hash": "Q14",
                    "robust_strategy_ids": ["P1114-ONE"],
                },
                "qualification": {
                    "strategies": [
                        {
                            "strategy_id": "P1114-ONE",
                            "robust_pass": True,
                            "portfolio_invariants_go": True,
                            "forward_observer_candidate": True,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    survivor_root.joinpath(
        "latest-forward-observation.json"
    ).write_text(
        json.dumps(
            {
                "status": "GO",
                "qualification_hash": "Q14",
                "EXECUTION_AUTHORITY": "NONE",
                "ORDER_CALLS": 0,
                "BROKER_CALLS": 0,
                "observations": [
                    {
                        "strategy_id": "P1114-ONE",
                        "symbol": "AAPL",
                        "timeframe": "4h",
                        "formula": "choppiness_breakout",
                        "profile": "balanced",
                        "closed_bar_timestamp": (
                            "2026-07-28T16:00:00Z"
                        ),
                        "raw_active_signals": [
                            {"symbol": "AAPL", "score": 0.8}
                        ],
                        "current_attested_target_weights": {
                            "AAPL": 0.25
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    notifier = TelegramNotifier(
        tmp_path,
        configured,
        sleep=lambda _: None,
    )

    first = telegram_send_shadow_digest(tmp_path, notifier)
    second = telegram_send_shadow_digest(tmp_path, notifier)

    assert first["execution_authority"] == "NONE"
    assert first["orders_generated"] == 0
    assert first["survivor_targets_considered"] == 1
    assert first["survivor_observation_status"] == "GO"
    assert second["enqueue_status"] == "SKIPPED_DUPLICATE"
    with TelegramQueue(tmp_path) as queue:
        message = queue.connection.execute(
            """
            SELECT message_text FROM notifications
            WHERE message_type='LOWER_TIMEFRAME_SHADOW_DIGEST'
            """
        ).fetchone()[0]
    assert "Prospective episodes: 12 gesloten" in message
    assert "Forward netto PF: 1.2" in message
    assert "Frozen nested survivor targets:" in message
    assert "Research target 25.0%" in message
    assert "Executionauthority: NONE" in message


def test_notifier_process_can_bound_shared_queue_work(
    tmp_path: Path,
) -> None:
    configured = settings(tmp_path)
    notifier = TelegramNotifier(
        tmp_path,
        configured,
        sleep=lambda _: None,
    )
    for index in range(3):
        notifier.enqueue_text(
            message=f"bounded-{index}",
            message_type="BOUNDED_TEST",
            identity={"type": "BOUNDED_TEST", "index": index},
        )

    report = notifier.process(max_items=1)

    assert report["processed"] == 1
    assert report["counts"]["SENT"] == 1
    assert report["counts"]["PENDING"] == 2


def test_survivor_digest_blocks_hash_mismatch_and_unattested_raw_signal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output" / "research" / "phase11_14"
    root.mkdir(parents=True)
    status = {
        "status": "GO",
        "qualification_boundary": {
            "status": "FROZEN",
            "qualification_hash": "EXPECTED",
            "robust_strategy_ids": ["P1114-ONE"],
        },
        "qualification": {
            "strategies": [
                {
                    "strategy_id": "P1114-ONE",
                    "robust_pass": True,
                    "portfolio_invariants_go": True,
                    "forward_observer_candidate": True,
                }
            ]
        },
    }
    observation = {
        "status": "GO",
        "qualification_hash": "MISMATCH",
        "EXECUTION_AUTHORITY": "NONE",
        "ORDER_CALLS": 0,
        "BROKER_CALLS": 0,
        "observations": [],
    }
    (root / "status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    observation_path = root / "latest-forward-observation.json"
    observation_path.write_text(
        json.dumps(observation),
        encoding="utf-8",
    )

    blocked, blocked_audit = _phase11_14_survivor_targets(tmp_path)

    assert blocked == []
    assert blocked_audit["status"] == (
        "QUALIFICATION_OR_AUTHORITY_BLOCKED"
    )

    observation["qualification_hash"] = "EXPECTED"
    observation["observations"] = [
        {
            "strategy_id": "P1114-ONE",
            "timeframe": "4h",
            "formula": "breakout",
            "raw_active_signals": [
                {"symbol": "AAPL", "score": 0.9}
            ],
            "current_attested_target_weights": {},
        }
    ]
    observation_path.write_text(
        json.dumps(observation),
        encoding="utf-8",
    )

    unattested, audit = _phase11_14_survivor_targets(tmp_path)

    assert unattested == []
    assert audit["status"] == "GO"
    assert audit["target_count"] == 0
