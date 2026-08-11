from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import stocks.live.automatic as automatic
from stocks.live.quote import regular_session_open, validate_quote


def test_candidate_selection_enforces_allowlist_and_new_closed_bar() -> None:
    allowlist, observation, signal = _selection_fixture()

    selected, blockers = automatic.select_candidate(
        allowlist, observation, [signal]
    )
    unauthorized, unauthorized_blockers = automatic.select_candidate(
        allowlist,
        observation,
        [{**signal, "strategy_id": "UNAUTHORIZED"}],
    )
    same_bar, same_bar_blockers = automatic.select_candidate(
        allowlist,
        observation,
        [{**signal, "data_timestamp": "2026-07-20T00:00:00+00:00"}],
    )

    assert selected is not None
    assert selected["symbol"] == "TEST"
    assert blockers == []
    assert unauthorized is None
    assert "UNAUTHORIZED_STRATEGY_REJECTED" in unauthorized_blockers
    assert same_bar is None
    assert "SAME_OR_PRIOR_BAR_REJECTED" in same_bar_blockers


def test_pit_mtf_observation_is_selectable_with_exact_strategy_id() -> None:
    allowlist, observation, signal = _selection_fixture()
    observation["observations"][0]["observation_status"] = (
        "PIT_OBSERVATION_COMPLETE"
    )

    selected, blockers = automatic.select_candidate(
        allowlist, observation, [signal]
    )

    assert blockers == []
    assert selected is not None
    assert selected["strategy_id"] == "STRATEGY"
    assert selected["signal_id"] == "SIGNAL"


def test_observation_merge_preserves_phase11_10_and_phase11_13() -> None:
    merged = automatic._merge_observations(
        {
            "schema": "phase11_13",
            "observations": [{"strategy_id": "FAST"}],
        },
        {
            "schema": "phase11_10",
            "observations": [{"strategy_id": "MTF"}],
        },
        {
            "schema": "phase11_14",
            "observations": [{"strategy_id": "SURVIVOR"}],
        },
    )

    assert merged["source_schemas"] == [
        "phase11_13",
        "phase11_10",
        "phase11_14",
    ]
    assert [row["strategy_id"] for row in merged["observations"]] == [
        "FAST",
        "MTF",
        "SURVIVOR",
    ]


def test_phase11_14_observation_publishes_exact_live_candidate() -> None:
    payload = {
        "schema": "phase11_14_forward_observation_v2",
        "observations": [
            {
                "strategy_id": "SURVIVOR",
                "independent_forward_session": True,
                "data_freshness": "FRESH_CLOSED_BAR",
                "closed_bar_timestamp": "2026-07-29T17:30:00+00:00",
                "current_attested_target_weights": {"AAPL": 0.25},
                "raw_active_signals": [
                    {
                        "symbol": "AAPL",
                        "signal_id": "SIGNAL",
                        "action": "BUY",
                        "data_freshness": "FRESH",
                        "data_timestamp": "2026-07-29T17:30:00+00:00",
                        "execution_envelope_status": "GO",
                        "stop_loss": 280,
                        "take_profit_1": 330,
                        "take_profit_2": 345,
                        "confidence_score": 1.0,
                    },
                    {
                        "symbol": "XOM",
                        "execution_envelope_status": "GO",
                    },
                ],
            }
        ],
    }

    signals = automatic._observation_signals(payload)

    assert len(signals) == 1
    assert signals[0]["strategy_id"] == "SURVIVOR"
    assert signals[0]["ticker"] == "AAPL"
    assert signals[0]["stop_loss"] == 280
    assert signals[0]["take_profit_1"] == 330


def test_quote_and_regular_session_gates() -> None:
    quote = validate_quote({"bid": "9.99", "ask": "10.01"})
    wide = validate_quote({"bid": "9", "ask": "10"})
    session = regular_session_open(
        "NASDAQ",
        datetime(2026, 7, 27, 15, 0, tzinfo=UTC),
    )
    closed = regular_session_open(
        "NASDAQ",
        datetime(2026, 7, 26, 15, 0, tzinfo=UTC),
    )

    assert quote["quote_validation_status"] == "GO"
    assert wide["quote_validation_status"] == "NO_GO"
    assert "LIVE_SPREAD_TOO_WIDE" in wide["quote_blockers"]
    assert session["session_status"] == "REGULAR_SESSION_OPEN"
    assert closed["session_status"] == "MARKET_CLOSED"


def test_automatic_cycle_requires_authority_and_calls_no_broker(
    tmp_path: Path,
) -> None:
    result = automatic.automatic_cycle(tmp_path)

    assert result["status"] == "NO_GO"
    assert result["cycle_status"] == "AUTHORITY_NOT_GRANTED"
    assert result["market_data_calls"] == 0
    assert result["live_place_order_calls"] == 0


def test_automatic_cycle_routes_one_whole_share_through_submitter(
    tmp_path: Path, monkeypatch
) -> None:
    allowlist, observation, signal = _selection_fixture()
    _write_json(
        tmp_path / "output" / "ibkr" / "live" / "strategy-allowlist.json",
        allowlist,
    )
    _write_json(
        tmp_path
        / "output"
        / "research"
        / "phase11_13"
        / "latest-forward-observation.json",
        observation,
    )
    _write_json(
        tmp_path / "output" / "signals" / "active_signals.json",
        [signal],
    )
    _write_json(
        tmp_path / "output" / "ibkr" / "live" / "capital-safety.json",
        {"status": "GO", "buying_power_sufficient": True},
    )
    _write_json(
        tmp_path / "output" / "capital" / "daily_profit_target.json",
        {
            "status": "GO",
            "session_date": "2026-07-27",
            "input_source": (
                "BROKER_RECONCILED_EQUITY_AND_CASH_CHANGE"
            ),
            "enforcement_active": True,
            "target_reached": False,
            "new_entries_allowed": True,
            "risk_chasing_allowed": False,
            "risk_reducing_exits_allowed": True,
        },
    )
    contract_path = (
        tmp_path / "output" / "ibkr" / "contracts" / "stocks.parquet"
    )
    contract_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "con_id": 123,
                "symbol": "TEST",
                "local_symbol": "TEST",
                "security_type": "STK",
                "currency": "EUR",
                "exchange": "SMART",
                "primary_exchange": "NASDAQ",
                "min_tick": 0.01,
                "resolved_at": datetime.now(UTC),
                "server_version": 225,
                "contract_hash": "A" * 64,
            }
        ]
    ).to_parquet(contract_path, index=False)
    env = _write_env(tmp_path)
    monkeypatch.setattr(
        automatic,
        "authority_status",
        lambda _root: {
            "execution_authority": "LIVE_LEVEL_ONE",
            "automatic_order_submission": True,
        },
    )
    now = datetime(2026, 7, 27, 15, 0, tzinfo=UTC)
    submitted: list[str] = []

    def quote_provider(_config, _contract):
        return {
            "status": "GO",
            "quote_validation_status": "GO",
            "quote_blockers": [],
            "bid": "4.99",
            "ask": "5.00",
            "fx_rate_to_eur": "1",
            "captured_at": now.isoformat(),
            "market_data_calls": 1,
        }

    def submitter(_root, _env, intent_id):
        submitted.append(intent_id)
        return {
            "status": "GO",
            "live_place_order_calls": 3,
            "blockers": [],
        }

    result = automatic.automatic_cycle(
        tmp_path,
        env_file=env,
        quote_provider=quote_provider,
        submitter=submitter,
        now=now,
        preflight_report={"status": "GO", "blockers": []},
        session_report={
            "status": "GO",
            "session_status": "REGULAR_SESSION_OPEN",
        },
    )

    assert result["status"] == "GO"
    assert result["cycle_status"] == "LIVE_LEVEL_ONE_ORDER_SUBMITTED"
    assert result["live_place_order_calls"] == 3
    assert len(submitted) == 1
    assert result["strategy_id"] == "STRATEGY"
    assert result["symbol"] == "TEST"


def test_manual_level_one_cannot_enter_automatic_submit_route(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        automatic,
        "authority_status",
        lambda _root: {
            "execution_authority": "LIVE_LEVEL_ONE",
            "automatic_order_submission": False,
        },
    )

    result = automatic.automatic_cycle(tmp_path)

    assert result["status"] == "NO_GO"
    assert result["cycle_status"] == "AUTOMATIC_SUBMISSION_NOT_AUTHORIZED"
    assert result["live_place_order_calls"] == 0


def test_daily_profit_target_gate_requires_broker_source_and_throttles() -> None:
    root = Path("C:/not-used")
    now = datetime(2026, 7, 27, 15, 0, tzinfo=UTC)
    original = automatic._read_json
    try:
        automatic._read_json = lambda _path: {
            "status": "GO",
            "session_date": "2026-07-27",
            "input_source": "OPERATOR_SUPPLIED",
            "enforcement_active": True,
            "target_reached": False,
            "new_entries_allowed": True,
            "risk_chasing_allowed": False,
            "risk_reducing_exits_allowed": True,
        }
        blocked = automatic._daily_profit_target_gate(root, now)
        automatic._read_json = lambda _path: {
            "status": "GO",
            "session_date": "2026-07-27",
            "input_source": (
                "BROKER_RECONCILED_EQUITY_AND_CASH_CHANGE"
            ),
            "enforcement_active": True,
            "target_reached": True,
            "new_entries_allowed": False,
            "risk_chasing_allowed": False,
            "risk_reducing_exits_allowed": True,
        }
        reached = automatic._daily_profit_target_gate(root, now)
    finally:
        automatic._read_json = original

    assert blocked["status"] == "NO_GO"
    assert "BROKER_DERIVED_DAILY_PNL_REQUIRED" in blocked["blockers"]
    assert reached == {
        "status": "GO",
        "blockers": [],
        "target_reached": True,
    }


def test_automatic_cycle_offline_audit_and_freeze(tmp_path: Path) -> None:
    source_files = {
        "src/stocks/live/automatic.py": "AUTO = True\n",
        "src/stocks/live/authority.py": "AUTH = True\n",
        "src/stocks/live/quote.py": "QUOTE = True\n",
        "src/stocks/live/service.py": "SERVICE = True\n",
        "tests/test_live_automatic_cycle.py": "TEST = True\n",
    }
    for relative, content in source_files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    audit = automatic.automatic_cycle_audit(tmp_path)
    freeze = automatic.automatic_cycle_freeze(tmp_path)
    status = automatic.automatic_cycle_status(tmp_path)

    assert audit["status"] == "GO"
    assert freeze["freeze_status"] == (
        "LIVE_LEVEL_ONE_AUTOMATIC_CYCLE_FROZEN_GO"
    )
    assert status["hash_integrity"] is True


def _selection_fixture() -> tuple[dict, dict, dict]:
    allowlist = {
        "status": "GO",
        "qualification_hash": "QUALIFICATION",
        "strategies": [
            {
                "strategy_id": "STRATEGY",
                "allowed_symbols": ["TEST"],
                "training_data_end": "2026-07-20T00:00:00+00:00",
            }
        ],
    }
    observation = {
        "observations": [
            {
                "strategy_id": "STRATEGY",
                "independent_forward_session": True,
                "observation_status": "OBSERVATION_COMPLETE",
                "current_attested_target_weights": {"TEST": 1.0},
            }
        ]
    }
    signal = {
        "signal_id": "SIGNAL",
        "strategy_id": "STRATEGY",
        "ticker": "TEST",
        "action": "BUY",
        "data_freshness": "FRESH",
        "data_timestamp": "2026-07-21T00:00:00+00:00",
        "stop_loss": "4",
        "take_profit_1": "7",
        "confidence_score": "0.8",
    }
    return allowlist, observation, signal


def _write_env(root: Path) -> Path:
    path = root / ".env.ibkr.live"
    path.write_text(
        "\n".join(
            [
                "IBKR_ENVIRONMENT=LIVE",
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7496",
                "IBKR_CLIENT_ID=91",
                "IBKR_RECON_CLIENT_ID=92",
                "IBKR_QUOTE_CLIENT_ID=93",
                "IBKR_READ_ONLY=false",
                "IBKR_ORDER_AUTHORITY=CANARY",
                "IBKR_ALLOW_ORDER_TRANSMISSION=true",
                "IBKR_LIVE_TRADING_ENABLED=true",
                "IBKR_LIVE_AUTOSCALE_ENABLED=false",
                "IBKR_MAX_ORDER_EUR=250",
                "IBKR_MAX_TOTAL_EXPOSURE_EUR=250",
                "IBKR_MAX_RISK_EUR=9",
                "IBKR_MAX_OPEN_POSITIONS=1",
                "IBKR_MAX_NEW_ORDERS_PER_DAY=1",
                "IBKR_ALLOW_FRACTIONAL_SHARES=false",
                "IBKR_ALLOW_FUTURES=false",
                "IBKR_ALLOW_SHORTS=false",
                "IBKR_ALLOW_MARGIN=false",
                "IBKR_ALLOW_OPTIONS=false",
                "IBKR_ALLOW_FOREX_SPECULATION=false",
                "IBKR_ACCOUNT_FINGERPRINT_KEY=test-key",
                "IBKR_LIVE_ACCOUNT_FINGERPRINT=TEST-FINGERPRINT",
                "IBKR_MANUAL_APPROVAL_PHRASE=EXACT APPROVAL",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "data" / "portfolio" / "private" / "current-state.json",
        {
            "account_state": {
                "status": "GO",
                "net_liquidation_eur": "1870",
                "eur_available_for_new_longs": "1870",
                "available_funds_eur": "1870",
                "total_cash_value_eur": "1870",
            },
            "whole_share_sizing": {
                "positions": [],
                "current_gross_exposure_pct": "0",
                "current_portfolio_heat": "0",
            },
        },
    )
    _write_json(
        root / "output" / "capital" / "capacity_report.json",
        {
            "instruments": [
                {"symbol": "TEST", "maximum_order_value_eur": "100000"}
            ]
        },
    )
    return path


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
