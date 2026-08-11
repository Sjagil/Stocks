from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from stocks.live.adapter import build_bracket_orders, build_stock_contract
from stocks.live.approvals import approval_challenge
from stocks.live.config import load_live_canary_config
from stocks.live.service import (
    _FakeLiveApp,
    _live_intent_binding,
    _load_live_intent,
    _offline_live_intent,
    live_approve,
    live_audit,
    live_prepare,
)
from stocks.live.store import LiveExecutionStore
from stocks.live.submission import submit_bracket_once


def test_live_config_is_strict_and_accepts_only_level_one_contract(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    config, errors = load_live_canary_config(tmp_path, env)

    assert errors == []
    assert config is not None
    assert config.max_order_eur == Decimal("250")
    assert config.canary_risk_fraction == Decimal("0.005")


def test_repository_live_example_is_a_valid_level_one_template(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[1] / ".env.ibkr.live.example"
    contents = source.read_text(encoding="utf-8")
    contents = contents.replace(
        "IBKR_ACCOUNT_FINGERPRINT_KEY=",
        "IBKR_ACCOUNT_FINGERPRINT_KEY=TEST-KEY",
    ).replace(
        "IBKR_LIVE_ACCOUNT_FINGERPRINT=",
        "IBKR_LIVE_ACCOUNT_FINGERPRINT=TEST-FINGERPRINT",
    ).replace(
        "IBKR_MANUAL_APPROVAL_PHRASE=",
        "IBKR_MANUAL_APPROVAL_PHRASE=TEST-PRIVATE-PHRASE",
    )
    env = tmp_path / ".env.ibkr.live"
    env.write_text(contents, encoding="utf-8")

    config, errors = load_live_canary_config(tmp_path, env)

    assert errors == []
    assert config is not None
    assert config.writer_enabled is True
    assert config.fractional_shares_enabled is False
    assert config.max_total_exposure_eur == Decimal("250")
    assert config.max_risk_eur == Decimal("9")
    assert config.writer_client_id != config.recon_client_id
    assert config.writer_enabled is True
    assert config.fractional_shares_enabled is False

    unsafe = env.read_text(encoding="utf-8").replace(
        "IBKR_MAX_ORDER_EUR=250", "IBKR_MAX_ORDER_EUR=251"
    )
    env.write_text(unsafe, encoding="utf-8")
    _, errors = load_live_canary_config(tmp_path, env)
    assert "LIVE_LEVEL_ONE_CAPS_BLOCKED" in errors

    unsafe = unsafe.replace(
        "IBKR_MAX_ORDER_EUR=251", "IBKR_MAX_ORDER_EUR=250"
    ).replace(
        "IBKR_ALLOW_FRACTIONAL_SHARES=false",
        "IBKR_ALLOW_FRACTIONAL_SHARES=true",
    )
    env.write_text(unsafe, encoding="utf-8")
    _, errors = load_live_canary_config(tmp_path, env)
    assert "FRACTIONAL_SHARES_MUST_BE_DISABLED" in errors


def test_prepare_blocks_fractional_and_requires_exact_approval(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    _write_contract_cache(tmp_path)

    fractional = live_prepare(
        tmp_path,
        env_file=env,
        con_id=265598,
        quantity=Decimal("0.05"),
        entry_limit_price=Decimal("100"),
        stop_price=Decimal("95"),
        take_profit_price=Decimal("110"),
        fx_rate_to_eur=Decimal("0.90"),
        reason="fractional quantity must be blocked",
    )
    assert fractional["status"] == "NO_GO"
    assert "FRACTIONAL_QUANTITY_FORBIDDEN" in fractional["risk_status"][
        "blockers"
    ]

    prepared = live_prepare(
        tmp_path,
        env_file=env,
        con_id=265598,
        quantity=Decimal("1"),
        entry_limit_price=Decimal("10"),
        stop_price=Decimal("9.5"),
        take_profit_price=Decimal("12.5"),
        fx_rate_to_eur=Decimal("0.90"),
        reason="manual Level-1 live canary",
    )

    assert prepared["status"] == "GO"
    assert Decimal(
        prepared["risk_status"]["estimated_notional_eur"]
    ) == Decimal("9.00")
    assert Decimal(
        prepared["risk_status"]["maximum_planned_loss_eur"]
    ) > Decimal("0.450")
    assert prepared["risk_status"]["canary_qty"] == 1
    store = LiveExecutionStore.from_project_root(tmp_path)
    intent = _load_live_intent(store, str(prepared["intent_id"]))
    assert intent is not None
    challenge = approval_challenge(intent)

    rejected = live_approve(
        tmp_path,
        env_file=env,
        intent_id=intent.intent_id,
        approval="WRONG",
    )
    assert rejected["status"] == "NO_GO"
    assert rejected["approval_status"] == "APPROVAL_MISMATCH"

    approved = live_approve(
        tmp_path,
        env_file=env,
        intent_id=intent.intent_id,
        approval=challenge,
    )
    assert approved["status"] == "GO"
    public_prepare = json.loads(
        (
            tmp_path / "output" / "ibkr" / "live" / "prepare.json"
        ).read_text(encoding="utf-8")
    )
    assert "approval_challenge" not in public_prepare
    assert "risk_status" not in public_prepare
    assert public_prepare["financial_values_private"] is True


def test_prepare_blocks_notional_risk_and_invalid_bracket(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    _write_contract_cache(tmp_path)

    blocked = live_prepare(
        tmp_path,
        env_file=env,
        con_id=265598,
        quantity=Decimal("1"),
        entry_limit_price=Decimal("100"),
        stop_price=Decimal("101"),
        take_profit_price=Decimal("90"),
        fx_rate_to_eur=Decimal("1"),
        reason="must be blocked",
    )

    assert blocked["status"] == "NO_GO"
    blockers = blocked["risk_status"]["blockers"]
    assert "LONG_BRACKET_PRICE_ORDER_BLOCKED" in blockers


def test_prepare_blocks_stale_contract_cache_identity(tmp_path: Path) -> None:
    env = _write_live_env(tmp_path)
    _write_contract_cache(
        tmp_path,
        resolved_at=datetime.now(UTC) - timedelta(days=8),
    )

    blocked = live_prepare(
        tmp_path,
        env_file=env,
        con_id=265598,
        quantity=Decimal("1"),
        entry_limit_price=Decimal("10"),
        stop_price=Decimal("9.5"),
        take_profit_price=Decimal("12.5"),
        fx_rate_to_eur=Decimal("0.90"),
        reason="stale contract must block",
    )

    assert blocked["status"] == "NO_GO"
    assert "EXACT_RESOLVED_STK_CONTRACT_REQUIRED" in blocked["risk_status"]["blockers"]


def test_atomic_bracket_is_idempotent_and_never_marketable(
    tmp_path: Path,
) -> None:
    report = live_audit(tmp_path)
    assert report["status"] == "GO"
    assert report["atomic_bracket_transmission"] is True
    assert report["real_broker_app_used"] is False
    assert report["live_place_order_calls_in_offline_audit"] == 0
    assert report["whole_share_enforced"] is True
    assert report["fractional_quantity_supported"] is False


def test_live_intent_is_rebound_to_strategy_symbol_and_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_contract_cache(tmp_path)
    allowlist = {
        "status": "GO",
        "strategies": [
            {
                "strategy_id": "STRATEGY-1",
                "allowed_symbols": ["AAPL"],
            }
        ],
    }
    monkeypatch.setattr(
        "stocks.live.service.live_strategy_allowlist",
        lambda _root: allowlist,
    )
    intent = replace(
        _offline_live_intent(),
        strategy_id="STRATEGY-1",
        con_id=265598,
        symbol="AAPL",
        currency="USD",
        exchange="SMART",
        contract_hash="A" * 64,
    )

    valid = _live_intent_binding(tmp_path, intent)
    wrong_strategy = _live_intent_binding(
        tmp_path, replace(intent, strategy_id="STRATEGY-2")
    )
    wrong_symbol = _live_intent_binding(
        tmp_path, replace(intent, symbol="NVDA")
    )
    wrong_contract = _live_intent_binding(
        tmp_path, replace(intent, contract_hash="B" * 64)
    )
    missing_strategy = _live_intent_binding(
        tmp_path, replace(intent, strategy_id=None)
    )

    assert valid["status"] == "GO"
    assert valid["contract_bound"] is True
    assert "STRATEGY_NOT_PIT_LIVE_ALLOWLISTED" in wrong_strategy["blockers"]
    assert "SYMBOL_NOT_ALLOWED_FOR_LIVE_STRATEGY" in wrong_symbol["blockers"]
    assert "LIVE_INTENT_CONTRACT_SYMBOL_MISMATCH" in wrong_symbol["blockers"]
    assert "LIVE_INTENT_CONTRACT_HASH_MISMATCH" in wrong_contract["blockers"]
    assert "LIVE_STRATEGY_ID_REQUIRED" in missing_strategy["blockers"]


def test_live_intent_accepts_provider_symbol_bound_to_exact_broker_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    allowlist = {
        "status": "GO",
        "strategies": [
            {
                "strategy_id": "URANIUM-STRATEGY",
                "allowed_symbols": ["U-UN.TO"],
            }
        ],
    }
    contract = {
        "con_id": 503299503,
        "symbol": "U.UN",
        "security_type": "STK",
        "currency": "CAD",
        "exchange": "SMART",
        "contract_hash": "U" * 64,
    }
    monkeypatch.setattr(
        "stocks.live.service.live_strategy_allowlist",
        lambda _root: allowlist,
    )
    monkeypatch.setattr(
        "stocks.live.service.broad_asset_metadata",
        lambda _root: {
            "U-UN.TO": {
                "broker_symbol": "U.UN",
                "currency": "CAD",
                "primary_exchange": "TSE",
            }
        },
    )
    monkeypatch.setattr(
        "stocks.live.service._contract_by_con_id",
        lambda _root, _con_id: contract,
    )
    intent = replace(
        _offline_live_intent(),
        strategy_id="URANIUM-STRATEGY",
        con_id=503299503,
        symbol="U.UN",
        currency="CAD",
        contract_hash="U" * 64,
    )

    result = _live_intent_binding(tmp_path, intent)

    assert result["status"] == "GO"
    assert result["contract_bound"] is True
    assert result["strategy_identity_symbols"] == ["U-UN.TO", "U.UN"]


def test_submit_bracket_once_allows_exactly_three_calls(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    _write_contract_cache(tmp_path)
    prepared = live_prepare(
        tmp_path,
        env_file=env,
        con_id=265598,
        quantity=Decimal("1"),
        entry_limit_price=Decimal("10"),
        stop_price=Decimal("9.5"),
        take_profit_price=Decimal("12.5"),
        fx_rate_to_eur=Decimal("0.90"),
        reason="idempotence audit",
    )
    store = LiveExecutionStore.from_project_root(tmp_path)
    intent = _load_live_intent(store, str(prepared["intent_id"]))
    assert intent is not None
    for order_id in (100, 101, 102):
        status, allocated = store.allocate_order_id(order_id, intent.intent_id)
        assert status == "ORDER_ID_READY"
        assert allocated == order_id
    app = _FakeLiveApp()
    orders = build_bracket_orders(intent, parent_order_id=100)

    first = submit_bracket_once(
        app,
        order_ids=(100, 101, 102),
        contract=build_stock_contract(intent),
        orders=orders,
        store=store,
        intent_id=intent.intent_id,
    )
    second = submit_bracket_once(
        app,
        order_ids=(100, 101, 102),
        contract=build_stock_contract(intent),
        orders=orders,
        store=store,
        intent_id=intent.intent_id,
    )

    assert first["status"] == "GO"
    assert first["live_place_order_calls"] == 3
    assert second["status"] == "NO_GO"
    assert second["live_place_order_calls"] == 0
    assert len(app.calls) == 3
    assert [bool(order.transmit) for order in orders] == [
        False,
        False,
        True,
    ]


def test_submission_boundary_rejects_fractional_parent_before_broker_call(
    tmp_path: Path,
) -> None:
    store = LiveExecutionStore.from_project_root(tmp_path)
    store.initialize()
    intent = _offline_live_intent()
    orders = build_bracket_orders(intent, parent_order_id=100)
    orders[0].totalQuantity = 0.5
    app = _FakeLiveApp()

    result = submit_bracket_once(
        app,
        order_ids=(100, 101, 102),
        contract=build_stock_contract(intent),
        orders=orders,
        store=store,
        intent_id=intent.intent_id,
    )

    assert result["status"] == "NO_GO"
    assert result["submission_status"] == "FRACTIONAL_QUANTITY_FORBIDDEN"
    assert result["live_place_order_calls"] == 0
    assert app.calls == []


def test_live_daily_submission_claim_is_single_flight(
    tmp_path: Path,
) -> None:
    store = LiveExecutionStore.from_project_root(tmp_path)
    store.initialize()

    first = store.claim_daily_submission(
        session_date="2026-07-27",
        intent_id="LIVE-ONE",
    )
    second = store.claim_daily_submission(
        session_date="2026-07-27",
        intent_id="LIVE-TWO",
    )

    assert first == "LIVE_DAILY_SUBMISSION_CLAIMED"
    assert second == "LIVE_DAILY_SUBMISSION_ALREADY_CLAIMED"


def _write_live_env(root: Path) -> Path:
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
                "IBKR_LIVE_APPROVAL_TTL_SECONDS=300",
                "IBKR_LIVE_CALLBACK_TIMEOUT_SECONDS=1",
                "IBKR_ALLOW_FRACTIONAL_SHARES=false",
                "IBKR_ALLOW_FUTURES=false",
                "IBKR_ALLOW_SHORTS=false",
                "IBKR_ALLOW_MARGIN=false",
                "IBKR_ALLOW_OPTIONS=false",
                "IBKR_ALLOW_FOREX_SPECULATION=false",
                "IBKR_ACCOUNT_FINGERPRINT_KEY=test-key",
                "IBKR_LIVE_ACCOUNT_FINGERPRINT=TEST-FINGERPRINT",
                "IBKR_MANUAL_APPROVAL_PHRASE=ACTIVATE-TEST-LIVE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = root / "data" / "portfolio" / "private" / "current-state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )
    capacity = root / "output" / "capital" / "capacity_report.json"
    capacity.parent.mkdir(parents=True, exist_ok=True)
    capacity.write_text(
        json.dumps(
            {
                "instruments": [
                    {
                        "symbol": "AAPL",
                        "maximum_order_value_eur": "100000",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_contract_cache(
    root: Path,
    *,
    resolved_at: datetime | None = None,
) -> None:
    path = root / "output" / "ibkr" / "contracts" / "stocks.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "con_id": 265598,
                "symbol": "AAPL",
                "local_symbol": "AAPL",
                "security_type": "STK",
                "currency": "USD",
                "exchange": "SMART",
                "primary_exchange": "NASDAQ",
                "min_tick": 0.01,
                "resolved_at": resolved_at or datetime.now(UTC),
                "server_version": 225,
                "contract_hash": "A" * 64,
            }
        ]
    ).to_parquet(path, index=False)
