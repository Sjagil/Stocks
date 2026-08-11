from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

from stocks.capital.service import capital_level_limits
from stocks.live.adapter import build_bracket_orders, build_stock_contract
from stocks.live.models import LiveCanaryConfig
from stocks.live.config import load_live_portfolio_config
from stocks.live.portfolio_targets import evaluate_controlled_purchase_target
from stocks.live.store import LiveExecutionStore
from stocks.live.submission import submit_bracket_once


def _limits(tmp_path: Path) -> dict:
    source = (
        Path(__file__).parents[1]
        / "config/capital_scaling/levels_v1.json"
    )
    target = tmp_path / "config/capital_scaling/levels_v1.json"
    target.parent.mkdir(parents=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return capital_level_limits(
        tmp_path,
        level=2,
        account_equity_eur=Decimal("1870"),
    )


def _target() -> dict:
    return {
        "symbol": "ON",
        "asset_class": "EQUITY",
        "action": "BUY_DELTA",
        "quantity_delta": 2,
        "strategy_source": "validated_strategy",
    }


def _sizing() -> dict:
    return {
        "ticker": "ON",
        "planned_notional_eur": "140.3718115",
        "actual_risk_eur": "9.47686987",
        "remaining_cash_eur": "1729.6281885",
        "reference_price": "81.17",
        "stop_price": "75.69",
        "take_profit_price": "92.13",
        "currency": "USD",
        "fx_to_eur": "0.8646779",
        "execution_candidate_status": "EXECUTABLE_WHOLE_SHARE",
    }


def _contract() -> dict:
    return {
        "con_id": 8677881,
        "symbol": "ON",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "contract_hash": "CONTRACT-HASH",
    }


def test_two_share_target_is_live_ready_when_every_gate_is_proven(
    tmp_path: Path,
) -> None:
    result = evaluate_controlled_purchase_target(
        target=_target(),
        sizing=_sizing(),
        opportunity={"deployment_blockers": []},
        contract=_contract(),
        limits=_limits(tmp_path),
        current_capital_level=2,
        execution_authority="LIVE_LEVEL_TWO",
        p0_gate_go=True,
    )

    assert result["target_technically_executable"] is True
    assert result["live_ready"] is True
    assert result["quantity"] == "2"
    assert result["within_order_notional_cap"] is True
    assert result["within_trade_risk_cap"] is True
    assert result["blockers"] == []
    assert result["submits_orders"] is False


def test_two_share_target_stays_fail_closed_without_runtime_authority(
    tmp_path: Path,
) -> None:
    result = evaluate_controlled_purchase_target(
        target=_target(),
        sizing=_sizing(),
        opportunity={
            "deployment_blockers": [
                "EXECUTION_AUTHORITY_NONE",
                "STRATEGY_DEPLOYMENT_EVIDENCE_REQUIRED",
            ]
        },
        contract=_contract(),
        limits=_limits(tmp_path),
        current_capital_level=0,
        execution_authority="NONE",
        p0_gate_go=False,
    )

    assert result["target_technically_executable"] is True
    assert result["live_ready"] is False
    assert result["blockers"] == [
        "CAPITAL_LEVEL_2_REQUIRED",
        "LIVE_LEVEL_TWO_AUTHORITY_REQUIRED",
        "P0_EXECUTION_INFRASTRUCTURE_READY_REQUIRED",
        "STRATEGY_DEPLOYMENT_EVIDENCE_REQUIRED",
    ]


def test_provider_alias_resolves_to_exact_ibkr_commodity_trust_contract(
    tmp_path: Path,
) -> None:
    target = {
        **_target(),
        "symbol": "U-UN.TO",
        "asset_class": "COMMODITY_EXPOSURE",
    }
    sizing = {
        **_sizing(),
        "ticker": "U-UN.TO",
        "currency": "CAD",
    }
    contract = {
        **_contract(),
        "con_id": 503299503,
        "symbol": "U.UN",
        "broker_symbol": "U.UN",
        "portfolio_symbol": "U-UN.TO",
        "currency": "CAD",
        "exchange": "SMART",
        "primary_exchange": "TSE",
    }

    result = evaluate_controlled_purchase_target(
        target=target,
        sizing=sizing,
        opportunity={"deployment_blockers": []},
        contract=contract,
        limits=_limits(tmp_path),
        current_capital_level=2,
        execution_authority="LIVE_LEVEL_TWO",
        p0_gate_go=True,
    )

    assert result["target_technically_executable"] is True
    assert result["live_ready"] is True
    assert result["contract_resolved"] is True
    assert result["contract"]["symbol"] == "U.UN"
    assert result["contract"]["portfolio_symbol"] == "U-UN.TO"
    assert result["blockers"] == []


def test_target_above_controlled_order_cap_is_rejected(
    tmp_path: Path,
) -> None:
    sizing = {**_sizing(), "planned_notional_eur": "300"}
    result = evaluate_controlled_purchase_target(
        target=_target(),
        sizing=sizing,
        opportunity={"deployment_blockers": []},
        contract=_contract(),
        limits=_limits(tmp_path),
        current_capital_level=2,
        execution_authority="LIVE_LEVEL_TWO",
        p0_gate_go=True,
    )

    assert result["target_technically_executable"] is False
    assert result["live_ready"] is False
    assert "CONTROLLED_ORDER_NOTIONAL_EXCEEDED" in result["blockers"]


def test_controlled_preflight_requires_current_writer_freeze(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from stocks.live import portfolio_targets, service

    config = LiveCanaryConfig(
        host="127.0.0.1",
        port=7496,
        writer_client_id=191,
        recon_client_id=192,
        quote_client_id=193,
        account_fingerprint_key="KEY",
        approved_account_fingerprint="FINGERPRINT",
        manual_activation_phrase="PRIVATE PHRASE",
        writer_enabled=True,
        max_order_eur=Decimal("250"),
        max_total_exposure_eur=Decimal("1122"),
        max_risk_eur=Decimal("28.05"),
        max_open_positions=4,
        max_new_orders_per_day=1,
        approval_ttl_seconds=300,
        callback_timeout_seconds=15,
        fractional_shares_enabled=False,
        execution_authority="LIVE_LEVEL_TWO",
        maximum_quantity=Decimal("100"),
    )
    monkeypatch.setattr(
        portfolio_targets,
        "publish_controlled_purchase_plan",
        lambda _root: {},
    )
    monkeypatch.setattr(
        portfolio_targets,
        "_read_json",
        lambda _path: {
            "targets": [
                {
                    "symbol": "ON",
                    "blockers": [],
                    "target_technically_executable": True,
                }
            ]
        },
    )
    monkeypatch.setattr(
        portfolio_targets,
        "load_live_portfolio_config",
        lambda _root, _env: (config, []),
    )
    monkeypatch.setattr(
        service,
        "live_writer_integrity_command",
        lambda _root, _action: {"status": "NO_GO"},
    )

    report = portfolio_targets.controlled_live_preflight(
        tmp_path, symbol="ON"
    )

    assert report["status"] == "NO_GO"
    assert "CONTROLLED_WRITER_FREEZE_REQUIRED" in report["blockers"]
    assert report["broker_calls"] == 0


def test_level_two_writer_builds_and_submits_two_share_bracket_once_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from stocks.live import service

    config = LiveCanaryConfig(
        host="127.0.0.1",
        port=7496,
        writer_client_id=191,
        recon_client_id=192,
        quote_client_id=193,
        account_fingerprint_key="KEY",
        approved_account_fingerprint="FINGERPRINT",
        manual_activation_phrase="PRIVATE PHRASE",
        writer_enabled=True,
        max_order_eur=Decimal("250"),
        max_total_exposure_eur=Decimal("1122"),
        max_risk_eur=Decimal("28.05"),
        max_open_positions=4,
        max_new_orders_per_day=1,
        approval_ttl_seconds=300,
        callback_timeout_seconds=15,
        fractional_shares_enabled=False,
        execution_authority="LIVE_LEVEL_TWO",
        maximum_quantity=Decimal("100"),
    )
    contract = {
        "con_id": 8677881,
        "symbol": "ON",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "contract_hash": "CONTRACT-HASH",
    }
    monkeypatch.setattr(
        service,
        "_contract_by_con_id",
        lambda _root, _con_id: contract,
    )

    intent, risk = service._build_live_intent(
        tmp_path,
        config,
        con_id=8677881,
        quantity=Decimal("2"),
        entry_limit_price=Decimal("81.17"),
        stop_price=Decimal("75.69"),
        take_profit_price=Decimal("92.13"),
        fx_rate_to_eur=Decimal("0.8646779"),
        reason="controlled desired target",
        strategy_id="VALIDATED-STRATEGY",
    )

    assert intent is not None
    assert risk["status"] == "GO"
    assert risk["quantity_within_profile_limit"] is True
    assert intent.quantity == Decimal("2")
    assert intent.intent_id.startswith("LIVE-PORTFOLIO-")
    orders = build_bracket_orders(intent, parent_order_id=100)
    assert [Decimal(str(order.totalQuantity)) for order in orders] == [
        Decimal("2"),
        Decimal("2"),
        Decimal("2"),
    ]

    store = LiveExecutionStore(tmp_path / "live.sqlite3")
    store.initialize()
    assert store.register_intent(intent.jsonable()) == "INTENT_REGISTERED"
    for order_id in (100, 101, 102):
        assert store.allocate_order_id(order_id, intent.intent_id)[0] == (
            "ORDER_ID_READY"
        )

    class FakeApp:
        def __init__(self) -> None:
            self.calls: list[tuple[int, object, object]] = []

        def placeOrder(self, order_id, broker_contract, order) -> None:  # noqa: N802
            self.calls.append((order_id, broker_contract, order))

    app = FakeApp()
    submitted = submit_bracket_once(
        app,
        order_ids=(100, 101, 102),
        contract=build_stock_contract(intent),
        orders=orders,
        store=store,
        intent_id=intent.intent_id,
    )
    duplicate = submit_bracket_once(
        app,
        order_ids=(100, 101, 102),
        contract=build_stock_contract(intent),
        orders=orders,
        store=store,
        intent_id=intent.intent_id,
    )

    assert submitted["status"] == "GO"
    assert submitted["live_place_order_calls"] == 3
    assert len(app.calls) == 3
    assert duplicate["status"] == "NO_GO"


def test_level_two_writer_accepts_meaningful_multi_share_commodity_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from stocks.live import service

    config = LiveCanaryConfig(
        host="127.0.0.1",
        port=7496,
        writer_client_id=191,
        recon_client_id=192,
        quote_client_id=193,
        account_fingerprint_key="KEY",
        approved_account_fingerprint="FINGERPRINT",
        manual_activation_phrase="PRIVATE PHRASE",
        writer_enabled=True,
        max_order_eur=Decimal("250"),
        max_total_exposure_eur=Decimal("1122"),
        max_risk_eur=Decimal("28.05"),
        max_open_positions=4,
        max_new_orders_per_day=1,
        approval_ttl_seconds=300,
        callback_timeout_seconds=15,
        fractional_shares_enabled=False,
        execution_authority="LIVE_LEVEL_TWO",
        maximum_quantity=Decimal("100"),
    )
    contract = {
        "con_id": 503299503,
        "symbol": "U.UN",
        "security_type": "STK",
        "currency": "CAD",
        "exchange": "SMART",
        "contract_hash": "URANIUM-CONTRACT-HASH",
    }
    monkeypatch.setattr(
        service,
        "_contract_by_con_id",
        lambda _root, _con_id: contract,
    )

    intent, risk = service._build_live_intent(
        tmp_path,
        config,
        con_id=503299503,
        quantity=Decimal("8"),
        entry_limit_price=Decimal("40"),
        stop_price=Decimal("38"),
        take_profit_price=Decimal("45"),
        fx_rate_to_eur=Decimal("0.62"),
        reason="controlled uranium trust target",
        strategy_id="URANIUM-STRATEGY",
    )

    assert intent is not None
    assert risk["status"] == "GO"
    assert intent.symbol == "U.UN"
    assert intent.currency == "CAD"
    assert intent.quantity == Decimal("8")
    assert intent.estimated_notional_eur == Decimal("198.40")
    assert intent.estimated_notional_eur > Decimal("1870") * Decimal("0.04")
    assert intent.maximum_planned_loss_eur == Decimal("9.92")
    orders = build_bracket_orders(intent, parent_order_id=500)
    broker_contract = build_stock_contract(intent)
    assert broker_contract.symbol == "U.UN"
    assert broker_contract.currency == "CAD"
    assert [Decimal(str(order.totalQuantity)) for order in orders] == [
        Decimal("8"),
        Decimal("8"),
        Decimal("8"),
    ]


def test_dedicated_level_two_environment_is_bounded_by_capital_policy(
    tmp_path: Path,
) -> None:
    _limits(tmp_path)
    capital = tmp_path / "output/capital/current_level.json"
    capital.parent.mkdir(parents=True)
    capital.write_text(
        json.dumps({"CURRENT_CAPITAL_LEVEL": 2}), encoding="utf-8"
    )
    account = tmp_path / "data/portfolio/private/current-state.json"
    account.parent.mkdir(parents=True)
    account.write_text(
        json.dumps(
            {"account_state": {"net_liquidation_eur": "1870"}}
        ),
        encoding="utf-8",
    )
    env = tmp_path / ".env.ibkr.portfolio.live"
    env.write_text(
        "\n".join(
            (
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7496",
                "IBKR_ENVIRONMENT=LIVE",
                "IBKR_CLIENT_ID=191",
                "IBKR_RECON_CLIENT_ID=192",
                "IBKR_QUOTE_CLIENT_ID=193",
                "IBKR_READ_ONLY=false",
                "IBKR_ORDER_AUTHORITY=PORTFOLIO",
                "IBKR_ALLOW_ORDER_TRANSMISSION=true",
                "IBKR_LIVE_TRADING_ENABLED=true",
                "IBKR_LIVE_AUTOSCALE_ENABLED=false",
                "IBKR_MAX_ORDER_EUR=250",
                "IBKR_MAX_TOTAL_EXPOSURE_EUR=1122",
                "IBKR_MAX_RISK_EUR=28.05",
                "IBKR_MAX_OPEN_POSITIONS=4",
                "IBKR_MAX_NEW_ORDERS_PER_DAY=1",
                "IBKR_ALLOW_FRACTIONAL_SHARES=false",
                "IBKR_ALLOW_FUTURES=false",
                "IBKR_ALLOW_SHORTS=false",
                "IBKR_ALLOW_MARGIN=false",
                "IBKR_ALLOW_OPTIONS=false",
                "IBKR_ALLOW_FOREX_SPECULATION=false",
                "IBKR_ACCOUNT_FINGERPRINT_KEY=KEY",
                "IBKR_LIVE_ACCOUNT_FINGERPRINT=FINGERPRINT",
                "IBKR_MANUAL_APPROVAL_PHRASE=PRIVATE PHRASE",
            )
        ),
        encoding="utf-8",
    )

    config, errors = load_live_portfolio_config(tmp_path, env)

    assert errors == []
    assert config is not None
    assert config.execution_authority == "LIVE_LEVEL_TWO"
    assert config.maximum_quantity == Decimal("100")
    assert config.max_order_eur == Decimal("250")

    env.write_text(
        env.read_text(encoding="utf-8").replace(
            "IBKR_MAX_ORDER_EUR=250", "IBKR_MAX_ORDER_EUR=251"
        ),
        encoding="utf-8",
    )
    _, excessive_errors = load_live_portfolio_config(tmp_path, env)
    assert "LIVE_LEVEL_TWO_CAPS_BLOCKED" in excessive_errors
