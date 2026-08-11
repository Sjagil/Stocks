from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stocks.live import autonomous_policy
from stocks.live.autonomous_policy import (
    AUTONOMOUS_LEVEL_ONE,
    FINAL_GATES,
    autonomous_final_policy_check,
    autonomous_resilience_audit,
)
from stocks.live.authority import operational_allowlist_hash
from stocks.portfolio.orchestrator import build_authority_executable_portfolio
from stocks.portfolio.strategy_authority import (
    bind_exact_strategy,
    load_strategy_authority_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict:
    return {
        "status": "GO",
        "actual_operational_allowlist_hash": "ALLOW",
        "qualification_hash": "QUAL",
        "strategies": [
            {
                "strategy_id": "S1",
                "version": "V1",
                "source_hash": "SOURCE",
                "parameter_hash": "PARAM",
                "allowed_symbols": ["AAPL"],
                "allowed_asset_classes": ["STK"],
                "deployment_status": "LIVE_AUTHORIZED_LEVEL_ONE",
            }
        ],
    }


def _candidate(**updates: object) -> dict:
    value = {
        "symbol": "AAPL",
        "asset_class": "STK",
        "strategy_ids": ["S1"],
    }
    value.update(updates)
    return value


def _intent() -> SimpleNamespace:
    return SimpleNamespace(
        intent_id="INTENT-1",
        economic_order_key="ECONOMIC-1",
        strategy_id="S1",
        asset_class="STK",
        symbol="AAPL",
        con_id=265598,
        quantity=1,
        entry_limit_price="200",
        stop_price="190",
        take_profit_price="220",
        estimated_notional_eur="180",
        planned_total_risk_eur="9",
        risk_per_share_eur="9",
        desired_qty="1",
        normal_allowed_qty="1",
        canary_qty="1",
        cash_before_eur="1870",
        cash_after_eur="1690",
        portfolio_weight="0.096",
    )


def test_exact_strategy_binding_never_infers_or_promotes() -> None:
    exact = bind_exact_strategy(_candidate(), _registry())
    missing = bind_exact_strategy(
        _candidate(symbol="ON", strategy_ids=["RESEARCH-ONLY"]), _registry()
    )
    ambiguous_registry = _registry()
    ambiguous_registry["strategies"].append(
        {
            **ambiguous_registry["strategies"][0],
            "strategy_id": "S2",
        }
    )
    ambiguous = bind_exact_strategy(
        _candidate(strategy_ids=["S1", "S2"]), ambiguous_registry
    )

    assert exact["status"] == "GO"
    assert exact["strategy_id"] == "S1"
    assert missing["status"] == "NO_GO"
    assert "NO_EXACT_LIVE_AUTHORIZED_CONTRIBUTING_STRATEGY" in missing["blockers"]
    assert ambiguous["status"] == "NO_GO"
    assert "AMBIGUOUS_LIVE_AUTHORIZED_CONTRIBUTING_STRATEGY" in ambiguous["blockers"]
    assert exact["inference_used"] is False
    assert exact["automatic_promotion_used"] is False


def test_current_authority_portfolio_preserves_unauthorized_research_target() -> None:
    integer = {
        "cash_eur": 1800,
        "positions": [{"symbol": "ON", "quantity": 1, "notional_eur": 70}],
    }
    opportunities = [
        {
            "symbol": "ON",
            "strategy_binding": bind_exact_strategy(
                _candidate(symbol="ON", strategy_ids=["RESEARCH-ONLY"]),
                _registry(),
            ),
        }
    ]
    result = build_authority_executable_portfolio(
        integer,
        opportunities,
        _registry(),
        {"execution_authority": AUTONOMOUS_LEVEL_ONE},
    )

    assert result["positions"] == []
    assert result["cash_eur"] == 1870
    assert result["excluded_research_positions"][0]["symbol"] == "ON"
    assert result["excluded_research_positions"][0]["research_target_preserved"] is True


def test_registry_fails_closed_on_allowlist_hash_change(tmp_path: Path) -> None:
    registry_path = tmp_path / "config/portfolio"
    registry_path.mkdir(parents=True)
    configured = json.loads(
        (ROOT / "config/portfolio/strategy_authority_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    allowlist = json.loads(
        (ROOT / "output/ibkr/live/strategy-allowlist.json").read_text(
            encoding="utf-8"
        )
    )
    assert operational_allowlist_hash(allowlist) == configured[
        "operational_allowlist_hash"
    ]
    (registry_path / "strategy_authority_registry_v1.json").write_text(
        json.dumps(configured), encoding="utf-8"
    )
    allowlist["strategies"][0]["maximum_position_weight"] = "0.16"

    status = load_strategy_authority_registry(tmp_path, allowlist=allowlist)

    assert status["status"] == "NO_GO"
    assert "FROZEN_OPERATIONAL_ALLOWLIST_HASH_MISMATCH" in status["blockers"]


def test_autonomous_decision_is_immutable_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stocks.live.authority as authority_module
    import stocks.live.level_one_reauthorization as p02_module
    import stocks.live.service as service_module

    monkeypatch.setattr(
        authority_module,
        "authority_status",
        lambda _root: {
            "execution_authority": AUTONOMOUS_LEVEL_ONE,
            "p0_safety_gate_go": True,
            "allowlist_hash_matches": True,
            "qualification_hash_matches": True,
            "kill_switch_active": False,
        },
    )
    monkeypatch.setattr(p02_module, "verify_p02_freeze", lambda _root: {"status": "GO", "freeze_hash": "P02"})
    monkeypatch.setattr(service_module, "live_writer_integrity_command", lambda *_args: {"status": "GO", "current_manifest_hash": "WRITER"})
    monkeypatch.setattr(autonomous_policy, "verify_p2_1_freeze", lambda _root: {"status": "GO", "freeze_hash": "P21"})
    monkeypatch.setattr(autonomous_policy, "load_strategy_authority_registry", lambda _root: _registry())
    gates = {name: True for name in FINAL_GATES}

    first = autonomous_final_policy_check(
        tmp_path,
        _intent(),
        policy_gates=gates,
        candidate=_candidate(),
    )
    second = autonomous_final_policy_check(
        tmp_path,
        _intent(),
        policy_gates=gates,
        candidate=_candidate(),
    )

    assert first["approved"] is True
    assert first["state_transitions"] == [
        "PREPARED",
        "AUTONOMOUS_FINAL_POLICY_CHECK",
        "AUTONOMOUS_APPROVED",
        "SUBMIT",
    ]
    assert first["broker_write_calls_during_policy_check"] == 0
    assert first["per_trade_human_approval_required"] is False
    assert second["idempotent_replay"] is True
    assert second["record_hash"] == first["record_hash"]
    assert len(list((tmp_path / autonomous_policy.DECISION_ROOT).glob("*.json"))) == 1


def test_final_policy_rejects_single_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stocks.live.authority as authority_module
    import stocks.live.level_one_reauthorization as p02_module
    import stocks.live.service as service_module

    monkeypatch.setattr(authority_module, "authority_status", lambda _root: {"execution_authority": AUTONOMOUS_LEVEL_ONE, "p0_safety_gate_go": True, "allowlist_hash_matches": True, "qualification_hash_matches": True, "kill_switch_active": False})
    monkeypatch.setattr(p02_module, "verify_p02_freeze", lambda _root: {"status": "GO"})
    monkeypatch.setattr(service_module, "live_writer_integrity_command", lambda *_args: {"status": "GO"})
    monkeypatch.setattr(autonomous_policy, "verify_p2_1_freeze", lambda _root: {"status": "GO"})
    monkeypatch.setattr(autonomous_policy, "load_strategy_authority_registry", lambda _root: _registry())
    gates = {name: True for name in FINAL_GATES}
    gates["MARKET_DATA_FRESH"] = False

    decision = autonomous_final_policy_check(
        tmp_path,
        _intent(),
        policy_gates=gates,
        candidate=_candidate(),
        write_record=False,
    )

    assert decision["approved"] is False
    assert decision["final_state"] == "AUTONOMOUS_REJECTED"
    assert decision["blockers"] == ["MARKET_DATA_FRESH"]


def test_restart_partial_fill_disconnect_kill_switch_and_asset_mocks() -> None:
    audit = autonomous_resilience_audit()

    assert audit["status"] == "GO"
    assert set(audit["restart_scenarios"]) >= {
        "PREPARED",
        "SUBMITTING",
        "PARTIAL_FILL",
        "OPEN_POSITION",
        "DISCONNECTED",
        "KILL_SWITCH",
    }
    assert set(audit["asset_class_mocks"]) == {
        "STOCK",
        "ETF",
        "COMMODITY",
        "NO_TRADE",
    }
    assert audit["fill_consistent_quantities"] is True
    assert audit["economic_intent_exactly_once"] is True
    assert audit["broker_write_calls"] == 0
