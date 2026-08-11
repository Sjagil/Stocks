from __future__ import annotations

from decimal import Decimal
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pandas as pd

from stocks.ibkr.paper_execution import PHASE9_MARKER
from stocks.ibkr.paper_execution.storage import artifact
from stocks.execution.idempotency import stable_hash
from stocks.live.service import (
    _strategy_eligibility,
    live_canary,
    live_component_status,
    live_kill_switch,
    live_preflight,
    live_strategy_allowlist,
)


def _write_strategy_recommendation(
    root: Path,
    *,
    strategy_id: str,
    authority: str,
    canary_eligible: bool,
    paper_gate: bool,
) -> None:
    path = (
        root
        / "output"
        / "research"
        / "evidence_throughput"
        / "strategy-authority-recommendations.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "graduated_strategy_authority_recommendations_v1",
        "status": "GO",
        "candidates": [
            {
                "strategy_id": strategy_id,
                "recommended_strategy_authority": authority,
                "strategy_canary_eligible": canary_eligible,
                "evidence_components": {
                    "paper_session": {
                        "natural_strategy_session_gate_pass": paper_gate,
                    }
                },
            }
        ],
        "strategy_authority_applied": False,
        "automatic_promotion": False,
        "execution_authority": "NONE",
    }
    payload["content_hash"] = stable_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_live_strategy_eligibility_accepts_only_integrity_checked_canary(
    tmp_path: Path,
) -> None:
    strategy_id = "P1114-CANARY"
    _write_strategy_recommendation(
        tmp_path,
        strategy_id=strategy_id,
        authority="PROVISIONAL",
        canary_eligible=False,
        paper_gate=False,
    )
    provisional = _strategy_eligibility(tmp_path, strategy_id)
    assert provisional["eligible"] is False
    assert provisional["recommended_strategy_authority"] == "PROVISIONAL"

    _write_strategy_recommendation(
        tmp_path,
        strategy_id=strategy_id,
        authority="CANARY",
        canary_eligible=True,
        paper_gate=True,
    )
    canary = _strategy_eligibility(tmp_path, strategy_id)
    assert canary["eligible"] is True
    assert canary["status"] == "GRADUATED_EVIDENCE_ELIGIBLE"
    assert canary["source"] == "GRADUATED_EVIDENCE_RECOMMENDATION"

    path = (
        tmp_path
        / "output"
        / "research"
        / "evidence_throughput"
        / "strategy-authority-recommendations.json"
    )
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["candidates"][0]["strategy_canary_eligible"] = False
    path.write_text(json.dumps(tampered), encoding="utf-8")
    blocked = _strategy_eligibility(tmp_path, strategy_id)
    assert blocked["eligible"] is False
    assert blocked["status"] == "EVIDENCE_RECOMMENDATION_INTEGRITY_BLOCKED"


def write_live_env(root: Path) -> Path:
    path = root / ".env.ibkr.live"
    path.write_text(
        "\n".join(
            [
                "IBKR_ENVIRONMENT=LIVE",
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7496",
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
                "IBKR_ALLOW_FUTURES=false",
                "IBKR_ALLOW_SHORTS=false",
                "IBKR_ALLOW_MARGIN=false",
                "IBKR_ALLOW_OPTIONS=false",
                "IBKR_ALLOW_FOREX_SPECULATION=false",
                "IBKR_MANUAL_APPROVAL_PHRASE=EXACT TEST PHRASE",
            ]
        ),
        encoding="utf-8",
    )
    policy = root / "config" / "capital_scaling" / "levels_v1.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(
        (
            Path(__file__).parents[1]
            / "config"
            / "capital_scaling"
            / "levels_v1.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return path


def test_live_preflight_sends_no_order_and_blocks_missing_evidence(
    tmp_path: Path,
) -> None:
    env = write_live_env(tmp_path)
    report = live_preflight(
        tmp_path,
        env_file=env,
        approval="EXACT TEST PHRASE",
        probe_socket=False,
    )
    assert report["status"] == "NO_GO"
    assert "PHASE9_FILL_CLOSE_CANARY_REQUIRED" in report["blockers"]
    assert "LIVE_EXECUTION_WRITER_NOT_FROZEN" in report["blockers"]
    assert report["live_place_order_calls"] == 0


def test_live_preflight_does_not_claim_contract_resolution_without_symbol(
    tmp_path: Path,
) -> None:
    env = write_live_env(tmp_path)

    report = live_preflight(
        tmp_path,
        env_file=env,
        approval="EXACT TEST PHRASE",
        probe_socket=False,
    )

    assert report["checks"]["contract_resolution_evaluated"] is False
    assert report["checks"]["contract_resolution_status"] == "NOT_EVALUATED_NO_SYMBOL"
    assert report["checks"]["contract_resolved"] is False
    assert report["contract"] == {}
    assert "EXACT_RESOLVED_CONTRACT_REQUIRED" not in report["blockers"]


def test_live_preflight_requires_unique_fresh_contract_identity(
    tmp_path: Path,
) -> None:
    env = write_live_env(tmp_path)
    now = datetime.now(UTC)
    _write_preflight_contract(tmp_path, resolved_at=now - timedelta(days=8))

    stale = live_preflight(
        tmp_path,
        env_file=env,
        symbol="AAPL",
        approval="EXACT TEST PHRASE",
        probe_socket=False,
    )

    assert stale["checks"]["contract_resolution_evaluated"] is True
    assert stale["checks"]["contract_resolution_status"] == "CONTRACT_CACHE_STALE_BLOCKED"
    assert stale["checks"]["contract_resolved"] is False
    assert stale["contract"] == {}
    assert "EXACT_RESOLVED_CONTRACT_REQUIRED" in stale["blockers"]

    _write_preflight_contract(tmp_path, resolved_at=now - timedelta(hours=1))
    fresh = live_preflight(
        tmp_path,
        env_file=env,
        symbol="AAPL",
        approval="EXACT TEST PHRASE",
        probe_socket=False,
    )

    assert fresh["checks"]["contract_resolution_status"] == "FRESH_RESOLVED"
    assert fresh["checks"]["contract_resolved"] is True
    assert fresh["contract"]["con_id"] == 265598
    assert fresh["contract"]["cache_status"] == "FRESH"
    assert fresh["contract"]["contract_source"] == "PHASE2_EXACT_STK_CACHE"
    assert "EXACT_RESOLVED_CONTRACT_REQUIRED" not in fresh["blockers"]


def test_live_preflight_recognizes_exact_phase9_completion_marker(
    tmp_path: Path,
) -> None:
    env = write_live_env(tmp_path)
    phase9 = tmp_path / "output" / "ibkr" / "phase9" / "status.json"
    phase9.parent.mkdir(parents=True)
    phase9.write_text(
        json.dumps(
            artifact(
                "phase9_status_v1",
                {
                    "status": PHASE9_MARKER,
                    "phase9_marker": PHASE9_MARKER,
                    "checks": {
                        "submit_cancel_canary": True,
                        "fill_canary": True,
                        "closing_sell_canary": True,
                    },
                },
            )
        ),
        encoding="utf-8",
    )
    freeze = tmp_path / "output" / "ibkr" / "phase9" / "freeze-status.json"
    freeze.write_text(
        json.dumps(
            artifact(
                "phase9_freeze_status_v1",
                {
                    "freeze_status": (
                        "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_FROZEN_GO"
                    ),
                    "phase9_status": PHASE9_MARKER,
                },
            )
        ),
        encoding="utf-8",
    )

    report = live_preflight(
        tmp_path,
        env_file=env,
        approval="EXACT TEST PHRASE",
        probe_socket=False,
    )

    assert report["checks"]["paper_fill_close_proven"] is True
    assert report["checks"]["phase9_adapter_frozen"] is True
    assert report["checks"]["complete_paper_session_proven"] is True
    assert report["checks"]["paper_session_evidence_source"] == (
        "FROZEN_PHASE9_SUBMIT_FILL_CLOSE_LIFECYCLE"
    )
    assert "PHASE9_FILL_CLOSE_CANARY_REQUIRED" not in report["blockers"]
    assert "ONE_COMPLETE_PAPER_SESSION_REQUIRED" not in report["blockers"]
    assert report["live_place_order_calls"] == 0


def test_live_preflight_phase9_status_without_freeze_is_not_session_proof(
    tmp_path: Path,
) -> None:
    env = write_live_env(tmp_path)
    phase9 = tmp_path / "output" / "ibkr" / "phase9" / "status.json"
    phase9.parent.mkdir(parents=True)
    phase9.write_text(
        json.dumps(
            artifact(
                "phase9_status_v1",
                {
                    "status": PHASE9_MARKER,
                    "phase9_marker": PHASE9_MARKER,
                    "checks": {
                        "submit_cancel_canary": True,
                        "fill_canary": True,
                        "closing_sell_canary": True,
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    report = live_preflight(
        tmp_path,
        env_file=env,
        approval="EXACT TEST PHRASE",
        probe_socket=False,
    )

    assert report["checks"]["paper_fill_close_proven"] is True
    assert report["checks"]["phase9_adapter_frozen"] is False
    assert report["checks"]["complete_paper_session_proven"] is False
    assert "ONE_COMPLETE_PAPER_SESSION_REQUIRED" in report["blockers"]
    assert report["live_place_order_calls"] == 0


def test_live_preflight_rejects_tampered_phase9_marker(
    tmp_path: Path,
) -> None:
    env = write_live_env(tmp_path)
    phase9 = tmp_path / "output" / "ibkr" / "phase9" / "status.json"
    phase9.parent.mkdir(parents=True)
    payload = artifact(
        "phase9_status_v1",
        {
            "status": PHASE9_MARKER,
            "phase9_marker": PHASE9_MARKER,
            "checks": {
                "submit_cancel_canary": True,
                "fill_canary": True,
                "closing_sell_canary": True,
            },
        },
    )
    payload["checks"]["closing_sell_canary"] = False
    phase9.write_text(json.dumps(payload), encoding="utf-8")

    report = live_preflight(
        tmp_path,
        env_file=env,
        approval="EXACT TEST PHRASE",
        probe_socket=False,
    )

    assert report["checks"]["phase9_status_integrity"] is False
    assert "PHASE9_STATUS_INTEGRITY_BLOCKED" in report["blockers"]
    assert "PHASE9_FILL_CLOSE_CANARY_REQUIRED" in report["blockers"]
    assert report["live_place_order_calls"] == 0


def test_live_canary_never_bypasses_preflight(tmp_path: Path) -> None:
    env = write_live_env(tmp_path)
    result = live_canary(
        tmp_path,
        env_file=env,
        strategy_id="UNKNOWN",
        symbol="TEST",
        max_order_eur=Decimal("10"),
        approval="EXACT TEST PHRASE",
    )
    assert result["status"] == "NO_GO"
    assert result["order_sent"] is False
    assert result["live_place_order_calls"] == 0


def test_live_caps_and_forbidden_products_fail_closed(tmp_path: Path) -> None:
    env = write_live_env(tmp_path)
    text = env.read_text(encoding="utf-8")
    env.write_text(
        text.replace("IBKR_MAX_ORDER_EUR=250", "IBKR_MAX_ORDER_EUR=251")
        .replace("IBKR_ALLOW_FUTURES=false", "IBKR_ALLOW_FUTURES=true"),
        encoding="utf-8",
    )
    report = live_preflight(
        tmp_path,
        env_file=env,
        approval="EXACT TEST PHRASE",
        probe_socket=False,
    )
    assert "LIVE_LEVEL_ONE_CAPS_BLOCKED" in report["blockers"]
    assert "FORBIDDEN_PRODUCT_OR_LEVERAGE_FLAG" in report["blockers"]


def test_kill_switch_is_persistent(tmp_path: Path) -> None:
    activated = live_kill_switch(
        tmp_path, command="activate", reason="operator test"
    )
    status = live_kill_switch(tmp_path, command="status")
    assert activated["active"] is True
    assert status["active"] is True
    assert status["execution_authority"] == "NONE"


def test_unobserved_live_broker_counts_are_never_reported_as_zero(
    tmp_path: Path,
) -> None:
    positions = live_component_status(tmp_path, "positions")
    orders = live_component_status(tmp_path, "orders")

    assert positions["position_count"] == "UNAVAILABLE"
    assert positions["unknown_position_count"] == "UNAVAILABLE"
    assert orders["open_order_count"] == "UNAVAILABLE"
    assert orders["unknown_order_count"] == "UNAVAILABLE"


def test_live_allowlist_requires_frozen_forward_and_current_attestation(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    research = tmp_path / "output" / "research" / "phase11_13"
    research.mkdir(parents=True)
    (research / "schema.json").write_text(
        json.dumps(
            {
                "strategies": {
                    "WEEKLY_CROSS_SECTIONAL_MOMENTUM": {
                        "strategy": "risk_adjusted_momentum",
                        "symbols": ["AAPL"],
                        "timeframe": "1w",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (research / "qualification-boundary.json").write_text(
        json.dumps(
            {
                "qualification_hash": "QUALIFICATION-HASH",
                "qualified_at": now.isoformat(),
                "data_end_by_strategy": {
                    "WEEKLY_CROSS_SECTIONAL_MOMENTUM": "2026-07-24"
                },
            }
        ),
        encoding="utf-8",
    )
    (research / "qualification.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "WEEKLY_CROSS_SECTIONAL_MOMENTUM",
                        "robust_pass": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    observation = {
        "observations": [
            {
                "strategy_id": "WEEKLY_CROSS_SECTIONAL_MOMENTUM",
                "independent_forward_session": True,
                "observation_status": "OBSERVATION_COMPLETE",
                "current_attested_target_weights": {"AAPL": 0.25},
            }
        ]
    }
    (research / "latest-forward-observation.json").write_text(
        json.dumps(observation), encoding="utf-8"
    )
    attestations = (
        tmp_path / "config" / "screener" / "shariah_attestations_v1.json"
    )
    attestations.parent.mkdir(parents=True)
    attestations.write_text(
        json.dumps(
            {
                "attestations": [
                    {
                        "symbol": "AAPL",
                        "status": "SHARIAH_ELIGIBLE_PIT",
                        "screened_at": (now - timedelta(days=1)).isoformat(),
                        "expires_at": (now + timedelta(days=1)).isoformat(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "src" / "stocks" / "research" / "phase11_13.py"
    source.parent.mkdir(parents=True)
    source.write_text("FROZEN = True\n", encoding="utf-8")

    report = live_strategy_allowlist(tmp_path)

    assert report["status"] == "GO"
    assert report["strategy_count"] == 1
    strategy = report["strategies"][0]
    assert strategy["allowed_symbols"] == ["AAPL"]
    assert strategy["qualification_hash"] == "QUALIFICATION-HASH"
    assert strategy["training_data_end"] == "2026-07-24"
    assert strategy["canary_notional_hard_cap_eur"] == "250"
    assert strategy["primary_sizing_authority"] == "RISK_PER_WHOLE_SHARE"

    observation["observations"][0]["independent_forward_session"] = False
    (research / "latest-forward-observation.json").write_text(
        json.dumps(observation), encoding="utf-8"
    )
    blocked = live_strategy_allowlist(tmp_path)
    assert blocked["status"] == "NO_GO"
    assert blocked["strategy_count"] == 0
    assert "INDEPENDENT_FORWARD_SESSION_REQUIRED" in (
        blocked["candidates"][0]["blockers"]
    )


def _write_preflight_contract(root: Path, *, resolved_at: datetime) -> None:
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
                "resolved_at": resolved_at,
                "server_version": 225,
                "contract_hash": "A" * 64,
            }
        ]
    ).to_parquet(path, index=False)


def test_live_allowlist_accepts_frozen_mtf_pit_observation(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    research = tmp_path / "output" / "research" / "phase11_10"
    research.mkdir(parents=True)
    (research / "latest-pit-forward-observation.json").write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "strategy_id": "MTF-FROZEN",
                        "version": "PHASE11_10_MTF_V1",
                        "source_hash": "SOURCE",
                        "parameter_hash": "PARAMETERS",
                        "qualification_hash": "QUALIFICATION",
                        "qualified_at": now.isoformat(),
                        "training_data_end": "2026-07-24T17:30:00",
                        "allowed_timeframes": ["4h", "1d"],
                        "independent_forward_session": True,
                        "current_attested_target_weights": {
                            "AAPL": 0.25
                        },
                        "provider_continuity_status": (
                            "SAME_PRIMARY_PROVIDER_GO"
                        ),
                        "qualification_status": (
                            "ROBUST_SHORTLIST_FROZEN"
                        ),
                        "observation_status": "PIT_OBSERVATION_COMPLETE",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    attestations = (
        tmp_path / "config" / "screener" / "shariah_attestations_v1.json"
    )
    attestations.parent.mkdir(parents=True)
    attestations.write_text(
        json.dumps(
            {
                "attestations": [
                    {
                        "symbol": "AAPL",
                        "status": "SHARIAH_ELIGIBLE_PIT",
                        "screened_at": (
                            now - timedelta(days=1)
                        ).isoformat(),
                        "expires_at": (
                            now + timedelta(days=1)
                        ).isoformat(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = live_strategy_allowlist(tmp_path)

    assert report["status"] == "GO"
    assert report["strategy_count"] == 1
    strategy = report["strategies"][0]
    assert strategy["strategy_id"] == "MTF-FROZEN"
    assert strategy["allowed_symbols"] == ["AAPL"]
    assert strategy["allowed_timeframes"] == ["4h", "1d"]
    assert report["execution_authority"] == "NONE"


def test_live_allowlist_accepts_only_frozen_attested_phase11_14_stock(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    research = tmp_path / "output" / "research" / "phase11_14"
    research.mkdir(parents=True)
    strategy_id = "P1114-STOCK"
    boundary = {
        "status": "FROZEN",
        "qualification_hash": "SURVIVOR-QUALIFICATION",
        "frozen_at": now.isoformat(),
        "data_end_by_strategy": {
            strategy_id: "2026-07-28T17:30:00"
        },
        "robust_strategy_ids": [strategy_id],
    }
    (research / "qualification-boundary.json").write_text(
        json.dumps(boundary),
        encoding="utf-8",
    )
    (research / "qualification.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": strategy_id,
                        "source_strategy_id": "BULK-STOCK",
                        "formula": "choppiness_breakout",
                        "frozen_profile": "balanced",
                        "timeframe": "4h",
                        "asset_class": "STOCK",
                        "evidence_scope": (
                            "SELECTION_CONDITIONED_REUSED_HISTORY"
                        ),
                        "robust_pass": True,
                        "portfolio_invariants_go": True,
                        "forward_observer_candidate": True,
                    },
                    {
                        "strategy_id": "P1114-COMMODITY",
                        "formula": "trend_quality_52w",
                        "frozen_profile": "balanced",
                        "timeframe": "4h",
                        "asset_class": "COMMODITY_PROXY",
                        "robust_pass": True,
                        "portfolio_invariants_go": True,
                        "forward_observer_candidate": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    observation = {
        "status": "GO",
        "qualification_hash": "SURVIVOR-QUALIFICATION",
        "observations": [
            {
                "strategy_id": strategy_id,
                "timeframe": "4h",
                "data_freshness": "FRESH_CLOSED_BAR",
                "independent_forward_session": True,
                "current_attested_target_weights": {"AAPL": 0.25},
            },
            {
                "strategy_id": "P1114-COMMODITY",
                "timeframe": "4h",
                "data_freshness": "FRESH_CLOSED_BAR",
                "independent_forward_session": True,
                "current_attested_target_weights": {"AAPL": 0.25},
            },
        ],
    }
    (research / "latest-forward-observation.json").write_text(
        json.dumps(observation),
        encoding="utf-8",
    )
    attestations = (
        tmp_path
        / "config"
        / "screener"
        / "shariah_attestations_v1.json"
    )
    attestations.parent.mkdir(parents=True)
    attestations.write_text(
        json.dumps(
            {
                "attestations": [
                    {
                        "symbol": "AAPL",
                        "status": "SHARIAH_ELIGIBLE_PIT",
                        "screened_at": (
                            now - timedelta(days=1)
                        ).isoformat(),
                        "expires_at": (
                            now + timedelta(days=1)
                        ).isoformat(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source = (
        tmp_path / "src" / "stocks" / "research" / "phase11_14.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("FROZEN = True\n", encoding="utf-8")

    report = live_strategy_allowlist(tmp_path)

    assert report["status"] == "GO"
    assert report["strategy_count"] == 1
    strategy = report["strategies"][0]
    assert strategy["strategy_id"] == strategy_id
    assert strategy["allowed_symbols"] == ["AAPL"]
    assert strategy["allowed_timeframes"] == ["4h"]
    assert strategy["canary_notional_hard_cap_eur"] == "250"
    assert strategy["primary_sizing_authority"] == "RISK_PER_WHOLE_SHARE"
    commodity = next(
        row
        for row in report["candidates"]
        if row["strategy_id"] == "P1114-COMMODITY"
    )
    assert "CONTROLLED_LIVE_STOCK_ONLY" in commodity["blockers"]
    assert "FROZEN_ROBUST_STRATEGY_ID_REQUIRED" in (
        commodity["blockers"]
    )
    assert report["execution_authority"] == "NONE"

    observation["qualification_hash"] = "MISMATCH"
    (research / "latest-forward-observation.json").write_text(
        json.dumps(observation),
        encoding="utf-8",
    )
    blocked = live_strategy_allowlist(tmp_path)
    assert blocked["strategy_count"] == 0
    assert "FROZEN_QUALIFICATION_HASH_REQUIRED" in (
        blocked["candidates"][0]["blockers"]
    )
