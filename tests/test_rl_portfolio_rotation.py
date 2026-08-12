from __future__ import annotations

from typing import Any

from stocks.rl.portfolio import (
    FEATURE_SCHEMA_HASH,
    evaluate_shadow_portfolio_rotation,
)


def _config() -> dict[str, Any]:
    return {
        "schema": "rl_portfolio_rotation_contract_v1",
        "mode": "SHADOW_ONLY",
        "top_n": 10,
        "accepted_entry_lifecycle_states": [
            "ENTRY_READY",
            "ROTATION_CANDIDATE",
        ],
        "accepted_shariah_statuses": ["SHARIAH_ELIGIBLE_PIT"],
        "accepted_product_shariah_statuses": [
            "SHARIAH_PRODUCT_ELIGIBLE_PIT"
        ],
        "require_contract_resolved": True,
        "require_positive_expected_net_return": True,
        "require_positive_expected_r": True,
        "require_whole_share_preflight": True,
        "require_causal_candidate_history": True,
        "require_promoted_dedicated_policy": True,
        "require_oos_promotion_gate": True,
        "candidate_history_path": "history.parquet",
        "promotion_evidence_path": "promotion.json",
        "policy_may_create_signals": False,
        "policy_may_change_eligibility": False,
        "policy_may_increase_weight": False,
        "policy_may_override_caps": False,
        "financial_effect_applied": False,
        "automatic_order_submission": False,
        "execution_authority": "NONE",
    }


def _candidate(ticker: str = "SPUS") -> dict[str, Any]:
    return {
        "ticker": ticker,
        "contract_resolved": True,
        "lifecycle_state": "ENTRY_READY",
        "shariah_status": "SHARIAH_ELIGIBLE_PIT",
        "research_allocation_blockers": [],
        "expected_net_return": 0.03,
        "expected_r": 1.5,
        "portfolio_objective_score": 0.75,
        "opportunity_score": 0.8,
        "stop_risk_pct": 0.04,
        "components": {"liquidity": 0.9, "regime_fit": 0.8},
        "real_asset_context": {"status": "NOT_APPLICABLE"},
    }


def _allocation(ticker: str = "SPUS", weight: float = 0.08) -> dict[str, Any]:
    return {
        "allocations": [{"ticker": ticker, "target_weight": weight}],
        "research_target_exposure": weight,
    }


def _preflight(ticker: str = "SPUS") -> dict[str, Any]:
    return {
        "selection_filter_applied": True,
        "feasible_tickers": [ticker],
    }


def _ready() -> dict[str, Any]:
    return {
        "policy_id": "RL-PORTFOLIO-TEST",
        "causal_candidate_history": True,
        "dedicated_policy_promoted": True,
        "oos_promotion_gate": True,
        "feature_schema_match": True,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
    }


def test_cash_is_only_policy_action_without_promoted_causal_evidence() -> None:
    report = evaluate_shadow_portfolio_rotation(
        ranked=[_candidate()],
        allocation=_allocation(),
        whole_share_preflight=_preflight(),
        config=_config(),
        policy_readiness={},
        proposed_ticker="SPUS",
    )

    assert report["candidate_gate_mask"] == [1, 1]
    assert report["policy_action_mask"] == [1, 0]
    assert report["shadow_action"] == "CASH"
    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["financial_effect_applied"] is False


def test_masked_candidate_can_never_be_selected() -> None:
    candidate = _candidate()
    candidate["research_allocation_blockers"] = ["RISK_BUDGET_EXHAUSTED"]
    report = evaluate_shadow_portfolio_rotation(
        ranked=[candidate],
        allocation=_allocation(),
        whole_share_preflight=_preflight(),
        config=_config(),
        policy_readiness=_ready(),
        proposed_ticker="SPUS",
    )

    assert report["candidate_gate_mask"] == [1, 0]
    assert report["shadow_action"] == "CASH"
    assert "PROPOSAL_CANDIDATE_MASKED" in report["proposal_blockers"]


def test_shadow_selection_never_exceeds_deterministic_allocator_weight() -> None:
    report = evaluate_shadow_portfolio_rotation(
        ranked=[_candidate()],
        allocation=_allocation(weight=0.0375),
        whole_share_preflight=_preflight(),
        config=_config(),
        policy_readiness=_ready(),
        proposed_ticker="SPUS",
    )

    assert report["shadow_action"] == "SPUS"
    assert report["shadow_target_weight"] == 0.0375
    assert report["deterministic_allocator_unchanged"] is True
    assert report["policy_may_increase_weight"] is False
    assert report["execution_authority"] == "NONE"


def test_lifecycle_and_whole_share_gates_fail_closed() -> None:
    candidate = _candidate()
    candidate["lifecycle_state"] = "DISCOVERED"
    report = evaluate_shadow_portfolio_rotation(
        ranked=[candidate],
        allocation=_allocation(),
        whole_share_preflight={"selection_filter_applied": False},
        config=_config(),
        policy_readiness=_ready(),
        proposed_ticker="SPUS",
    )

    blockers = report["candidates"][0]["blockers"]
    assert "LIFECYCLE_NOT_ENTRY_READY:DISCOVERED" in blockers
    assert "WHOLE_SHARE_PREFLIGHT_REQUIRED" in blockers
    assert report["shadow_action"] == "CASH"


def test_physical_product_requires_separate_current_shariah_eligibility() -> None:
    candidate = _candidate("GLD")
    candidate["real_asset_context"] = {
        "product_identity": {
            "product_structure": "PHYSICAL_GOLD_TRUST",
            "physical_structure_verified": True,
            "shariah_product_status": "ATTESTATION_REQUIRED",
        }
    }
    report = evaluate_shadow_portfolio_rotation(
        ranked=[candidate],
        allocation=_allocation("GLD"),
        whole_share_preflight=_preflight("GLD"),
        config=_config(),
        policy_readiness=_ready(),
        proposed_ticker="GLD",
    )

    assert "SHARIAH_PRODUCT_NOT_ELIGIBLE:ATTESTATION_REQUIRED" in (
        report["candidates"][0]["blockers"]
    )
    assert report["shadow_action"] == "CASH"
