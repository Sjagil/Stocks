from __future__ import annotations

from pathlib import Path

from stocks.research.promotion import (
    PROMOTION_REQUIREMENTS,
    PromotionEvidence,
    PromotionStage,
    StrategyLifecycleStage,
    classify_evidence,
    evaluate_lifecycle_transition,
    recover_survivors,
    sample_minimum,
)


def evidence(**overrides: object) -> PromotionEvidence:
    values: dict[str, object] = {
        "candidate_id": "C1",
        "strategy_name": "MA crossover",
        "family": "ma_crossover",
        "timeframe": "1d",
        "source_path": "result.csv",
        "source_row": 0,
        "parameters": "{}",
        "net_cagr": 0.08,
        "net_expectancy": 0.002,
        "profit_factor": 1.12,
        "stressed_profit_factor": 1.01,
        "maximum_drawdown": -0.2,
        "sample_count": 60,
        "positive_periods": 3,
        "total_periods": 5,
        "costs_included": True,
        "lookahead_free": True,
        "repainting_free": True,
        "valid_entry": True,
        "valid_exit": True,
        "valid_risk": True,
        "data_origin": "HISTORICAL_PROVIDER_DATA",
        "statistical_evidence": {
            "PBO_pass": False,
            "DSR_pass": False,
            "multiple_testing_pass": False,
        },
    }
    values.update(overrides)
    return PromotionEvidence(**values)  # type: ignore[arg-type]


def test_statistical_failure_does_not_hard_reject_positive_strategy() -> None:
    decision = classify_evidence(evidence())
    assert decision.stage == PromotionStage.FROZEN_SHADOW
    assert decision.economic_interest is True
    assert decision.best_in_search_proven is False
    assert "BEST_IN_SEARCH_NOT_PROVEN" in decision.limitations


def test_negative_expectancy_is_hard_reject() -> None:
    decision = classify_evidence(evidence(net_expectancy=-0.001))
    assert decision.stage == PromotionStage.REJECT
    assert "NEGATIVE_NET_ECONOMICS" in decision.hard_reject_reasons


def test_missing_cost_proof_is_hard_reject() -> None:
    decision = classify_evidence(evidence(costs_included=False))
    assert decision.stage == PromotionStage.REJECT
    assert "TRANSACTION_COSTS_NOT_PROVEN" in decision.hard_reject_reasons


def test_monthly_sample_threshold_is_frequency_aware() -> None:
    assert sample_minimum("1mo") == 15
    assert sample_minimum("1w") == 25
    assert sample_minimum("1d") == 40
    assert sample_minimum("1h") == 100


def test_recovery_writes_all_required_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "output" / "research" / "run"
    root.mkdir(parents=True)
    (root / "individual_portfolios.csv").write_text(
        "strategy,family,horizon,test_CAGR,test_daily_profit_factor,"
        "test_expectancy,test_trade_count,test_maximum_drawdown,cost_bps\n"
        "MA crossover,ma_crossover,1d,0.08,1.12,0.002,60,-0.20,10\n",
        encoding="utf-8",
    )
    result = recover_survivors(tmp_path)
    assert result["status"] == "GO"
    assert result["survivor_count"] == 1
    for name in (
        "recovered_survivors.json",
        "recovered_survivors.csv",
        "recovered_survivors.html",
        "recovered_survivors_report.md",
    ):
        assert (tmp_path / "output" / "research" / name).exists()


def test_positive_backtest_cannot_skip_forward_and_paper_stages() -> None:
    report = evaluate_lifecycle_transition(
        StrategyLifecycleStage.BACKTEST_POSITIVE,
        StrategyLifecycleStage.LIVE_CANDIDATE,
        {requirement: True for requirement in PROMOTION_REQUIREMENTS[StrategyLifecycleStage.LIVE_CANDIDATE]},
    )
    assert report["status"] == "NO_GO"
    assert report["transition_allowed"] is False
    assert report["automatic_promotion"] is False
    assert report["execution_authority"] == "NONE"


def test_paper_candidate_requires_all_evidence_and_separate_authority() -> None:
    requirements = PROMOTION_REQUIREMENTS[StrategyLifecycleStage.PAPER_ACTIVE]
    report = evaluate_lifecycle_transition(
        StrategyLifecycleStage.PAPER_CANDIDATE,
        StrategyLifecycleStage.PAPER_ACTIVE,
        {requirement: True for requirement in requirements},
    )
    assert report["status"] == "ELIGIBLE_FOR_OPERATOR_REVIEW"
    assert report["authority_granted"] is False
    assert "AUTHORITY_REQUIRES_SEPARATE_OPERATOR_ACTION" in report["blockers"]


def test_missing_forward_evidence_blocks_paper_candidate() -> None:
    report = evaluate_lifecycle_transition(
        StrategyLifecycleStage.FORWARD_OBSERVER,
        StrategyLifecycleStage.PAPER_CANDIDATE,
        {},
    )
    assert report["status"] == "NO_GO"
    assert "canonical_forward_evidence_go" in report["missing_requirements"]
