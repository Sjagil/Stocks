from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from dataclasses import fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.research.autopilot.components import (
    component_registry,
    component_registry_report as _component_registry_report,
)
from stocks.research.autopilot.accounting import validate_market_session_index
from stocks.research.autopilot.contracts import (
    ResearchLevel,
    StrategyFamily,
    StrategySpec,
    StrategyStatus,
    stable_hash,
)
from stocks.research.autopilot.engine import (
    ANNUAL_PERIODS,
    deterministic_fixture,
    run_backtest,
)
from stocks.research.autopilot.ensemble import VoteMode, build_ensemble
from stocks.research.autopilot.generator import (
    FAMILY_PARAMETER_BOUNDS,
    generate_strategies,
)
from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger
from stocks.research.autopilot.statistics import (
    cohort_stability,
    parameter_neighbor_stability,
    probability_of_backtest_overfitting,
    robustness_statistics,
    sample_classification,
)
from stocks.research.autopilot.taxonomy import taxonomy_coverage_report
from stocks.research.promotion import (
    PromotionEvidence,
    PromotionStage,
    classify_evidence,
)
from stocks.shadow.audit import freeze_integrity


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCREENER_DB = PROJECT_ROOT / "data" / "screener" / "private" / "daily_screener.sqlite3"
YFINANCE_ROOT = PROJECT_ROOT / "data" / "research" / "critical_trading" / "yfinance"
TOTAL_RETURN_ROOT = PROJECT_ROOT / "data" / "total_returns"
AUTHORITY = {
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "paper_strategy_authority": "NONE",
    "live_strategy_authority": "NONE",
    "broker_calls": 0,
    "order_calls": 0,
}


def _parameter_contract_payload(
    contract: tuple[float, float] | type,
) -> str | list[float]:
    if contract is bool:
        return "boolean"
    if not isinstance(contract, tuple) or len(contract) != 2:
        raise ValueError("Unsupported parameter contract")
    return [float(contract[0]), float(contract[1])]


def component_registry_report() -> dict[str, Any]:
    report = _component_registry_report()
    return _publish("components.json", {**report, **AUTHORITY})


def autopilot_taxonomy() -> dict[str, Any]:
    return _publish(
        "component-taxonomy-coverage.json",
        {**taxonomy_coverage_report(), **AUTHORITY},
    )


def autopilot_generate(
    *, budget: int = 100, family: str | None = None, seed: int = 20260726
) -> dict[str, Any]:
    strategies = generate_strategies(budget=budget, family=family, seed=seed)
    with _ledger() as ledger:
        component_inserted = ledger.register_components(component_registry().values())
        result = ledger.register_strategies(strategies)
        parameter_inserted = ledger.register_parameters(strategies)
        counts = ledger.counts()
    payload = {
        "status": "GO",
        "schema": "swing_strategy_generation_v1",
        "seed": seed,
        "family": family or "ALL",
        "requested_budget": budget,
        "generated_count": len(strategies),
        "component_rows_inserted": component_inserted,
        "strategy_rows": result,
        "parameter_rows_inserted": parameter_inserted,
        "strategy_ids": [item.strategy_id for item in strategies],
        "parameter_registry": {
            family.value: {
                name: _parameter_contract_payload(contract)
                for name, contract in bounds.items()
            }
            for family, bounds in FAMILY_PARAMETER_BOUNDS.items()
        },
        "ledger_counts": counts,
        **AUTHORITY,
    }
    return _publish("generation.json", payload)


def autopilot_smoke(*, family: str | None = None) -> dict[str, Any]:
    selected = generate_strategies(budget=100, family=family)
    bars, eligibility = deterministic_fixture()
    results: list[dict[str, Any]] = []
    with _ledger() as ledger:
        ledger.register_components(component_registry().values())
        ledger.register_strategies(selected)
        ledger.register_parameters(selected)
        campaign_id = ledger.register_campaign(
            {
                "cadence": "SMOKE",
                "family": family,
                "data_origin": "SYNTHETIC_TEST_FIXTURE",
                "financial_evidence": False,
            }
        )
        for strategy in selected:
            result = run_backtest(
                strategy,
                bars,
                eligible=eligibility,
                cost_profile="NORMAL",
                fixture=True,
            )
            trial_id = ledger.append_trial(
                campaign_id=campaign_id,
                strategy_id=strategy.strategy_id,
                stage=2,
                cost_profile="NORMAL",
                status=result.status,
                metrics=result.metrics,
                provenance=result.provenance,
            )
            passed = result.status == "COMPLETE" and result.metrics["maximum_exposure"] <= 1.000001
            ledger.append_decision(
                strategy_id=strategy.strategy_id,
                new_status=StrategyStatus.SMOKE_PASS if passed else StrategyStatus.REJECTED,
                research_level=ResearchLevel.NO_CLASSIFICATION,
                reasons=[
                    "ENGINE_CORRECTNESS_FIXTURE_ONLY"
                    if passed
                    else f"SMOKE_FAILED:{result.status}"
                ],
                evidence=result.public_payload(),
            )
            results.append(
                {
                    "trial_id": trial_id,
                    "strategy_id": strategy.strategy_id,
                    "family": strategy.family,
                    "status": "SMOKE_PASS" if passed else "SMOKE_FAIL",
                    "metrics": result.metrics,
                }
            )
    payload = {
        "status": "GO" if all(row["status"] == "SMOKE_PASS" for row in results) else "BLOCKED",
        "schema": "swing_strategy_smoke_v1",
        "fixture_only": True,
        "financial_evidence": False,
        "result_count": len(results),
        "results": results,
        **AUTHORITY,
    }
    return _publish("smoke-results.json", payload)


def autopilot_campaign(
    family: str,
    *,
    cadence: str = "MANUAL",
    max_trials: int = 40,
) -> dict[str, Any]:
    selected_family = StrategyFamily(family)
    strategies = generate_strategies(
        budget=min(max_trials, 100),
        family=selected_family.value,
    )
    bars = _load_yfinance()
    eligibility, eligibility_audit = _load_pit_eligibility(
        bars, family=selected_family.value
    )
    accounting_returns, fx_returns, accounting_contract = _load_accounting_data()
    metadata = _load_asset_metadata()
    fundamental_scores = _load_pit_fundamental_scores(bars)
    with _ledger() as ledger:
        ledger.register_components(component_registry().values())
        ledger.register_strategies(strategies)
        ledger.register_parameters(strategies)
        campaign_id = ledger.register_campaign(
            {
                "cadence": cadence,
                "family": selected_family.value,
                "data_origin": "LOCAL_PROVIDER_CACHE",
                "eligibility_hash": stable_hash(eligibility_audit),
            }
        )
        trials: list[dict[str, Any]] = []
        decision_queue: list[
            tuple[StrategySpec, list[Any], list[tuple[str, Any]]]
        ] = []
        for strategy in strategies:
            profiles = ("NORMAL", "DOUBLE")
            strategy_results = []
            for profile in profiles:
                result = run_backtest(
                    strategy,
                    bars,
                    eligible=eligibility,
                    cost_profile=profile,
                    fixture=False,
                    accounting_returns=accounting_returns,
                    fx_returns=fx_returns,
                    metadata=metadata,
                    data_contract={
                        **accounting_contract,
                        "point_in_time_fundamentals": (
                            eligibility_audit["status"] == "GO"
                        ),
                        "stale_data_gate": True,
                    },
                    fundamental_scores=fundamental_scores,
                )
                if result.status == "COMPLETE":
                    result.metrics["robustness_statistics"] = robustness_statistics(
                        result.returns,
                        closed_episodes=int(
                            result.metrics.get("closed_episodes") or 0
                        ),
                        trial_count=len(strategies),
                        periods_per_year=float(
                            result.provenance.get(
                                "periods_per_year",
                                ANNUAL_PERIODS[strategy.entry_timeframe],
                            )
                        ),
                    )
                trial_id = ledger.append_trial(
                    campaign_id=campaign_id,
                    strategy_id=strategy.strategy_id,
                    stage=3,
                    cost_profile=profile,
                    status=result.status,
                    metrics=result.metrics,
                    provenance={**result.provenance, "eligibility_audit": eligibility_audit},
                )
                strategy_results.append(result)
                trials.append(
                    {
                        "trial_id": trial_id,
                        "strategy_id": strategy.strategy_id,
                        "cost_profile": profile,
                        "status": result.status,
                        "metrics": result.metrics,
                    }
                )
            fold_results = (
                _walk_forward_results(
                    strategy,
                    bars,
                    eligibility,
                    accounting_returns=accounting_returns,
                    fx_returns=fx_returns,
                    metadata=metadata,
                    data_contract={
                        **accounting_contract,
                        "point_in_time_fundamentals": True,
                        "stale_data_gate": True,
                    },
                    fundamental_scores=fundamental_scores,
                )
                if all(result.status == "COMPLETE" for result in strategy_results)
                else []
            )
            for fold_id, result in fold_results:
                trial_id = ledger.append_trial(
                    campaign_id=campaign_id,
                    strategy_id=strategy.strategy_id,
                    stage=4,
                    cost_profile="NORMAL",
                    status=result.status,
                    metrics={**result.metrics, "fold_id": fold_id},
                    provenance={**result.provenance, "fold_id": fold_id},
                )
                trials.append(
                    {
                        "trial_id": trial_id,
                        "strategy_id": strategy.strategy_id,
                        "cost_profile": "NORMAL",
                        "stage": 4,
                        "fold_id": fold_id,
                        "status": result.status,
                        "metrics": result.metrics,
                    }
                )
            decision_queue.append((strategy, strategy_results, fold_results))
        complete_normal_metrics = [
            results[0].metrics
            for _, results, _ in decision_queue
            if results and results[0].status == "COMPLETE"
        ]
        multiple_testing = _campaign_pbo(decision_queue)
        for strategy, strategy_results, fold_results in decision_queue:
            if strategy_results and strategy_results[0].status == "COMPLETE":
                strategy_results[0].metrics["multiple_testing"] = (
                    multiple_testing
                )
                neighbor_metrics = [
                    item
                    for item in complete_normal_metrics
                    if item is not strategy_results[0].metrics
                ]
                strategy_results[0].metrics["parameter_neighbor_stability"] = (
                    parameter_neighbor_stability(neighbor_metrics)
                )
                cohorts = _cohort_stress(
                    strategy,
                    bars,
                    eligibility,
                    accounting_returns=accounting_returns,
                    fx_returns=fx_returns,
                    metadata=metadata,
                    data_contract={
                        **accounting_contract,
                        "point_in_time_fundamentals": True,
                        "stale_data_gate": True,
                    },
                    fundamental_scores=fundamental_scores,
                )
                strategy_results[0].metrics["cohort_stability"] = (
                    cohort_stability(cohorts)
                )
                strategy_results[0].metrics["cohort_metrics"] = cohorts
                strategy_results[0].metrics["concentration_stress"] = (
                    _concentration_stress(
                        strategy,
                        strategy_results[0],
                        bars,
                        eligibility,
                        accounting_returns=accounting_returns,
                        fx_returns=fx_returns,
                        metadata=metadata,
                        data_contract={
                            **accounting_contract,
                            "point_in_time_fundamentals": True,
                            "stale_data_gate": True,
                        },
                        fundamental_scores=fundamental_scores,
                    )
                )
                stage5_metrics = {
                    key: strategy_results[0].metrics[key]
                    for key in (
                        "parameter_neighbor_stability",
                        "cohort_stability",
                        "cohort_metrics",
                        "concentration_stress",
                        "robustness_statistics",
                        "multiple_testing",
                    )
                }
                stage5_trial_id = ledger.append_trial(
                    campaign_id=campaign_id,
                    strategy_id=strategy.strategy_id,
                    stage=5,
                    cost_profile="NORMAL",
                    status="COMPLETE",
                    metrics=stage5_metrics,
                    provenance={
                        "stage": "ROBUSTNESS_AND_COHORT_STRESS",
                        "fixture": False,
                        "code_hash": strategy_results[0].provenance[
                            "code_hash"
                        ],
                    },
                )
                trials.append(
                    {
                        "trial_id": stage5_trial_id,
                        "strategy_id": strategy.strategy_id,
                        "cost_profile": "NORMAL",
                        "stage": 5,
                        "status": "COMPLETE",
                        "metrics": stage5_metrics,
                    }
                )
                stage6_metrics = {
                    key: strategy_results[0].metrics.get(key)
                    for key in (
                        "net_total_return",
                        "benchmark_total_return",
                        "benchmark_champion",
                        "benchmark_results",
                        "benchmark_selection_policy",
                        "excess_total_return",
                        "information_ratio",
                        "maximum_exposure",
                        "maximum_position_weight",
                        "average_cash",
                        "sector_exposure",
                        "region_exposure",
                        "currency_exposure",
                    )
                }
                stage6_trial_id = ledger.append_trial(
                    campaign_id=campaign_id,
                    strategy_id=strategy.strategy_id,
                    stage=6,
                    cost_profile="NORMAL",
                    status="COMPLETE",
                    metrics=stage6_metrics,
                    provenance={
                        "stage": "PORTFOLIO_AND_BENCHMARK",
                        "fixture": False,
                        "code_hash": strategy_results[0].provenance[
                            "code_hash"
                        ],
                    },
                )
                trials.append(
                    {
                        "trial_id": stage6_trial_id,
                        "strategy_id": strategy.strategy_id,
                        "cost_profile": "NORMAL",
                        "stage": 6,
                        "status": "COMPLETE",
                        "metrics": stage6_metrics,
                    }
                )
            _decide(ledger, strategy, strategy_results, fold_results)
        counts = ledger.counts()
    complete = sum(row["status"] == "COMPLETE" for row in trials)
    payload = {
        "status": "GO" if complete else "DATA_BLOCKED",
        "schema": "swing_research_campaign_v1",
        "campaign_id": campaign_id,
        "family": selected_family.value,
        "trial_count": len(trials),
        "complete_trial_count": complete,
        "eligibility": eligibility_audit,
        "trials": trials,
        "ledger_counts": counts,
        "FINANCIAL_FINALIST_GO": False,
        **AUTHORITY,
    }
    return _publish(f"campaign-{selected_family.value}.json", payload)


def autopilot_daily() -> dict[str, Any]:
    generated = autopilot_generate(budget=20)
    status = autopilot_status()
    screener = _daily_screener_summary()
    with _ledger() as ledger:
        observers = len(ledger.forward_registrations())
    return _publish(
        f"daily/{date.today().isoformat()}.json",
        {
            "status": "GO",
            "cadence": "DAILY",
            "actions": ["COMPONENT_AUDIT", "BOUNDED_GENERATION", "STATUS_REFRESH"],
            "generated_count": generated["generated_count"],
            "ledger_counts": status["ledger_counts"],
            "new_assets_with_potential": screener["eligible_latest_count"],
            "active_research_candidates": status["forward_candidate_count"],
            "active_forward_observers": observers,
            "campaigns_started": 0,
            "reason": "DAILY_AUTOPILOT_DOES_NOT_CONSUME_HISTORICAL_ELIGIBILITY",
            "technical_blockers": [],
            "financial_blockers": [
                "HISTORICAL_POINT_IN_TIME_SHARIAH_UNAVAILABLE",
                "FORWARD_EVIDENCE_UNAVAILABLE",
            ],
            "technical_status": "GO",
            "historically_interesting": False,
            "financial_finalist": False,
            "forward_proven": False,
            "paper_authorized": False,
            "live_authorized": False,
            **AUTHORITY,
        },
    )


def autopilot_weekly(*, max_trials: int = 40) -> dict[str, Any]:
    campaigns = [
        autopilot_campaign(family.value, cadence="WEEKLY", max_trials=max_trials)
        for family in StrategyFamily
    ]
    return _publish(
        f"weekly/{date.today().isoformat()}.json",
        {
            "status": "GO" if any(item["status"] == "GO" for item in campaigns) else "DATA_BLOCKED",
            "cadence": "WEEKLY",
            "campaigns": [
                {
                    "campaign_id": item["campaign_id"],
                    "family": item["family"],
                    "status": item["status"],
                }
                for item in campaigns
            ],
            **AUTHORITY,
        },
    )


def autopilot_monthly() -> dict[str, Any]:
    with _ledger() as ledger:
        trials = ledger.trials()
    family_summary: dict[str, dict[str, Any]] = {}
    for family in StrategyFamily:
        rows = [row for row in trials if _strategy_family(row["strategy_id"]) == family.value]
        fixture_complete = [
            row
            for row in rows
            if row["status"] == "COMPLETE"
            and bool(row["provenance"].get("fixture"))
        ]
        actual_complete = [
            row
            for row in rows
            if row["status"] == "COMPLETE"
            and not bool(row["provenance"].get("fixture"))
        ]
        family_summary[family.value] = {
            "trial_count": len(rows),
            "technical_fixture_complete_count": len(fixture_complete),
            "actual_market_complete_count": len(actual_complete),
            "actual_market_median_net_return": _median(
                [
                    row["metrics"].get("net_total_return")
                    for row in actual_complete
                ]
            ),
            "fixture_results_are_financial_evidence": False,
        }
    return _publish(
        f"monthly/{date.today().strftime('%Y-%m')}.json",
        {
            "status": "GO",
            "cadence": "MONTHLY",
            "family_summary": family_summary,
            "automatic_retirements": 0,
            "automatic_authority_changes": 0,
            **AUTHORITY,
        },
    )


def autopilot_candidates() -> dict[str, Any]:
    return _list_by_level(
        {
            ResearchLevel.RESEARCH_WATCHLIST,
            ResearchLevel.FORWARD_OBSERVER_CANDIDATE,
            ResearchLevel.FINANCIAL_FINALIST,
        },
        "candidates.json",
    )


def autopilot_rejected() -> dict[str, Any]:
    with _ledger() as ledger:
        rows = [
            item
            for item in ledger.strategies()
            if item.get("latest_decision")
            and item["latest_decision"]["new_status"] == StrategyStatus.REJECTED
        ]
    return _publish("rejected.json", {"status": "GO", "count": len(rows), "strategies": rows, **AUTHORITY})


def autopilot_strategy(strategy_id: str) -> dict[str, Any]:
    with _ledger() as ledger:
        strategy = ledger.strategy(strategy_id)
    return {
        "status": "GO" if strategy is not None else "NOT_FOUND",
        "strategy": strategy,
        **AUTHORITY,
    }


def autopilot_compare(strategy_ids: list[str]) -> dict[str, Any]:
    rows = [autopilot_strategy(strategy_id)["strategy"] for strategy_id in strategy_ids]
    rows = [row for row in rows if row]
    return {
        "status": "GO",
        "strategy_count": len(rows),
        "comparison": [
            {
                "strategy_id": row["strategy_id"],
                "family": row["family"],
                "latest_decision": row["latest_decision"],
                "latest_metrics": row["trials"][-1]["metrics"] if row["trials"] else None,
            }
            for row in rows
        ],
        **AUTHORITY,
    }


def autopilot_combine(
    strategy_ids: list[str],
    *,
    mode: str = VoteMode.MAJORITY,
) -> dict[str, Any]:
    with _ledger() as ledger:
        strategies = [ledger.strategy(strategy_id) for strategy_id in strategy_ids]
        missing = [
            strategy_id
            for strategy_id, strategy in zip(strategy_ids, strategies, strict=True)
            if strategy is None
        ]
        if missing:
            return {
                "status": "NOT_FOUND",
                "missing_strategy_ids": missing,
                **AUTHORITY,
            }
        resolved_strategies = [
            strategy for strategy in strategies if strategy is not None
        ]
        blocked = [
            strategy["strategy_id"]
            for strategy in resolved_strategies
            if not strategy.get("hypothesis")
            or not strategy.get("latest_decision")
            or strategy["latest_decision"]["new_status"]
            in {StrategyStatus.GENERATED, StrategyStatus.REJECTED}
        ]
        if blocked:
            return {
                "status": "ENSEMBLE_COMPONENT_EVIDENCE_BLOCKED",
                "blocked_strategy_ids": blocked,
                "reason": "COMPONENTS_MUST_BE_INDEPENDENTLY_INTERPRETABLE_AND_SMOKE_PASS",
                **AUTHORITY,
            }
        spec = build_ensemble(
            strategy_ids,
            [str(strategy["family"]) for strategy in resolved_strategies],
            vote_mode=mode,
        )
        registration = ledger.register_ensemble(
            {
                key: value
                for key, value in spec.__dict__.items()
            }
        )
    return _publish(
        f"ensembles/{spec.ensemble_id}.json",
        {
            "status": "GO",
            "ensemble": spec.__dict__,
            "registration": registration,
            **AUTHORITY,
        },
    )


def autopilot_leaderboard() -> dict[str, Any]:
    with _ledger() as ledger:
        rows = [row for row in ledger.trials() if row["status"] == "COMPLETE"]
    actual = [
        row for row in rows if not bool(row["provenance"].get("fixture"))
    ]
    fixtures = [
        row for row in rows if bool(row["provenance"].get("fixture"))
    ]

    def rank(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            values,
            key=lambda row: (
                row["metrics"].get("Sharpe") is not None,
                row["metrics"].get("Sharpe") or -999.0,
            ),
            reverse=True,
        )

    ranked_actual = rank(actual)
    ranked_fixtures = rank(fixtures)
    return _publish(
        "leaderboard.json",
        {
            "status": "GO",
            "actual_market_count": len(ranked_actual),
            "actual_market_rows": ranked_actual[:100],
            "technical_fixture_count": len(ranked_fixtures),
            "technical_fixture_rows": ranked_fixtures[:100],
            "fixture_results_are_financial_evidence": False,
            **AUTHORITY,
        },
    )


def autopilot_audit() -> dict[str, Any]:
    components = _component_registry_report()
    with _ledger() as ledger:
        counts = ledger.counts()
        strategies = ledger.strategies()
    invalid_timeframes = [
        row["strategy_id"]
        for row in strategies
        if row["entry_timeframe"] in {"5m", "15m", "30m"}
    ]
    security = _security_audit()
    frozen = freeze_integrity(PROJECT_ROOT)
    _, _, accounting_contract = _load_accounting_data()
    payload = {
        "status": (
            "GO"
            if not invalid_timeframes
            and security["status"] == "GO"
            and all(
                value == "GO" or value is True
                for value in frozen.values()
            )
            else "BLOCKED"
        ),
        "component_registry": components["status"],
        "ledger_counts": counts,
        "forbidden_timeframe_strategies": invalid_timeframes,
        "closed_candle_contract": "REQUIRED",
        "next_bar_execution": "REQUIRED",
        "historical_shariah_backprojection": 0,
        "financial_finalist_count": _count_level(strategies, ResearchLevel.FINANCIAL_FINALIST),
        "accounting_data_contract": accounting_contract,
        "security_audit": security,
        "frozen_dependency_integrity": frozen,
        "legacy_standalone_paths": {
            "strategy1": "FAIL_CLOSED_INSPIRATION_ONLY",
            "strategy_research_hub_options_orderflow": "BLOCKED",
            "rsi2_timeframe_gate": "SWING_ONLY",
        },
        **AUTHORITY,
    }
    return _publish("audit.json", payload)


def autopilot_status() -> dict[str, Any]:
    with _ledger() as ledger:
        counts = ledger.counts()
        strategies = ledger.strategies()
    payload = {
        "status": "GO",
        "schema": "swing_research_autopilot_status_v1",
        "ledger_counts": counts,
        "family_count": len({row["family"] for row in strategies}),
        "financial_finalist_count": _count_level(strategies, ResearchLevel.FINANCIAL_FINALIST),
        "forward_candidate_count": _count_level(
            strategies, ResearchLevel.FORWARD_OBSERVER_CANDIDATE
        ),
        "bulk_strategy_catalog_count": counts.get(
            "bulk_strategy_dna", 0
        ),
        "bulk_trial_count": counts.get("bulk_trials", 0),
        "minimum_1000_bulk_strategies_registered": (
            counts.get("bulk_strategy_dna", 0) >= 1_000
        ),
        "minimum_2000_bulk_trials_registered": (
            counts.get("bulk_trials", 0) >= 2_000
        ),
        "historical_shariah_status": _pit_status(),
        "FINANCIAL_FINALIST_GO": False,
        "FORWARD_RESEARCH_SHADOW": "blocked",
        **AUTHORITY,
    }
    return _publish("status.json", payload)


def autopilot_freeze() -> dict[str, Any]:
    audit = autopilot_audit()
    status = autopilot_status()
    source_paths = [
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "src" / "stocks" / "daily.py",
        PROJECT_ROOT / "src" / "stocks" / "data" / "multitimeframe.py",
        PROJECT_ROOT / "src" / "stocks" / "readiness.py",
        PROJECT_ROOT / "src" / "stocks" / "research" / "promotion.py",
        PROJECT_ROOT / "src" / "stocks" / "research" / "phase11_8.py",
        PROJECT_ROOT / "src" / "stocks" / "research" / "phase11_9.py",
        PROJECT_ROOT / "src" / "stocks" / "research" / "phase11_12.py",
        *sorted(
            (PROJECT_ROOT / "src" / "stocks" / "research" / "autopilot").glob(
                "*.py"
            )
        ),
        *sorted((PROJECT_ROOT / "src" / "stocks" / "signals").glob("*.py")),
        *sorted(
            (PROJECT_ROOT / "src" / "stocks" / "notifications").glob("*.py")
        ),
        *sorted((PROJECT_ROOT / "src" / "stocks" / "live").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "stocks" / "screener").glob("*.py")),
        PROJECT_ROOT / "docs" / "SIGNALS_AUTOPILOT_AND_LIVE_CANARY.md",
        PROJECT_ROOT / "docs" / "TELEGRAM_NOTIFICATIONS.md",
        PROJECT_ROOT
        / "docs"
        / "PHASE11_8_REALISTIC_MULTISTRATEGY_FORWARD.md",
        PROJECT_ROOT
        / "docs"
        / "PHASE11_9_ACCELERATED_MULTITIMEFRAME_DISCOVERY.md",
        PROJECT_ROOT / "SIGNALS_AUTOPILOT_STATUS.md",
        PROJECT_ROOT / "PHASE11_8_STATUS.md",
        PROJECT_ROOT / "PHASE11_9_STATUS.md",
        PROJECT_ROOT / "TELEGRAM_STATUS.md",
        PROJECT_ROOT / ".env.ibkr.live.example",
        PROJECT_ROOT / ".env.signals.example",
        PROJECT_ROOT / ".env.telegram.example",
    ]
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest().upper()
        for path in source_paths
    }
    technical_go = (
        audit["status"] == "GO"
        and status["family_count"] == 5
        and status["ledger_counts"]["components"] >= 99
        and status["ledger_counts"]["strategies"] >= 26
        and status["ledger_counts"].get("bulk_strategy_dna", 0) >= 1_000
        and status["ledger_counts"].get("bulk_trials", 0) >= 2_000
        and status["strategy_authority"] == "NONE"
        and status["execution_authority"] == "NONE"
    )
    manifest_hash = stable_hash(
        {
            "schema": "swing_research_autopilot_freeze_v1",
            "source_hashes": source_hashes,
            "component_registry_hash": _component_registry_report()[
                "registry_hash"
            ],
            "ledger_counts": status["ledger_counts"],
            "frozen_dependency_integrity": audit[
                "frozen_dependency_integrity"
            ],
        }
    )
    payload = {
        "status": "GO" if technical_go else "BLOCKED",
        "marker": (
            "SWING_RESEARCH_AUTOPILOT_TECHNICAL_FROZEN_GO"
            if technical_go
            else "SWING_RESEARCH_AUTOPILOT_FREEZE_BLOCKED"
        ),
        "schema": "swing_research_autopilot_freeze_v1",
        "manifest_hash": manifest_hash,
        "source_hashes": source_hashes,
        "component_registry_hash": _component_registry_report()[
            "registry_hash"
        ],
        "family_count": status["family_count"],
        "ledger_counts": status["ledger_counts"],
        "frozen_dependency_integrity": audit["frozen_dependency_integrity"],
        "technical_go": technical_go,
        "FINANCIAL_FINALIST_GO": False,
        "financial_blockers": [
            "HISTORICAL_POINT_IN_TIME_SHARIAH_UNAVAILABLE",
            "DELISTING_SETTLEMENT_CONTRACT_INCOMPLETE",
            "FORWARD_EVIDENCE_UNAVAILABLE",
            "PHASE9_FILL_CLOSE_CANARY_OPERATOR_ACTION_REQUIRED",
        ],
        **AUTHORITY,
    }
    frozen = _publish_immutable(
        f"frozen/technical-{manifest_hash}.json",
        payload,
    )
    _publish(
        "freeze-status.json",
        {
            **payload,
            "frozen_artifact": (
                f"output/research/autopilot/frozen/"
                f"technical-{manifest_hash}.json"
            ),
        },
    )
    return frozen


def forward_register(strategy_id: str) -> dict[str, Any]:
    with _ledger() as ledger:
        result = ledger.register_forward(strategy_id)
        if result["status"] == "GO":
            ledger.append_decision(
                strategy_id=strategy_id,
                new_status=StrategyStatus.FORWARD_OBSERVER,
                research_level=ResearchLevel.FORWARD_OBSERVER_CANDIDATE,
                reasons=[
                    "FROZEN_FORWARD_REGISTRATION",
                    "STRATEGY_AUTHORITY_NONE",
                    "EXECUTION_AUTHORITY_NONE",
                ],
                evidence=result,
            )
    payload = {**result, **AUTHORITY}
    if result["status"] == "GO":
        return _publish_immutable(
            f"../forward/registrations/{result['registration_id']}.json",
            payload,
        )
    return payload


def forward_run() -> dict[str, Any]:
    bars = _load_yfinance()
    accounting_returns, fx_returns, accounting_contract = _load_accounting_data()
    metadata = _load_asset_metadata()
    fundamental_scores = _load_pit_fundamental_scores(bars)
    latest_session = (
        max(frame.index.max() for frame in bars.values() if not frame.empty)
        if bars
        else pd.Timestamp(date.today(), tz="UTC")
    )
    session_date = latest_session.date().isoformat()
    with _ledger() as ledger:
        registrations = ledger.forward_registrations()
        observations = []
        for registration in registrations:
            frozen = json.loads(registration["frozen_payload_json"])
            if stable_hash(frozen) != registration["frozen_hash"]:
                raise ValueError("FORWARD_FROZEN_HASH_MISMATCH")
            strategy = strategy_from_payload(frozen)
            eligibility, eligibility_audit = _load_pit_eligibility(
                bars, family=strategy.family
            )
            result = run_backtest(
                strategy,
                bars,
                eligible=eligibility,
                cost_profile="NORMAL",
                fixture=False,
                accounting_returns=accounting_returns,
                fx_returns=fx_returns,
                metadata=metadata,
                data_contract={
                    **accounting_contract,
                    "point_in_time_fundamentals": (
                        eligibility_audit["status"] == "GO"
                    ),
                    "stale_data_gate": True,
                },
                fundamental_scores=fundamental_scores,
            )
            target_positions = (
                {
                    symbol: round(float(weight), 10)
                    for symbol, weight in result.weights.iloc[-1].items()
                    if float(weight) > 0
                }
                if result.status == "COMPLETE" and not result.weights.empty
                else {}
            )
            prior = [
                item
                for item in ledger.forward_observations()
                if item["registration_id"] == registration["registration_id"]
                and item["session_date"] < session_date
            ]
            previous_positions = (
                prior[-1]["payload"].get("target_positions", {}) if prior else {}
            )
            payload = {
                "signal_status": (
                    "OBSERVATION_COMPLETE"
                    if result.status == "COMPLETE"
                    else result.status
                ),
                "strategy_id": strategy.strategy_id,
                "strategy_hash": strategy.strategy_hash,
                "frozen_hash": registration["frozen_hash"],
                "target_positions": target_positions,
                "hypothetical_entries": sorted(
                    set(target_positions) - set(previous_positions)
                ),
                "hypothetical_exits": sorted(
                    set(previous_positions) - set(target_positions)
                ),
                "hypothetical_net_return": (
                    float(result.returns.iloc[-1])
                    if result.status == "COMPLETE" and not result.returns.empty
                    else None
                ),
                "order_intents": [],
                "automatic_orders": 0,
                "parameters_mutated": False,
                "reason": "FROZEN_FORWARD_OBSERVATION_ONLY",
                **AUTHORITY,
            }
            observation_id = ledger.append_forward_observation(
                registration_id=registration["registration_id"],
                session_date=session_date,
                payload=payload,
            )
            observations.append(
                {
                    "observation_id": observation_id,
                    "registration_id": registration["registration_id"],
                    "signal_status": payload["signal_status"],
                    "target_count": len(target_positions),
                    "entry_count": len(payload["hypothetical_entries"]),
                    "exit_count": len(payload["hypothetical_exits"]),
                }
            )
    return _publish(
        f"../forward/{session_date}.json",
        {
            "status": "GO",
            "session_date": session_date,
            "observation_count": len(observations),
            "observations": observations,
            **AUTHORITY,
        },
    )


def forward_status() -> dict[str, Any]:
    with _ledger() as ledger:
        registrations = ledger.forward_registrations()
        observations = ledger.forward_observations()
    return {
        "status": "GO",
        "registration_count": len(registrations),
        "observation_count": len(observations),
        "registrations": registrations,
        "frozen_hashes_valid": all(
            stable_hash(json.loads(row["frozen_payload_json"]))
            == row["frozen_hash"]
            for row in registrations
        ),
        "automatic_orders": 0,
        **AUTHORITY,
    }


def portfolio_backtest(strategy_id: str) -> dict[str, Any]:
    return _portfolio_evaluate(strategy_id, ("NORMAL",))


def portfolio_stress(strategy_id: str) -> dict[str, Any]:
    return _portfolio_evaluate(strategy_id, ("NORMAL", "DOUBLE", "STRESS"))


def _portfolio_evaluate(
    strategy_id: str,
    profiles: tuple[str, ...],
) -> dict[str, Any]:
    with _ledger() as ledger:
        payload = ledger.strategy(strategy_id)
    if payload is None:
        return {"status": "NOT_FOUND", "strategy_id": strategy_id, **AUTHORITY}
    strategy = strategy_from_payload(payload)
    bars = _load_yfinance()
    eligibility, audit = _load_pit_eligibility(bars, family=strategy.family)
    accounting_returns, fx_returns, accounting_contract = _load_accounting_data()
    metadata = _load_asset_metadata()
    fundamental_scores = _load_pit_fundamental_scores(bars)
    results = [
        run_backtest(
            strategy,
            bars,
            eligible=eligibility,
            cost_profile=profile,
            fixture=False,
            accounting_returns=accounting_returns,
            fx_returns=fx_returns,
            metadata=metadata,
            data_contract={
                **accounting_contract,
                "point_in_time_fundamentals": audit["status"] == "GO",
                "stale_data_gate": True,
            },
            fundamental_scores=fundamental_scores,
        ).public_payload()
        for profile in profiles
    ]
    return _publish(
        f"portfolio/{strategy_id}-{'stress' if len(profiles) > 1 else 'backtest'}.json",
        {
            "status": (
                "GO"
                if all(item["status"] == "COMPLETE" for item in results)
                else "DATA_BLOCKED"
            ),
            "strategy_id": strategy_id,
            "eligibility": audit,
            "results": results,
            **AUTHORITY,
        },
    )


def _decide(
    ledger: ResearchLedger,
    strategy: StrategySpec,
    results: list[Any],
    fold_results: list[tuple[str, Any]],
) -> None:
    blocked = [result for result in results if result.status != "COMPLETE"]
    if blocked:
        previous = ledger.latest_decision(strategy.strategy_id)
        retained_status = (
            str(previous["new_status"])
            if previous is not None
            and previous["new_status"] not in {StrategyStatus.REJECTED}
            else StrategyStatus.GENERATED
        )
        ledger.append_decision(
            strategy_id=strategy.strategy_id,
            new_status=retained_status,
            research_level=ResearchLevel.NO_CLASSIFICATION,
            reasons=[
                "RESEARCH_EVIDENCE_DATA_BLOCKED_NOT_STRATEGY_REJECTION",
                *[result.status for result in blocked],
            ],
            evidence={"results": [result.public_payload() for result in results]},
        )
        return
    normal, doubled = results
    metrics = normal.metrics
    evaluable_folds = [
        result for _, result in fold_results if result.status == "COMPLETE"
    ]
    positive_folds = sum(
        result.metrics.get("net_total_return", 0.0) > 0
        and (result.metrics.get("period_profit_factor") or 0.0) > 1.0
        for result in evaluable_folds
    )
    multiple_testing = metrics.get("multiple_testing", {})
    statistics = metrics.get("robustness_statistics", {})
    provenance = getattr(normal, "provenance", {})
    net_total = float(metrics.get("net_total_return") or 0.0)
    episodes = int(
        metrics.get("trade_episodes") or metrics.get("closed_episodes") or 0
    )
    evidence = PromotionEvidence(
        candidate_id=strategy.strategy_id,
        strategy_name=strategy.strategy_id,
        family=strategy.family,
        timeframe=strategy.entry_timeframe,
        source_path="CANONICAL_RESEARCH_LEDGER",
        source_row=0,
        parameters=json.dumps(strategy.parameters, sort_keys=True),
        net_cagr=float(metrics.get("CAGR") or net_total),
        net_expectancy=float(
            metrics.get("expectancy")
            or (net_total / episodes if episodes else 0.0)
        ),
        profit_factor=(
            metrics.get("episode_profit_factor")
            or metrics.get("period_profit_factor")
        ),
        stressed_profit_factor=(
            doubled.metrics.get("episode_profit_factor")
            or doubled.metrics.get("period_profit_factor")
        ),
        maximum_drawdown=metrics.get("maximum_drawdown"),
        sample_count=episodes,
        positive_periods=positive_folds if evaluable_folds else None,
        total_periods=len(evaluable_folds) if evaluable_folds else None,
        costs_included=True,
        lookahead_free=bool(provenance.get("next_bar_execution", True)),
        repainting_free=bool(provenance.get("closed_candles_only", True)),
        valid_entry=bool(strategy.entry_components),
        valid_exit=bool(strategy.exit_components),
        valid_risk=(
            float(metrics.get("maximum_exposure") or 0.0) <= 1.000001
            and float(metrics.get("average_cash") or 0.0) >= -0.000001
        ),
        data_origin=str(provenance.get("bar_origin") or "HISTORICAL_PROVIDER_DATA"),
        statistical_evidence={
            "PBO_pass": (
                multiple_testing.get("PBO") is not None
                and multiple_testing["PBO"] <= 0.50
            ),
            "DSR_pass": (
                statistics.get("deflated_sharpe_probability") is not None
                and statistics["deflated_sharpe_probability"] >= 0.50
            ),
            "multiple_testing_pass": multiple_testing.get("status") == "GO",
        },
    )
    decision = classify_evidence(
        evidence, governance_cap=PromotionStage.FROZEN_SHADOW
    )
    mapping = {
        PromotionStage.REJECT: (
            StrategyStatus.REJECTED,
            ResearchLevel.NO_CLASSIFICATION,
        ),
        PromotionStage.EXPERIMENTAL: (
            StrategyStatus.EXPERIMENTAL,
            ResearchLevel.EXPERIMENTAL,
        ),
        PromotionStage.RESEARCH_CANDIDATE: (
            StrategyStatus.RESEARCH_CANDIDATE,
            ResearchLevel.RESEARCH_WATCHLIST,
        ),
        PromotionStage.FROZEN_SHADOW: (
            StrategyStatus.FROZEN_SHADOW,
            ResearchLevel.FORWARD_OBSERVER_CANDIDATE,
        ),
    }
    new_status, research_level = mapping[decision.stage]
    ledger.append_decision(
        strategy_id=strategy.strategy_id,
        new_status=new_status,
        research_level=research_level,
        reasons=[
            f"USAGE_SPECIFIC_CLASSIFICATION:{decision.stage.value}",
            f"POSITIVE_OOS_FOLDS:{positive_folds}/{len(evaluable_folds)}",
            *decision.hard_reject_reasons,
            *decision.limitations,
        ],
        evidence={
            "results": [result.public_payload() for result in results],
            "folds": [
                {"fold_id": fold_id, **result.public_payload()}
                for fold_id, result in fold_results
            ],
            "promotion_decision": {
                "stage": decision.stage.value,
                "economic_interest": decision.economic_interest,
                "best_in_search_proven": decision.best_in_search_proven,
                "hard_reject_reasons": decision.hard_reject_reasons,
                "limitations": decision.limitations,
                "automatic_authority": decision.automatic_authority,
            },
        },
    )


def _walk_forward_results(
    strategy: StrategySpec,
    bars: dict[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    *,
    fixture: bool = False,
    accounting_returns: pd.DataFrame | None = None,
    fx_returns: pd.DataFrame | None = None,
    metadata: dict[str, dict[str, str]] | None = None,
    data_contract: dict[str, bool] | None = None,
    fundamental_scores: pd.DataFrame | None = None,
) -> list[tuple[str, Any]]:
    eligible_dates = eligibility.index[eligibility.any(axis=1)]
    if len(eligible_dates) < 756:
        return []
    evaluation_dates = eligible_dates[252:]
    boundaries = [
        int(round(index * len(evaluation_dates) / 5))
        for index in range(6)
    ]
    results: list[tuple[str, Any]] = []
    for fold in range(5):
        segment = evaluation_dates[boundaries[fold] : boundaries[fold + 1]]
        if len(segment) < 63:
            continue
        result = run_backtest(
            strategy,
            bars,
            eligible=eligibility,
            cost_profile="NORMAL",
            fixture=fixture,
            evaluation_start=segment[0],
            evaluation_end=segment[-1],
            accounting_returns=accounting_returns,
            fx_returns=fx_returns,
            metadata=metadata,
            data_contract=data_contract,
            fundamental_scores=fundamental_scores,
        )
        embargo_periods = 5
        prior_end_index = max(0, boundaries[fold] - embargo_periods)
        prior_start_index = max(0, prior_end_index - len(segment))
        prior_segment = evaluation_dates[
            prior_start_index:prior_end_index
        ]
        result.metrics["purged_split"] = True
        result.metrics["embargo_periods"] = embargo_periods
        if len(prior_segment) >= 63:
            in_sample = run_backtest(
                strategy,
                bars,
                eligible=eligibility,
                cost_profile="NORMAL",
                fixture=fixture,
                evaluation_start=prior_segment[0],
                evaluation_end=prior_segment[-1],
                accounting_returns=accounting_returns,
                fx_returns=fx_returns,
                metadata=metadata,
                data_contract=data_contract,
                fundamental_scores=fundamental_scores,
            )
            result.metrics["in_sample_sharpe"] = (
                in_sample.metrics.get("Sharpe")
                if in_sample.status == "COMPLETE"
                else None
            )
        results.append((f"OOS-{fold + 1:02d}", result))
    return results


def _campaign_pbo(
    decision_queue: list[
        tuple[StrategySpec, list[Any], list[tuple[str, Any]]]
    ],
) -> dict[str, Any]:
    in_sample: dict[str, dict[str, float | None]] = {}
    out_of_sample: dict[str, dict[str, float | None]] = {}
    for strategy, _, folds in decision_queue:
        for fold_id, result in folds:
            in_sample.setdefault(fold_id, {})[strategy.strategy_id] = (
                result.metrics.get("in_sample_sharpe")
            )
            out_of_sample.setdefault(fold_id, {})[strategy.strategy_id] = (
                result.metrics.get("Sharpe")
            )
    if not in_sample:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "PBO": None,
            "fold_count": 0,
            "configuration_count": len(decision_queue),
        }
    return probability_of_backtest_overfitting(
        pd.DataFrame.from_dict(in_sample, orient="index"),
        pd.DataFrame.from_dict(out_of_sample, orient="index"),
    )


def _cohort_stress(
    strategy: StrategySpec,
    bars: dict[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    *,
    accounting_returns: pd.DataFrame,
    fx_returns: pd.DataFrame,
    metadata: dict[str, dict[str, str]],
    data_contract: dict[str, bool],
    fundamental_scores: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for symbol in eligibility.columns:
        detail = metadata.get(symbol, {})
        for field in ("region", "sector"):
            value = str(detail.get(field) or "UNKNOWN")
            if value != "UNKNOWN":
                groups.setdefault(f"{field}:{value}", []).append(symbol)
    candidates = sorted(
        (
            (name, sorted(set(symbols)))
            for name, symbols in groups.items()
            if len(set(symbols)) >= 3
        ),
        key=lambda item: (-len(item[1]), item[0]),
    )[:6]
    result: dict[str, dict[str, Any]] = {}
    for name, symbols in candidates:
        cohort_eligibility = eligibility.copy()
        cohort_eligibility.loc[
            :, [column for column in eligibility.columns if column not in symbols]
        ] = False
        evaluation = run_backtest(
            strategy,
            bars,
            eligible=cohort_eligibility,
            cost_profile="NORMAL",
            fixture=False,
            accounting_returns=accounting_returns,
            fx_returns=fx_returns,
            metadata=metadata,
            data_contract=data_contract,
            fundamental_scores=fundamental_scores,
        )
        if evaluation.status != "COMPLETE":
            continue
        result[name] = {
            "net_total_return": evaluation.metrics["net_total_return"],
            "maximum_drawdown": evaluation.metrics["maximum_drawdown"],
            "closed_episodes": evaluation.metrics["closed_episodes"],
            "sample_status": sample_classification(
                observations=evaluation.metrics["observations"],
                closed_episodes=evaluation.metrics["closed_episodes"],
                active_months=max(
                    0, int(evaluation.metrics["observations"] / 21)
                ),
            ),
        }
    return result


def _concentration_stress(
    strategy: StrategySpec,
    full_result: Any,
    bars: dict[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    *,
    accounting_returns: pd.DataFrame,
    fx_returns: pd.DataFrame,
    metadata: dict[str, dict[str, str]],
    data_contract: dict[str, bool],
    fundamental_scores: pd.DataFrame,
) -> dict[str, Any]:
    symbol = full_result.metrics.get("top_positive_contributor")
    if not symbol or symbol not in eligibility:
        return {"status": "INSUFFICIENT_CONCENTRATION_EVIDENCE"}
    without = eligibility.copy()
    without.loc[:, symbol] = False
    result = run_backtest(
        strategy,
        bars,
        eligible=without,
        cost_profile="NORMAL",
        fixture=False,
        accounting_returns=accounting_returns,
        fx_returns=fx_returns,
        metadata=metadata,
        data_contract=data_contract,
        fundamental_scores=fundamental_scores,
    )
    if result.status != "COMPLETE":
        return {
            "status": "INSUFFICIENT_CONCENTRATION_EVIDENCE",
            "removed_symbol": symbol,
            "result_status": result.status,
        }
    full_return = float(full_result.metrics["net_total_return"])
    without_return = float(result.metrics["net_total_return"])
    degradation = (
        (full_return - without_return) / abs(full_return)
        if full_return != 0
        else None
    )
    stable = without_return > 0 and (
        degradation is None or degradation <= 0.50
    )
    return {
        "status": "GO" if stable else "CONCENTRATION_FRAGILITY",
        "removed_symbol": symbol,
        "without_symbol_net_total_return": without_return,
        "relative_degradation": degradation,
    }


def _load_yfinance() -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for path in sorted(YFINANCE_ROOT.glob("*.parquet")):
        frame = pd.read_parquet(path)
        columns = {str(column): str(column).lower() for column in frame.columns}
        frame = frame.rename(columns=columns)
        if "session_date" in frame:
            frame.index = pd.to_datetime(frame["session_date"], utc=True)
        elif not isinstance(frame.index, pd.DatetimeIndex):
            continue
        else:
            frame.index = pd.to_datetime(frame.index, utc=True)
        required = {"open", "high", "low", "close"}
        if not required.issubset(frame.columns):
            continue
        selected = frame[[column for column in ("open", "high", "low", "close", "volume") if column in frame]]
        selected = selected[~selected.index.duplicated(keep="last")].sort_index()
        selected["is_closed"] = True
        bars[path.stem.upper()] = selected
    return bars


def _load_accounting_data(
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    manifest_path = TOTAL_RETURN_ROOT / "total_return_manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {
            "corporate_actions": False,
            "delisting_settlement": False,
            "eur_fx_accounting": False,
            "market_calendar": False,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total: dict[str, pd.Series] = {}
    fx: dict[str, pd.Series] = {}
    calendar_checks: list[bool] = []
    metadata = _load_asset_metadata()
    for path in sorted(
        TOTAL_RETURN_ROOT.glob(
            "security_type=STK/con_id=*/interval=1d/total_returns.parquet"
        )
    ):
        frame = pd.read_parquet(path)
        if frame.empty or not {
            "symbol",
            "session_date",
            "eur_total_return",
            "fx_return",
        }.issubset(frame.columns):
            continue
        symbol = str(frame["symbol"].iloc[0]).upper()
        index = pd.to_datetime(frame["session_date"], utc=True)
        total[symbol] = pd.Series(
            pd.to_numeric(frame["eur_total_return"], errors="coerce").to_numpy(),
            index=index,
        )
        fx[symbol] = pd.Series(
            pd.to_numeric(frame["fx_return"], errors="coerce").to_numpy(),
            index=index,
        )
        exchange = metadata.get(symbol, {}).get("exchange")
        if exchange:
            calendar_checks.append(
                validate_market_session_index(
                    pd.DatetimeIndex(index), exchange=exchange
                )["status"]
                == "GO"
            )
        else:
            calendar_checks.append(False)
    contract = {
        "corporate_actions": bool(
            manifest.get("status") == "GO"
            and manifest.get("split_adjusted")
            and manifest.get("dividend_adjusted")
        ),
        # The current Phase 5 universe is active-only; historical delisting
        # settlement is therefore not silently claimed.
        "delisting_settlement": False,
        "eur_fx_accounting": bool(
            manifest.get("status") == "GO"
            and manifest.get("base_currency") == "EUR"
            and manifest.get("currency_adjusted")
        ),
        "market_calendar": bool(calendar_checks and all(calendar_checks)),
    }
    return (
        pd.DataFrame(total).sort_index(),
        pd.DataFrame(fx).sort_index(),
        contract,
    )


def _load_asset_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    contract_path = (
        PROJECT_ROOT / "output" / "ibkr" / "contracts" / "stocks.parquet"
    )
    if contract_path.exists():
        contracts = pd.read_parquet(contract_path)
        for row in contracts.to_dict("records"):
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                metadata[symbol] = {
                    "sector": str(row.get("industry") or "UNKNOWN"),
                    "region": "UNKNOWN",
                    "currency": str(row.get("currency") or "UNKNOWN"),
                    "asset_type": "STOCK_OR_ETF",
                    "exchange": str(
                        row.get("primary_exchange")
                        or row.get("exchange")
                        or "UNKNOWN"
                    ),
                }
    if not SCREENER_DB.exists():
        return metadata
    connection = sqlite3.connect(f"file:{SCREENER_DB}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT symbol, public_json FROM screener_observations
            ORDER BY screening_date, observation_id
            """
        ).fetchall()
    finally:
        connection.close()
    for symbol, raw in rows:
        payload = json.loads(raw)
        key = str(symbol).upper()
        prior = metadata.get(key, {})
        metadata[key] = {
            "sector": str(payload.get("sector") or prior.get("sector") or "UNKNOWN"),
            "region": str(payload.get("region") or prior.get("region") or "UNKNOWN"),
            "currency": str(
                payload.get("currency") or prior.get("currency") or "UNKNOWN"
            ),
            "asset_type": str(
                payload.get("asset_type") or prior.get("asset_type") or "UNKNOWN"
            ),
            "exchange": str(
                payload.get("exchange") or prior.get("exchange") or "UNKNOWN"
            ),
        }
    return metadata


def _load_pit_fundamental_scores(
    bars: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    index = (
        pd.DatetimeIndex(
            sorted({stamp for frame in bars.values() for stamp in frame.index})
        )
        if bars
        else pd.DatetimeIndex([], tz="UTC")
    )
    scores = pd.DataFrame(float("nan"), index=index, columns=sorted(bars))
    if not SCREENER_DB.exists():
        return scores
    connection = sqlite3.connect(f"file:{SCREENER_DB}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT screening_date, symbol, public_json
            FROM screener_observations ORDER BY screening_date, symbol
            """
        ).fetchall()
    finally:
        connection.close()
    for screening_date, symbol, raw in rows:
        payload = json.loads(raw)
        score = payload.get("fundamental_score")
        timestamp = pd.Timestamp(screening_date, tz="UTC")
        if (
            score is not None
            and symbol in scores.columns
            and timestamp in scores.index
        ):
            scores.loc[timestamp, symbol] = float(score)
    return scores


def _load_pit_eligibility(
    bars: dict[str, pd.DataFrame],
    *,
    family: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = pd.DatetimeIndex(
        sorted({stamp for frame in bars.values() for stamp in frame.index}), tz="UTC"
    ) if bars else pd.DatetimeIndex([], tz="UTC")
    eligibility = pd.DataFrame(False, index=index, columns=sorted(bars))
    observations = 0
    eligible_observations = 0
    if SCREENER_DB.exists():
        connection = sqlite3.connect(f"file:{SCREENER_DB}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                """
                SELECT screening_date, symbol, classification, public_json
                FROM screener_observations ORDER BY screening_date, symbol
                """
            ).fetchall()
        finally:
            connection.close()
        for screening_date, symbol, classification, public_json in rows:
            observations += 1
            public = json.loads(public_json)
            asset_type = str(public.get("asset_type") or "").upper()
            family_compatible = _family_asset_compatible(
                family, asset_type, str(symbol)
            )
            eligible = (
                classification in {"HIGH_POTENTIAL", "WATCHLIST"}
                and public.get("shariah_status") == "SHARIAH_COMPLIANT"
                and not public.get("rejection_reasons")
                and family_compatible
            )
            if eligible and symbol in eligibility.columns:
                timestamp = pd.Timestamp(screening_date, tz="UTC")
                if timestamp in eligibility.index:
                    eligibility.loc[timestamp, symbol] = True
                    eligible_observations += 1
    audit = {
        "status": "GO" if eligible_observations else "PIT_ELIGIBILITY_UNAVAILABLE",
        "screener_observation_count": observations,
        "eligible_observation_count": eligible_observations,
        "backprojected_rows": 0,
        "policy": "EXACT_SCREENING_DATE_ONLY",
        "family": family,
        "database": str(SCREENER_DB),
    }
    return eligibility, audit


def _pit_status() -> dict[str, Any]:
    _, audit = _load_pit_eligibility({})
    return audit


def _daily_screener_summary() -> dict[str, Any]:
    if not SCREENER_DB.exists():
        return {"status": "UNAVAILABLE", "latest_date": None, "eligible_latest_count": 0}
    connection = sqlite3.connect(f"file:{SCREENER_DB}?mode=ro", uri=True)
    try:
        latest = connection.execute(
            "SELECT max(screening_date) FROM screener_observations"
        ).fetchone()[0]
        count = connection.execute(
            """
            SELECT count(*) FROM screener_observations
            WHERE screening_date=?
              AND classification IN ('HIGH_POTENTIAL','WATCHLIST')
            """,
            (latest,),
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "status": "GO",
        "latest_date": latest,
        "eligible_latest_count": int(count),
    }


def _family_asset_compatible(
    family: str | None,
    asset_type: str,
    symbol: str,
) -> bool:
    if family in {
        StrategyFamily.QUALITY_MOMENTUM.value,
        StrategyFamily.TREND_PULLBACK.value,
        StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT.value,
    }:
        return asset_type == "STOCK"
    if family == StrategyFamily.ETF_ROTATION.value:
        return asset_type in {"ETF", "SECTOR_ETF", "REGIONAL_ETF", "SHARIAH_ETF"}
    if family == StrategyFamily.COMMODITY_ETF_TREND.value:
        return asset_type in {"COMMODITY_ETF", "COMMODITY_ETC", "ETC"}
    return bool(symbol)


def _publish(relative: str, payload: dict[str, Any]) -> dict[str, Any]:
    root = AutopilotLayout(PROJECT_ROOT).output_root
    path = (root / relative).resolve()
    if not path.is_relative_to((PROJECT_ROOT / "output" / "research").resolve()):
        raise ValueError("ARTIFACT_PATH_OUTSIDE_RESEARCH_OUTPUT")
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "generated_at": datetime.now(UTC).isoformat(),
        "content_hash": stable_hash(payload),
    }
    path.write_text(json.dumps(enriched, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return enriched


def _publish_immutable(relative: str, payload: dict[str, Any]) -> dict[str, Any]:
    root = AutopilotLayout(PROJECT_ROOT).output_root
    path = (root / relative).resolve()
    if not path.is_relative_to((PROJECT_ROOT / "output" / "research").resolve()):
        raise ValueError("ARTIFACT_PATH_OUTSIDE_RESEARCH_OUTPUT")
    digest = stable_hash(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_hash") != digest:
            raise ValueError("FROZEN_ARTIFACT_IMMUTABILITY_CONFLICT")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "frozen_at": datetime.now(UTC).isoformat(),
        "content_hash": digest,
    }
    path.write_text(
        json.dumps(enriched, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return enriched


def _list_by_level(levels: set[ResearchLevel], filename: str) -> dict[str, Any]:
    with _ledger() as ledger:
        rows = [
            item
            for item in ledger.strategies()
            if item.get("latest_decision")
            and item["latest_decision"]["research_level"] in {level.value for level in levels}
        ]
    return _publish(filename, {"status": "GO", "count": len(rows), "strategies": rows, **AUTHORITY})


def _count_level(strategies: list[dict[str, Any]], level: ResearchLevel) -> int:
    return sum(
        bool(item.get("latest_decision"))
        and item["latest_decision"]["research_level"] == level.value
        for item in strategies
    )


def _strategy_family(strategy_id: str) -> str | None:
    with _ledger() as ledger:
        strategy = ledger.strategy(strategy_id)
    return None if strategy is None else str(strategy["family"])


def _median(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else float(pd.Series(numeric).median())


def _security_audit() -> dict[str, Any]:
    source_root = PROJECT_ROOT / "src" / "stocks" / "research" / "autopilot"
    forbidden_methods = (
        "place" + "Order",
        "cancel" + "Order",
        "req" + "Global" + "Cancel",
        "req" + "Ids",
        "req" + "Auto" + "Open" + "Orders",
        "exercise" + "Options",
        "req" + "Mkt" + "Data",
        "req" + "Historical" + "Data",
    )
    method_hits: list[str] = []
    for path in sorted(source_root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for method in forbidden_methods:
            if re.search(rf"\.\s*{re.escape(method)}\s*\(", text):
                method_hits.append(f"{path.name}:{method}")
    leak_patterns = {
        "raw_account_id": re.compile(r"\b(?:DU|U)\d{6,}\b"),
        "secret_assignment": re.compile(
            r"(?i)(?:api[_-]?key|password|secret)\s*[=:]\s*[\"'][^\"']+"
        ),
        "public_financial_value": re.compile(
            r"(?i)(?:NetLiquidation|AvailableFunds|BuyingPower)"
        ),
    }
    leaks: list[str] = []
    for root in (
        AutopilotLayout(PROJECT_ROOT).output_root,
        AutopilotLayout(PROJECT_ROOT).forward_root,
    ):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".jsonl"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in leak_patterns.items():
                if pattern.search(text):
                    leaks.append(f"{path.relative_to(PROJECT_ROOT)}:{name}")
    return {
        "status": "GO" if not method_hits and not leaks else "NO_GO",
        "forbidden_broker_method_hits": method_hits,
        "public_privacy_leaks": leaks,
        "broker_write_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "account_leaks": 0 if not any("raw_account_id" in item for item in leaks) else 1,
        "secret_leaks": 0 if not any("secret_assignment" in item for item in leaks) else 1,
    }


class _LedgerContext:
    def __enter__(self) -> ResearchLedger:
        self.ledger = ResearchLedger(AutopilotLayout(PROJECT_ROOT))
        return self.ledger

    def __exit__(self, *_: Any) -> None:
        self.ledger.close()


def _ledger() -> _LedgerContext:
    return _LedgerContext()


def strategy_from_payload(payload: dict[str, Any]) -> StrategySpec:
    family = StrategyFamily(payload["family"])
    default_scope = (
        ("STOCK",)
        if family in {StrategyFamily.QUALITY_MOMENTUM, StrategyFamily.TREND_PULLBACK}
        else ("ETF", "SECTOR_ETF", "REGIONAL_ETF", "SHARIAH_ETF")
        if family == StrategyFamily.ETF_ROTATION
        else ("STOCK", "ETF")
        if family == StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT
        else ("COMMODITY_ETF", "COMMODITY_ETC", "ETC")
    )
    payload = {
        "asset_scope": default_scope,
        "long_only": True,
        "leverage_allowed": False,
        "shorting_allowed": False,
        **payload,
    }
    names = {field.name for field in fields(StrategySpec)}
    return StrategySpec(**{key: payload[key] for key in names})
