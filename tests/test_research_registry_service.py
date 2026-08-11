from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocks.research.autopilot.components import component_registry_report
from stocks.research.autopilot.contracts import stable_hash
from stocks.research.autopilot.ledger import (
    AutopilotLayout,
    ResearchLedger,
)
from stocks.research.registry_service import (
    _evidence_summary,
    _feature_pit_status,
    _generation_summary,
    _phase11_results,
    _trial_rows,
    publish_research_registry,
    research_registry_command,
)


def test_closed_bar_features_are_pit_available_without_provider_data() -> None:
    components = component_registry_report()["components"]
    technical = [
        row
        for row in components
        if row["category"]
        not in {"fundamental", "valuation", "macro_regime"}
    ]
    provider_dependent = [
        row
        for row in components
        if row["category"]
        in {"fundamental", "valuation", "macro_regime"}
    ]

    assert technical
    assert all(
        _feature_pit_status(row) == "PIT_SOURCE_AVAILABLE"
        for row in technical
    )
    assert provider_dependent
    assert all(
        _feature_pit_status(row) == "PROVIDER_OR_VINTAGE_DEPENDENT"
        for row in provider_dependent
    )


def test_trial_costs_use_provenance_profile_and_explicit_bulk_cost() -> None:
    strategies = {
        "S1": {
            "strategy_id": "S1",
            "family": "trend",
            "formula": "ma",
            "timeframe": "1d",
            "asset_class": "STOCK",
            "profile": "BASE",
            "component_count": 1,
        }
    }
    standard = [
        {
            "trial_id": "T1",
            "strategy_id": "S1",
            "cost_profile": "DOUBLE",
            "status": "COMPLETE",
            "metrics": {},
            "provenance": {},
        },
        {
            "trial_id": "T2",
            "strategy_id": "S1",
            "cost_profile": "NORMAL",
            "status": "COMPLETE",
            "metrics": {},
            "provenance": {"cost_bps": 12.5},
        },
    ]
    bulk = [
        {
            "trial_id": "T3",
            "strategy_id": "S1",
            "cost_bps": 50.0,
            "status": "COMPLETE",
            "metrics": {},
            "provenance": {},
        }
    ]

    rows = _trial_rows(standard, bulk, strategies)

    assert [row["cost_bps"] for row in rows] == [20.0, 12.5, 50.0]


def test_generation_summary_has_no_empty_json_dimension_keys() -> None:
    summary = _generation_summary(
        [
            {
                "component_count": 1,
                "family": "",
                "formula": "test",
                "timeframe": None,
                "asset_class": "",
            }
        ]
    )

    assert summary["asset_class_counts"] == {"UNSPECIFIED": 1}
    assert summary["timeframe_counts"] == {"UNSPECIFIED": 1}
    assert "" not in summary["asset_class_counts"]


def test_phase11_results_merge_frozen_survivor_generations(
    tmp_path: Path,
) -> None:
    for phase_name, strategy_id, forward in (
        ("phase11_13", "S13", False),
        ("phase11_14", "S14", True),
    ):
        root = tmp_path / "output/research" / phase_name
        root.mkdir(parents=True)
        pd.DataFrame(
            [{"strategy_id": strategy_id, "fold_id": "F1"}]
        ).to_parquet(root / "fold-results.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "strategy_id": strategy_id,
                    "robust_pass": True,
                    "portfolio_invariants_go": True,
                    "forward_observer_candidate": forward,
                }
            ]
        ).to_parquet(root / "strategy-summary.parquet", index=False)
        (root / "latest-forward-observation.json").write_text(
            json.dumps(
                {
                    "observations": [
                        {
                            "strategy_id": strategy_id,
                            "status": "GO",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    results = _phase11_results(tmp_path)
    evidence = _evidence_summary([], [], results["portfolio"])

    assert {row["source_phase"] for row in results["portfolio"]} == {
        "phase11_13",
        "phase11_14",
    }
    assert len(results["folds"]) == 2
    assert len(results["holdout"]) == 2
    assert evidence["nested_walk_forward_survivor_count"] == 2
    assert evidence["forward_shadow_eligible_strategy_count"] == 1


def test_bulk_ledger_read_api_preserves_registered_payload(tmp_path: Path) -> None:
    payload = {
        "family": "trend",
        "formula": "ma_crossover",
        "timeframe": "1d",
        "profile": "BASE",
        "asset_class": "STOCK",
        "indicator_components": ["sma"],
        "long_only": True,
        "whole_shares": True,
        "base_currency": "EUR",
    }
    strategy = {
        **payload,
        "strategy_id": "BULK-TEST",
        "strategy_hash": stable_hash(payload),
    }
    ledger = ResearchLedger(AutopilotLayout(tmp_path))
    try:
        ledger.register_bulk_strategies([strategy])
        trial_id = ledger.append_bulk_trial(
            strategy_id=strategy["strategy_id"],
            cost_bps=50.0,
            status="COMPLETE",
            metrics={"CAGR": 0.1},
            provenance={"broker_calls": 0},
        )
        strategies = ledger.bulk_strategies()
        trials = ledger.bulk_trials()
    finally:
        ledger.close()

    assert strategies[0]["strategy_id"] == "BULK-TEST"
    assert strategies[0]["strategy_hash"] == stable_hash(payload)
    assert trials == [
        {
            "trial_id": trial_id,
            "strategy_id": "BULK-TEST",
            "cost_bps": 50.0,
            "status": "COMPLETE",
            "metrics": {"CAGR": 0.1},
            "provenance": {"broker_calls": 0},
            "trial_hash": trials[0]["trial_hash"],
            "created_at": trials[0]["created_at"],
        }
    ]


def test_publish_registry_creates_authority_none_artifact_contract(
    tmp_path: Path,
) -> None:
    summary = publish_research_registry(tmp_path)
    required = (
        "output/research/registry/feature_registry.json",
        "output/research/registry/feature_coverage.csv",
        "output/research/registry/family_pair_coverage.csv",
        "output/research/registry/block_role_coverage.csv",
        "output/research/strategies/strategy_registry.json",
        "output/research/strategies/strategy_dna.csv",
        "output/research/strategies/generation_summary.json",
        "output/research/strategies/queue_status.json",
        "output/research/strategies/rejection_reasons.csv",
        "output/research/universe/pit_universe_audit.json",
        "output/research/universe/survivorship_audit.json",
        "output/research/universe/shariah_eligibility.json",
        "output/research/data/data_coverage.json",
        "output/research/data/provider_availability.json",
        "output/research/data/point_in_time_completeness.csv",
        "output/research/data/futures_contract_coverage.csv",
        "output/research/data/macro_vintage_coverage.csv",
        "output/research/results/baseline_results.csv",
        "output/research/results/cost_stress.csv",
        "output/research/results/annual_returns.csv",
        "output/research/results/regime_results.csv",
        "output/research/results/ablation.csv",
        "output/research/results/walk_forward.csv",
        "output/research/results/holdout.csv",
        "output/research/results/portfolio_results.csv",
        "output/research/results/capacity.csv",
        "output/research/leaderboards/simple.html",
        "output/research/leaderboards/technical.html",
        "output/research/leaderboards/fundamental.html",
        "output/research/leaderboards/etf.html",
        "output/research/leaderboards/commodity.html",
        "output/research/leaderboards/macro.html",
        "output/research/leaderboards/portfolio.html",
        "output/research/reports/master_report.html",
        "output/research/reports/executive_summary.json",
        "output/research/reports/manifest.json",
    )

    assert summary["status"] == "GO_WITH_EVIDENCE_GAPS"
    assert summary["registry_feature_count"] >= 100
    assert summary["point_in_time_complete_feature_count"] > 0
    assert summary["execution_authority"] == "NONE"
    assert summary["broker_calls"] == 0
    assert summary["order_calls"] == 0
    assert all((tmp_path / relative).exists() for relative in required)

    status = research_registry_command(tmp_path, "status")
    assert status["content_hash"] == json.loads(
        (
            tmp_path
            / "output/research/reports/executive_summary.json"
        ).read_text(encoding="utf-8")
    )["content_hash"]
