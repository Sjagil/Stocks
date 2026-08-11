from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.research.autopilot.components import component_registry_report
from stocks.research.autopilot.contracts import stable_hash
from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger
from stocks.research.autopilot.risk import COST_MODELS
from stocks.research.autopilot.taxonomy import taxonomy_coverage_report
from stocks.research.role_leaderboards import publish_role_leaderboards


AUTHORITY = {
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "broker_calls": 0,
    "order_calls": 0,
}


def research_registry_command(
    project_root: Path,
    command: str,
) -> dict[str, Any]:
    if command == "publish":
        return publish_research_registry(project_root)
    if command == "roles":
        return publish_role_leaderboards(project_root)
    if command in {"coverage", "status"}:
        status_path = (
            project_root
            / "output/research/reports/executive_summary.json"
        )
        if not status_path.exists():
            return publish_research_registry(project_root)
        return json.loads(status_path.read_text(encoding="utf-8"))
    raise ValueError(f"UNKNOWN_RESEARCH_REGISTRY_COMMAND:{command}")


def publish_research_registry(project_root: Path) -> dict[str, Any]:
    generated_at = datetime.now(UTC).isoformat()
    layout = AutopilotLayout(project_root)
    ledger = ResearchLedger(layout)
    try:
        standard = ledger.strategies()
        bulk = ledger.bulk_strategies()
        standard_trials = ledger.trials()
        bulk_trials = ledger.bulk_trials()
        ledger_counts = ledger.counts()
    finally:
        ledger.close()
    components = component_registry_report()
    taxonomy = taxonomy_coverage_report()
    strategy_rows = _strategy_rows(
        standard,
        bulk,
        standard_trials,
        bulk_trials,
    )
    trial_rows = _trial_rows(
        standard_trials,
        bulk_trials,
        {row["strategy_id"]: row for row in strategy_rows},
    )
    outputs: list[Path] = []

    registry_root = project_root / "output/research/registry"
    strategies_root = project_root / "output/research/strategies"
    universe_root = project_root / "output/research/universe"
    data_root = project_root / "output/research/data"
    results_root = project_root / "output/research/results"
    leaderboard_root = project_root / "output/research/leaderboards"
    reports_root = project_root / "output/research/reports"
    for root in (
        registry_root,
        strategies_root,
        universe_root,
        data_root,
        results_root,
        leaderboard_root,
        reports_root,
    ):
        root.mkdir(parents=True, exist_ok=True)

    outputs.append(
        _write_json(
            registry_root / "feature_registry.json",
            {
                **components,
                "taxonomy_hash": taxonomy["taxonomy_hash"],
                "generated_at": generated_at,
                **AUTHORITY,
            },
        )
    )
    feature_coverage = _feature_coverage(
        components["components"],
        standard,
        bulk,
    )
    outputs.append(
        _write_csv(
            registry_root / "feature_coverage.csv",
            feature_coverage,
        )
    )
    outputs.append(
        _write_csv(
            registry_root / "family_pair_coverage.csv",
            _family_pair_coverage(standard, bulk),
        )
    )
    outputs.append(
        _write_csv(
            registry_root / "block_role_coverage.csv",
            _block_role_coverage(standard),
        )
    )

    outputs.append(
        _write_json(
            strategies_root / "strategy_registry.json",
            {
                "schema": "canonical_strategy_registry_v1",
                "status": "GO",
                "strategy_count": len(strategy_rows),
                "standard_strategy_count": len(standard),
                "bulk_strategy_count": len(bulk),
                "strategies": strategy_rows,
                "generated_at": generated_at,
                **AUTHORITY,
            },
        )
    )
    outputs.append(
        _write_csv(
            strategies_root / "strategy_dna.csv",
            strategy_rows,
        )
    )
    generation = _generation_summary(strategy_rows)
    outputs.append(
        _write_json(
            strategies_root / "generation_summary.json",
            {
                "schema": "canonical_generation_summary_v1",
                "status": "GO",
                **generation,
                "generated_at": generated_at,
                **AUTHORITY,
            },
        )
    )
    queue = _queue_status(strategy_rows, trial_rows)
    outputs.append(
        _write_json(
            strategies_root / "queue_status.json",
            {
                "schema": "canonical_research_queue_status_v1",
                "status": "GO",
                **queue,
                "generated_at": generated_at,
                **AUTHORITY,
            },
        )
    )
    outputs.append(
        _write_csv(
            strategies_root / "rejection_reasons.csv",
            _rejection_rows(standard, trial_rows),
        )
    )

    universe = _universe_artifacts(project_root)
    outputs.append(
        _write_json(
            universe_root / "pit_universe_audit.json",
            universe["pit"],
        )
    )
    outputs.append(
        _write_json(
            universe_root / "survivorship_audit.json",
            universe["survivorship"],
        )
    )
    outputs.append(
        _write_json(
            universe_root / "shariah_eligibility.json",
            universe["shariah"],
        )
    )

    data = _data_artifacts(project_root, components["components"])
    outputs.append(
        _write_json(
            data_root / "data_coverage.json",
            data["coverage"],
        )
    )
    outputs.append(
        _write_json(
            data_root / "provider_availability.json",
            data["providers"],
        )
    )
    outputs.append(
        _write_csv(
            data_root / "point_in_time_completeness.csv",
            data["pit"],
        )
    )
    outputs.append(
        _write_csv(
            data_root / "futures_contract_coverage.csv",
            data["futures"],
        )
    )
    outputs.append(
        _write_csv(
            data_root / "macro_vintage_coverage.csv",
            data["macro"],
        )
    )

    normal = [
        row for row in trial_rows if _number(row["cost_bps"]) == 10.0
    ]
    outputs.append(
        _write_csv(results_root / "baseline_results.csv", normal)
    )
    outputs.append(
        _write_csv(results_root / "cost_stress.csv", trial_rows)
    )
    unavailable = _unavailable_result_rows()
    for name in (
        "annual_returns.csv",
        "regime_results.csv",
        "ablation.csv",
    ):
        outputs.append(_write_csv(results_root / name, unavailable))
    phase11 = _phase11_results(project_root)
    outputs.append(
        _write_csv(
            results_root / "walk_forward.csv",
            phase11["folds"],
        )
    )
    outputs.append(
        _write_csv(
            results_root / "holdout.csv",
            phase11["holdout"],
        )
    )
    outputs.append(
        _write_csv(
            results_root / "portfolio_results.csv",
            phase11["portfolio"],
        )
    )
    outputs.append(
        _write_csv(
            results_root / "capacity.csv",
            _capacity_rows(project_root),
        )
    )

    leaderboard_frames = _leaderboards(
        pd.DataFrame(normal),
        pd.DataFrame(phase11["portfolio"]),
    )
    for name, frame in leaderboard_frames.items():
        path = leaderboard_root / f"{name}.html"
        _write_html(path, name, frame)
        outputs.append(path)
    role_leaderboards = publish_role_leaderboards(project_root)
    for role in (
        "strategic_allocation",
        "active_swing",
        "tactical_execution",
        "exploratory_forward",
    ):
        role_root = project_root / "output/research" / role
        outputs.extend(
            path
            for path in (
                role_root / "leaderboard.parquet",
                role_root / "leaderboard.json",
                role_root / "status.json",
            )
            if path.is_file()
        )
    role_status_path = (
        project_root / "output/research/role_leaderboards/status.json"
    )
    if role_status_path.is_file():
        outputs.append(role_status_path)

    evidence = _evidence_summary(
        strategy_rows,
        trial_rows,
        phase11["portfolio"],
    )
    components_rows = components["components"]
    technical_categories = {
        "exit",
        "liquidity",
        "mean_reversion",
        "momentum",
        "regime",
        "sizing",
        "structure",
        "trend",
        "volatility",
        "volume",
    }
    summary = {
        "schema": "canonical_research_registry_summary_v1",
        "status": "GO_WITH_EVIDENCE_GAPS",
        "generated_at": generated_at,
        "registry_feature_count": len(components_rows),
        "technical_feature_count": sum(
            row["category"] in technical_categories
            for row in components_rows
        ),
        "fundamental_feature_count": sum(
            row["category"] in {"fundamental", "valuation"}
            for row in components_rows
        ),
        "earnings_revision_feature_count": sum(
            any(
                token in str(row["name"]).lower()
                for token in ("earnings", "revision", "estimate")
            )
            for row in components_rows
        ),
        "etf_feature_count": sum(
            "ETF" in row.get("supported_assets", [])
            for row in components_rows
        ),
        "commodity_feature_count": sum(
            any(
                asset in {"COMMODITY_ETF", "ETC", "COMMODITY_FUTURE"}
                for asset in row.get("supported_assets", [])
            )
            for row in components_rows
        ),
        "macro_feature_count": sum(
            row["category"] == "macro_regime"
            for row in components_rows
        ),
        "point_in_time_complete_feature_count": sum(
            row["point_in_time_status"] == "PIT_SOURCE_AVAILABLE"
            for row in feature_coverage
        ),
        "provider_dependent_feature_count": sum(
            row["point_in_time_status"] != "PIT_SOURCE_AVAILABLE"
            for row in feature_coverage
        ),
        "strategy_block_count": len(
            {
                value
                for strategy in [*standard, *bulk]
                for value in _components(strategy)
            }
        ),
        "unique_strategy_dna_count": len(strategy_rows),
        "functional_leaderboard_status": role_leaderboards["status"],
        "functional_leaderboards": role_leaderboards["roles"],
        **generation,
        **evidence,
        "ledger_counts": ledger_counts,
        "pit_universe_status": universe["pit"]["status"],
        "survivorship_status": universe["survivorship"]["status"],
        "historical_shariah_status": universe["shariah"][
            "historical_status"
        ],
        "financial_finalist_go": False,
        "forward_shadow_go": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
        "open_evidence_gaps": [
            "POINT_IN_TIME_UNIVERSE_INCOMPLETE",
            "HISTORICAL_SHARIAH_INCOMPLETE",
            "INDEPENDENT_FORWARD_SAMPLE_INCOMPLETE",
            "ANNUAL_REGIME_AND_ABLATION_RESULTS_NOT_CANONICALIZED",
        ],
    }
    outputs.append(
        _write_json(
            reports_root / "executive_summary.json",
            summary,
        )
    )
    master = reports_root / "master_report.html"
    _write_master_report(master, summary, leaderboard_frames)
    outputs.append(master)
    manifest = {
        "schema": "canonical_research_artifact_manifest_v1",
        "status": "GO",
        "generated_at": generated_at,
        "artifact_count": len(outputs),
        "artifacts": [
            {
                "path": str(path.relative_to(project_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _file_hash(path),
            }
            for path in sorted(outputs)
        ],
        **AUTHORITY,
    }
    _write_json(reports_root / "manifest.json", manifest)
    return summary


def _strategy_rows(
    standard: list[dict[str, Any]],
    bulk: list[dict[str, Any]],
    standard_trials: list[dict[str, Any]],
    bulk_trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trial_ids = {
        str(row["strategy_id"])
        for row in [*standard_trials, *bulk_trials]
    }
    rows: list[dict[str, Any]] = []
    for item in standard:
        decision = item.get("latest_decision") or {}
        components = _components(item)
        rows.append(
            {
                "strategy_id": item["strategy_id"],
                "strategy_hash": item["strategy_hash"],
                "registry_source": "CANONICAL_AUTOPILOT",
                "family": item.get("family"),
                "formula": item.get("family"),
                "timeframe": item.get("entry_timeframe"),
                "confirmation_timeframe": item.get(
                    "confirmation_timeframe"
                ),
                "asset_class": (
                    "|".join(item.get("asset_scope", []))
                    or "UNSPECIFIED"
                ),
                "profile": item.get("mutation_type", "EXACT_TEMPLATE"),
                "component_count": len(components),
                "components": "|".join(components),
                "status": decision.get("new_status", "REGISTERED"),
                "research_level": decision.get(
                    "research_level",
                    "REGISTERED",
                ),
                "trial_available": item["strategy_id"] in trial_ids,
                "long_only": item.get("long_only", True),
                "whole_shares": True,
                "base_currency": "EUR",
            }
        )
    for item in bulk:
        components = _components(item)
        rows.append(
            {
                "strategy_id": item["strategy_id"],
                "strategy_hash": item["strategy_hash"],
                "registry_source": "BULK_STRATEGY_DNA",
                "family": item.get("family"),
                "formula": item.get("formula"),
                "timeframe": item.get("timeframe"),
                "confirmation_timeframe": None,
                "asset_class": item.get("asset_class") or "UNSPECIFIED",
                "profile": item.get("profile"),
                "component_count": len(components),
                "components": "|".join(components),
                "status": (
                    "BASELINE_COMPLETE"
                    if item["strategy_id"] in trial_ids
                    else "QUEUED"
                ),
                "research_level": "RESEARCH_ONLY",
                "trial_available": item["strategy_id"] in trial_ids,
                "long_only": item.get("long_only", True),
                "whole_shares": item.get("whole_shares", True),
                "base_currency": item.get("base_currency", "EUR"),
            }
        )
    return sorted(rows, key=lambda row: str(row["strategy_id"]))


def _trial_rows(
    standard_trials: list[dict[str, Any]],
    bulk_trials: list[dict[str, Any]],
    strategies: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for trial in [*standard_trials, *bulk_trials]:
        metrics = trial.get("metrics", {})
        provenance = trial.get("provenance", {})
        strategy = strategies.get(str(trial["strategy_id"]), {})
        cost_bps = provenance.get("cost_bps", trial.get("cost_bps"))
        if cost_bps is None:
            profile = str(trial.get("cost_profile", ""))
            model = COST_MODELS.get(profile)
            cost_bps = model.total_bps if model is not None else None
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "strategy_id": trial["strategy_id"],
                "family": strategy.get("family"),
                "formula": strategy.get("formula"),
                "timeframe": strategy.get("timeframe"),
                "asset_class": strategy.get("asset_class"),
                "profile": strategy.get("profile"),
                "component_count": strategy.get("component_count"),
                "cost_bps": (
                    float(cost_bps) if cost_bps is not None else None
                ),
                "status": trial.get("status"),
                "CAGR": metrics.get("CAGR"),
                "Sharpe": metrics.get("Sharpe"),
                "period_profit_factor": metrics.get(
                    "period_profit_factor"
                ),
                "maximum_drawdown": metrics.get("maximum_drawdown"),
                "fill_count": metrics.get("fill_count"),
                "maximum_gross_exposure": metrics.get(
                    "maximum_gross_exposure"
                ),
                "terminal_nav": metrics.get("terminal_nav"),
                "whole_shares": provenance.get(
                    "whole_shares",
                    True,
                ),
                "next_bar_execution": provenance.get(
                    "next_bar_execution",
                    True,
                ),
                "broker_calls": provenance.get(
                    "broker_calls",
                    0,
                ),
            }
        )
    return rows


def _feature_coverage(
    components: list[dict[str, Any]],
    standard: list[dict[str, Any]],
    bulk: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    usage: Counter[str] = Counter()
    for strategy in [*standard, *bulk]:
        usage.update(_components(strategy))
    return [
        {
            "feature": row["name"],
            "category": row["category"],
            "supported_assets": "|".join(row["supported_assets"]),
            "supported_timeframes": "|".join(
                row["supported_timeframes"]
            ),
            "causality_status": row["causality_status"],
            "test_status": row["test_status"],
            "strategy_usage_count": usage[row["name"]],
            "coverage_status": (
                "USED_IN_STRATEGY_DNA"
                if usage[row["name"]]
                else "REGISTERED_NOT_YET_USED"
            ),
            "point_in_time_status": _feature_pit_status(row),
        }
        for row in components
    ]


def _feature_pit_status(component: dict[str, Any]) -> str:
    category = str(component["category"])
    if category in {
        "fundamental",
        "valuation",
        "macro_regime",
        "commodity",
        "etf",
    }:
        return "PROVIDER_OR_VINTAGE_DEPENDENT"
    if str(component.get("causality_status", "")).startswith("CAUSAL"):
        return "PIT_SOURCE_AVAILABLE"
    return "CAUSALITY_CONTRACT_INCOMPLETE"


def _family_pair_coverage(
    standard: list[dict[str, Any]],
    bulk: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for strategy in [*standard, *bulk]:
        components = sorted(set(_components(strategy)))
        pairs.update(itertools.combinations(components, 2))
    return [
        {
            "component_a": pair[0],
            "component_b": pair[1],
            "strategy_count": count,
            "status": "GENERATED",
        }
        for pair, count in sorted(pairs.items())
    ]


def _block_role_coverage(
    standard: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    roles = {
        "entry": "entry_components",
        "confirmation": "confirmation_components",
        "regime": "regime_components",
        "exit": "exit_components",
        "sizing": "sizing_component",
    }
    counts: Counter[tuple[str, str]] = Counter()
    for strategy in standard:
        for role, field in roles.items():
            values = strategy.get(field, [])
            if isinstance(values, str):
                values = [values]
            for value in values or []:
                counts[(role, str(value))] += 1
    return [
        {
            "block_role": role,
            "component": component,
            "strategy_count": count,
            "status": "USED",
        }
        for (role, component), count in sorted(counts.items())
    ]


def _generation_summary(
    strategies: list[dict[str, Any]],
) -> dict[str, Any]:
    complexity = Counter(
        int(row["component_count"]) for row in strategies
    )
    return {
        "one_block_strategy_count": complexity[1],
        "two_block_strategy_count": complexity[2],
        "three_to_five_block_strategy_count": sum(
            count
            for value, count in complexity.items()
            if 3 <= value <= 5
        ),
        "more_than_five_block_strategy_count": sum(
            count for value, count in complexity.items() if value > 5
        ),
        "family_counts": dict(
            sorted(
                Counter(
                    str(row.get("family") or "UNCLASSIFIED")
                    for row in strategies
                ).items()
            )
        ),
        "formula_count": len(
            {str(row["formula"]) for row in strategies}
        ),
        "timeframe_counts": dict(
            sorted(
                Counter(
                    str(row.get("timeframe") or "UNSPECIFIED")
                    for row in strategies
                ).items()
            )
        ),
        "asset_class_counts": dict(
            sorted(
                Counter(
                    str(row.get("asset_class") or "UNSPECIFIED")
                    for row in strategies
                ).items()
            )
        ),
    }


def _queue_status(
    strategies: list[dict[str, Any]],
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = {
        str(row["strategy_id"])
        for row in trials
        if row.get("status") == "COMPLETE"
    }
    registered = {str(row["strategy_id"]) for row in strategies}
    pending = sorted(registered - completed)
    return {
        "registered_strategy_count": len(registered),
        "strategy_with_complete_trial_count": len(completed),
        "pending_strategy_count": len(pending),
        "pending_strategy_ids_sample": pending[:100],
        "pending_strategy_ids_hash": stable_hash(pending),
        "complete_pending_registry": (
            "output/research/strategies/strategy_registry.json"
        ),
        "resume_supported": True,
        "deduplication_key": "strategy_hash + cost_bps",
        "automatic_promotion": False,
    }


def _rejection_rows(
    standard: list[dict[str, Any]],
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for strategy in standard:
        decision = strategy.get("latest_decision") or {}
        status = str(decision.get("new_status", ""))
        if any(token in status for token in ("REJECT", "BLOCK")):
            rows.append(
                {
                    "strategy_id": strategy["strategy_id"],
                    "status": status,
                    "reason": decision.get("reason"),
                    "source": "CANONICAL_DECISION_LEDGER",
                }
            )
    for trial in trials:
        if (
            trial.get("status") == "COMPLETE"
            and _number(trial.get("period_profit_factor")) < 1.0
            and _number(trial.get("cost_bps")) == 10.0
        ):
            rows.append(
                {
                    "strategy_id": trial["strategy_id"],
                    "status": "WEAK_NEGATIVE",
                    "reason": "NORMAL_COST_PERIOD_PROFIT_FACTOR_BELOW_ONE",
                    "source": "BULK_BASELINE",
                }
            )
    return rows


def _universe_artifacts(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "output/screener/candidate-history.parquet"
    frame = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    symbols = (
        sorted(frame["symbol"].astype(str).str.upper().unique())
        if "symbol" in frame
        else []
    )
    decision_column = (
        "decision_time"
        if "decision_time" in frame
        else "screening_date"
        if "screening_date" in frame
        else None
    )
    latest = (
        frame.sort_values(decision_column)
        .groupby("symbol", sort=False)
        .tail(1)
        if decision_column and not frame.empty
        else frame
    )
    statuses = (
        Counter(latest["shariah_status"].astype(str))
        if "shariah_status" in latest
        else Counter()
    )
    phase = _read_json(
        project_root / "output/research/phase11_13/status.json"
    )
    pit_status = str(
        phase.get(
            "point_in_time_universe_status",
            "CURRENT_UNIVERSE_NOT_PROVEN_PIT",
        )
    )
    common = {
        "generated_at": datetime.now(UTC).isoformat(),
        **AUTHORITY,
    }
    return {
        "pit": {
            "schema": "canonical_pit_universe_audit_v1",
            "status": (
                "GO"
                if pit_status == "PIT_UNIVERSE_GO"
                else "PARTIAL"
            ),
            "current_symbol_count": len(symbols),
            "observation_row_count": len(frame),
            "decision_timestamp_column": decision_column,
            "point_in_time_universe_status": pit_status,
            "current_membership_not_retroactive": True,
            **common,
        },
        "survivorship": {
            "schema": "canonical_survivorship_audit_v1",
            "status": "BLOCKED",
            "current_symbol_count": len(symbols),
            "delisted_security_count": 0,
            "delisted_security_count_is_complete": False,
            "reason": "HISTORICAL_CONSTITUENT_AND_DELISTING_UNIVERSE_UNAVAILABLE",
            **common,
        },
        "shariah": {
            "schema": "canonical_shariah_eligibility_audit_v1",
            "status": "PARTIAL",
            "current_status_counts": dict(sorted(statuses.items())),
            "historical_status": phase.get(
                "historical_shariah_status",
                "SHARIAH_HISTORY_PARTIAL",
            ),
            "automatic_execution_allowed": False,
            **common,
        },
    }


def _data_artifacts(
    project_root: Path,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    daily = list(
        (
            project_root
            / "data/research/critical_trading/yfinance"
        ).glob("*.parquet")
    )
    mtf_root = (
        project_root
        / "data/research/multitimeframe/private"
    )
    multitimeframe = list(mtf_root.rglob("bars.parquet"))
    timeframe_counts: Counter[str] = Counter()
    for path in multitimeframe:
        for part in path.parts:
            if part.startswith("interval="):
                timeframe_counts[part.split("=", 1)[1]] += 1
    contracts = project_root / "output/ibkr/contracts"
    futures_path = contracts / "futures.parquet"
    futures = (
        pd.read_parquet(futures_path).to_dict("records")
        if futures_path.exists()
        else []
    )
    macro_status = _read_json(
        project_root / "output/macro/status.json"
    )
    provider_inventory = _read_json(
        project_root
        / "output/research/phase11_6/provider-inventory.json"
    )
    providers = [
        {
            "provider": "YFINANCE",
            "role": "OHLCV_RESEARCH",
            "available": bool(daily or multitimeframe),
            "daily_file_count": len(daily),
            "multitimeframe_file_count": len(multitimeframe),
        },
        {
            "provider": "IBKR_TWS_PAPER",
            "role": "CONTRACT_AND_READ_ONLY_BROKER_OBSERVATION",
            "available": (contracts / "stocks.parquet").exists(),
            "daily_file_count": 0,
            "multitimeframe_file_count": 0,
        },
        {
            "provider": "SCREENER_PIT_CACHE",
            "role": "FUNDAMENTAL_AND_SHARIAH_CONTEXT",
            "available": (
                project_root
                / "output/screener/candidate-history.parquet"
            ).exists(),
            "daily_file_count": 0,
            "multitimeframe_file_count": 0,
        },
        {
            "provider": "MACRO_CONTEXT",
            "role": "POINT_IN_TIME_CONTEXT_NOT_OHLCV",
            "available": bool(macro_status),
            "daily_file_count": 0,
            "multitimeframe_file_count": 0,
        },
    ]
    pit_rows = [
        {
            "feature": row["name"],
            "category": row["category"],
            "causality_status": row["causality_status"],
            "point_in_time_status": _feature_pit_status(row),
            "provider_required": (
                row["category"]
                in {"fundamental", "macro", "commodity", "etf"}
            ),
        }
        for row in components
    ]
    macro_rows = [
        {
            "dataset": "MACRO_CONTEXT",
            "status": macro_status.get("status", "UNAVAILABLE"),
            "vintage_contract": (
                "AVAILABLE_AT_REQUIRED"
                if macro_status
                else "UNAVAILABLE"
            ),
            "source_artifact": "output/macro/status.json",
        }
    ]
    return {
        "coverage": {
            "schema": "canonical_research_data_coverage_v1",
            "status": "GO_WITH_GAPS",
            "daily_symbol_file_count": len(daily),
            "multitimeframe_bar_file_count": len(multitimeframe),
            "timeframe_file_counts": dict(
                sorted(timeframe_counts.items())
            ),
            "provider_inventory_source_available": bool(
                provider_inventory
            ),
            "synthetic_intraday_allowed": False,
            "broker_calls": 0,
            "order_calls": 0,
        },
        "providers": {
            "schema": "canonical_provider_availability_v1",
            "status": "GO_WITH_GAPS",
            "providers": providers,
            "context_sources_not_used_as_ohlcv": True,
            **AUTHORITY,
        },
        "pit": pit_rows,
        "futures": futures
        or [
            {
                "status": "NO_RESOLVED_FUTURES",
                "con_id": None,
                "symbol": None,
                "expiry": None,
                "multiplier": None,
            }
        ],
        "macro": macro_rows,
    }


def _phase11_results(project_root: Path) -> dict[str, list[dict[str, Any]]]:
    folds: list[dict[str, Any]] = []
    portfolio: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for phase_name in ("phase11_13", "phase11_14"):
        root = project_root / "output/research" / phase_name
        folds_path = root / "fold-results.parquet"
        portfolio_path = root / "strategy-summary.parquet"
        if folds_path.exists():
            folds.extend(
                {
                    **row,
                    "source_phase": phase_name,
                }
                for row in pd.read_parquet(folds_path).to_dict("records")
            )
        if portfolio_path.exists():
            portfolio.extend(
                {
                    **row,
                    "source_phase": phase_name,
                }
                for row in pd.read_parquet(portfolio_path).to_dict("records")
            )
        holdout.extend(
            {
                **row,
                "source_phase": phase_name,
            }
            for row in _read_json(
                root / "latest-forward-observation.json"
            ).get("observations", [])
        )
    return {
        "folds": folds or _unavailable_result_rows(),
        "portfolio": portfolio or _unavailable_result_rows(),
        "holdout": holdout or _unavailable_result_rows(),
    }


def _capacity_rows(project_root: Path) -> list[dict[str, Any]]:
    return _read_json(
        project_root / "output/capital/capacity_report.json"
    ).get("instruments", []) or _unavailable_result_rows()


def _unavailable_result_rows() -> list[dict[str, Any]]:
    return [
        {
            "status": "SOURCE_NOT_CANONICALLY_AVAILABLE",
            "reason": "RESULT_FAMILY_REQUIRES_FUTURE_DEDICATED_EXPORT",
        }
    ]


def _leaderboards(
    baseline: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if not baseline.empty and "Sharpe" in baseline:
        baseline = baseline.sort_values(
            ["Sharpe", "CAGR"],
            ascending=False,
            na_position="last",
        )
    formula = (
        baseline.get("formula", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    asset = (
        baseline.get("asset_class", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    component_count = pd.to_numeric(
        baseline.get("component_count", pd.Series(dtype=float)),
        errors="coerce",
    )
    return {
        "simple": baseline[component_count <= 2].head(100),
        "technical": baseline.head(100),
        "fundamental": baseline[
            formula.str.contains("quality|fundamental|earning|revision")
        ].head(100),
        "etf": baseline[asset.str.contains("ETF")].head(100),
        "commodity": baseline[
            asset.str.contains("COMMODITY")
        ].head(100),
        "macro": baseline[formula.str.contains("macro")].head(100),
        "portfolio": portfolio.head(100),
    }


def _evidence_summary(
    strategies: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    portfolio: list[dict[str, Any]],
) -> dict[str, Any]:
    normal = [
        row for row in trials if _number(row["cost_bps"]) == 10.0
    ]
    stressed = [
        row for row in trials if _number(row["cost_bps"]) >= 50.0
    ]
    historical = {
        str(row["strategy_id"])
        for row in normal
        if _number(row.get("CAGR")) > 0
        and _number(row.get("period_profit_factor")) > 1
    }
    stress_survivors = {
        str(row["strategy_id"])
        for row in stressed
        if _number(row.get("CAGR")) > 0
        and _number(row.get("period_profit_factor")) > 1
    }
    robust = {
        str(row.get("strategy_id"))
        for row in portfolio
        if row.get("robust_pass")
    }
    forward = {
        str(row.get("strategy_id"))
        for row in portfolio
        if row.get("robust_pass")
        and row.get("portfolio_invariants_go")
        and row.get("forward_observer_candidate")
    }
    return {
        "executed_baseline_count": len(normal),
        "historically_positive_strategy_count": len(historical),
        "cost_stress_survivor_count": len(stress_survivors),
        "nested_walk_forward_survivor_count": len(robust),
        "forward_shadow_eligible_strategy_count": len(forward),
        "registered_strategy_count": len(strategies),
    }


def _components(strategy: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in (
        "indicator_components",
        "entry_components",
        "confirmation_components",
        "regime_components",
        "exit_components",
    ):
        raw = strategy.get(field, [])
        if isinstance(raw, str):
            raw = [raw]
        values.extend(str(value) for value in raw or [])
    sizing = strategy.get("sizing_component")
    if sizing:
        values.append(str(sizing))
    return list(dict.fromkeys(values))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    content = dict(payload)
    content.setdefault("content_hash", stable_hash(payload))
    path.write_text(
        json.dumps(
            content,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False)
    return path


def _write_html(path: Path, title: str, frame: pd.DataFrame) -> None:
    table = (
        "<p>No canonical rows available.</p>"
        if frame.empty
        else frame.to_html(index=False, escape=True)
    )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body><h1>{title}</h1>"
        "<p>Research evidence only. Strategy and execution authority: NONE.</p>"
        f"{table}</body></html>",
        encoding="utf-8",
    )


def _write_master_report(
    path: Path,
    summary: dict[str, Any],
    leaderboards: dict[str, pd.DataFrame],
) -> None:
    rows = "".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>"
        for key, value in summary.items()
        if not isinstance(value, (dict, list))
    )
    links = "".join(
        f"<li><a href='../leaderboards/{name}.html'>{name}</a>"
        f" ({len(frame)} rows)</li>"
        for name, frame in leaderboards.items()
    )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Canonical Research Master Report</title></head><body>"
        "<h1>Canonical Research Master Report</h1>"
        "<p>Research evidence only. No paper or live authority.</p>"
        f"<table>{rows}</table><h2>Leaderboards</h2><ul>{links}</ul>"
        "</body></html>",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if pd.notna(result) else 0.0
