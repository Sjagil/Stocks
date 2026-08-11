from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.execution.idempotency import stable_hash


class PromotionStage(StrEnum):
    REJECT = "REJECT"
    EXPERIMENTAL = "EXPERIMENTAL"
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
    FROZEN_SHADOW = "FROZEN_SHADOW"
    MANUAL_SIGNAL_CANDIDATE = "MANUAL_SIGNAL_CANDIDATE"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    LIVE_CANARY_CANDIDATE = "LIVE_CANARY_CANDIDATE"
    CONTROLLED_LIVE = "CONTROLLED_LIVE"


class StrategyLifecycleStage(StrEnum):
    IDEA = "IDEA"
    RESEARCH = "RESEARCH"
    BACKTEST_POSITIVE = "BACKTEST_POSITIVE"
    ROBUSTNESS_PENDING = "ROBUSTNESS_PENDING"
    FORWARD_OBSERVER = "FORWARD_OBSERVER"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    PAPER_ACTIVE = "PAPER_ACTIVE"
    PAPER_PROVEN = "PAPER_PROVEN"
    LIVE_CANDIDATE = "LIVE_CANDIDATE"
    LIVE_CANARY = "LIVE_CANARY"
    LIVE_ACTIVE = "LIVE_ACTIVE"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


LIFECYCLE_TRANSITIONS: dict[
    StrategyLifecycleStage, frozenset[StrategyLifecycleStage]
] = {
    StrategyLifecycleStage.IDEA: frozenset({StrategyLifecycleStage.RESEARCH}),
    StrategyLifecycleStage.RESEARCH: frozenset(
        {StrategyLifecycleStage.BACKTEST_POSITIVE, StrategyLifecycleStage.RETIRED}
    ),
    StrategyLifecycleStage.BACKTEST_POSITIVE: frozenset(
        {StrategyLifecycleStage.ROBUSTNESS_PENDING, StrategyLifecycleStage.RESEARCH}
    ),
    StrategyLifecycleStage.ROBUSTNESS_PENDING: frozenset(
        {StrategyLifecycleStage.FORWARD_OBSERVER, StrategyLifecycleStage.RESEARCH}
    ),
    StrategyLifecycleStage.FORWARD_OBSERVER: frozenset(
        {StrategyLifecycleStage.PAPER_CANDIDATE, StrategyLifecycleStage.SUSPENDED}
    ),
    StrategyLifecycleStage.PAPER_CANDIDATE: frozenset(
        {StrategyLifecycleStage.PAPER_ACTIVE, StrategyLifecycleStage.SUSPENDED}
    ),
    StrategyLifecycleStage.PAPER_ACTIVE: frozenset(
        {StrategyLifecycleStage.PAPER_PROVEN, StrategyLifecycleStage.SUSPENDED}
    ),
    StrategyLifecycleStage.PAPER_PROVEN: frozenset(
        {StrategyLifecycleStage.LIVE_CANDIDATE, StrategyLifecycleStage.SUSPENDED}
    ),
    StrategyLifecycleStage.LIVE_CANDIDATE: frozenset(
        {StrategyLifecycleStage.LIVE_CANARY, StrategyLifecycleStage.SUSPENDED}
    ),
    StrategyLifecycleStage.LIVE_CANARY: frozenset(
        {StrategyLifecycleStage.LIVE_ACTIVE, StrategyLifecycleStage.SUSPENDED}
    ),
    StrategyLifecycleStage.LIVE_ACTIVE: frozenset(
        {StrategyLifecycleStage.SUSPENDED}
    ),
    StrategyLifecycleStage.SUSPENDED: frozenset(
        {StrategyLifecycleStage.RESEARCH, StrategyLifecycleStage.RETIRED}
    ),
    StrategyLifecycleStage.RETIRED: frozenset(),
}


PROMOTION_REQUIREMENTS: dict[StrategyLifecycleStage, tuple[str, ...]] = {
    StrategyLifecycleStage.BACKTEST_POSITIVE: (
        "positive_net_expectancy",
        "costs_included",
        "lookahead_free",
    ),
    StrategyLifecycleStage.FORWARD_OBSERVER: (
        "multiple_assets",
        "multiple_regimes",
        "walk_forward_go",
        "cost_stress_go",
        "spread_stress_go",
        "slippage_stress_go",
        "parameter_stability_go",
        "monte_carlo_go",
        "bootstrap_go",
        "oos_go",
    ),
    StrategyLifecycleStage.PAPER_CANDIDATE: (
        "canonical_forward_evidence_go",
        "minimum_forward_episodes_go",
        "data_capabilities_go",
        "brokerability_go",
        "capacity_go",
    ),
    StrategyLifecycleStage.PAPER_ACTIVE: (
        "operator_approval",
        "phase9_paper_proven",
        "implementation_parity_go",
    ),
    StrategyLifecycleStage.PAPER_PROVEN: (
        "paper_round_trips_go",
        "paper_drawdown_go",
        "paper_operational_reliability_go",
        "canonical_paper_labels_go",
    ),
    StrategyLifecycleStage.LIVE_CANDIDATE: (
        "paper_proven",
        "forward_evidence_go",
        "writer_integrity_go",
        "live_reconciliation_go",
        "capacity_go",
    ),
    StrategyLifecycleStage.LIVE_CANARY: (
        "explicit_operator_approval",
        "live_canary_capability_go",
        "kill_switch_go",
    ),
    StrategyLifecycleStage.LIVE_ACTIVE: (
        "live_canary_round_trips_go",
        "live_expectancy_go",
        "live_operational_reliability_go",
        "explicit_operator_approval",
    ),
}


def evaluate_lifecycle_transition(
    current: StrategyLifecycleStage,
    target: StrategyLifecycleStage,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    allowed = target in LIFECYCLE_TRANSITIONS[current]
    requirements = PROMOTION_REQUIREMENTS.get(target, ())
    missing = sorted(
        requirement
        for requirement in requirements
        if evidence.get(requirement) is not True
    )
    eligible = allowed and not missing
    authority_stage = target in {
        StrategyLifecycleStage.PAPER_ACTIVE,
        StrategyLifecycleStage.LIVE_CANARY,
        StrategyLifecycleStage.LIVE_ACTIVE,
    }
    blockers = []
    if not allowed:
        blockers.append("INVALID_LIFECYCLE_TRANSITION")
    blockers.extend(f"MISSING_EVIDENCE:{item}" for item in missing)
    if authority_stage:
        blockers.append("AUTHORITY_REQUIRES_SEPARATE_OPERATOR_ACTION")
    return {
        "schema": "strategy_lifecycle_transition_gate_v1",
        "status": "ELIGIBLE_FOR_OPERATOR_REVIEW" if eligible else "NO_GO",
        "current_stage": current.value,
        "target_stage": target.value,
        "transition_allowed": allowed,
        "requirements": list(requirements),
        "missing_requirements": missing,
        "automatic_promotion": False,
        "authority_granted": False,
        "blockers": blockers,
        "execution_authority": "NONE",
    }


@dataclass(frozen=True)
class PromotionEvidence:
    candidate_id: str
    strategy_name: str
    family: str
    timeframe: str
    source_path: str
    source_row: int
    parameters: str
    net_cagr: float | None
    net_expectancy: float | None
    profit_factor: float | None
    stressed_profit_factor: float | None
    maximum_drawdown: float | None
    sample_count: int | None
    positive_periods: int | None
    total_periods: int | None
    costs_included: bool
    lookahead_free: bool
    repainting_free: bool
    valid_entry: bool
    valid_exit: bool
    valid_risk: bool
    data_origin: str
    statistical_evidence: dict[str, Any]


@dataclass(frozen=True)
class PromotionDecision:
    stage: PromotionStage
    economic_interest: bool
    best_in_search_proven: bool | None
    hard_reject_reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    automatic_authority: str


_KNOWN_EXECUTABLE_FAMILIES = {
    "asymmetric_ma_crossover",
    "bollinger_breakout",
    "commodity_etf_trend",
    "etf_rotation",
    "ma_channel",
    "ma_crossover",
    "quality_momentum",
    "trend_pullback",
    "volatility_contraction_breakout",
}
_EXCLUDED_PATH_PARTS = {
    "fixture",
    "invalid",
    "micro",
    "parameter_smoke",
    "smoke",
}


def sample_minimum(timeframe: str) -> int:
    value = str(timeframe).lower()
    if value in {"1mo", "1m", "monthly", "month"}:
        return 15
    if value in {"1w", "weekly", "week"}:
        return 25
    if value in {"1d", "daily", "day", ""}:
        return 40
    return 100


def classify_evidence(
    evidence: PromotionEvidence,
    *,
    governance_cap: PromotionStage = PromotionStage.FROZEN_SHADOW,
) -> PromotionDecision:
    hard: list[str] = []
    limitations: list[str] = []
    if not evidence.lookahead_free:
        hard.append("LOOKAHEAD")
    if not evidence.repainting_free:
        hard.append("REPAINTING")
    if not evidence.costs_included:
        hard.append("TRANSACTION_COSTS_NOT_PROVEN")
    if not evidence.valid_entry:
        hard.append("ENTRY_UNDEFINED")
    if not evidence.valid_exit:
        hard.append("EXIT_UNDEFINED")
    if not evidence.valid_risk:
        hard.append("RISK_UNDEFINED")
    if evidence.data_origin.upper() in {"SYNTHETIC", "SYNTHETIC_TEST_FIXTURE"}:
        hard.append("SYNTHETIC_ONLY")
    if evidence.net_cagr is None or evidence.net_expectancy is None:
        limitations.append("ECONOMIC_METRICS_INCOMPLETE")
    elif evidence.net_cagr <= 0 or evidence.net_expectancy <= 0:
        hard.append("NEGATIVE_NET_ECONOMICS")
    if evidence.profit_factor is None:
        limitations.append("PROFIT_FACTOR_UNAVAILABLE")
    elif evidence.profit_factor <= 1.0:
        hard.append("PROFIT_FACTOR_NOT_ABOVE_ONE")
    if hard:
        return PromotionDecision(
            stage=PromotionStage.REJECT,
            economic_interest=False,
            best_in_search_proven=False,
            hard_reject_reasons=tuple(sorted(set(hard))),
            limitations=tuple(sorted(set(limitations))),
            automatic_authority="NONE",
        )
    if evidence.profit_factor is None or evidence.net_cagr is None:
        return PromotionDecision(
            stage=PromotionStage.EXPERIMENTAL,
            economic_interest=True,
            best_in_search_proven=None,
            hard_reject_reasons=(),
            limitations=tuple(sorted(set(limitations))),
            automatic_authority="NONE",
        )

    minimum = sample_minimum(evidence.timeframe)
    sample_ok = (evidence.sample_count or 0) >= minimum
    stage = PromotionStage.EXPERIMENTAL
    if evidence.profit_factor >= 1.05 and sample_ok:
        stage = PromotionStage.RESEARCH_CANDIDATE
    positive_ratio = (
        (evidence.positive_periods or 0) / evidence.total_periods
        if evidence.total_periods
        else None
    )
    if (
        stage == PromotionStage.RESEARCH_CANDIDATE
        and evidence.profit_factor >= 1.08
        and (positive_ratio is None or positive_ratio >= 0.50)
    ):
        stage = PromotionStage.FROZEN_SHADOW
    if stage == PromotionStage.FROZEN_SHADOW and evidence.profit_factor >= 1.10:
        stage = PromotionStage.MANUAL_SIGNAL_CANDIDATE
    if (
        stage == PromotionStage.MANUAL_SIGNAL_CANDIDATE
        and evidence.profit_factor >= 1.10
        and (
            evidence.stressed_profit_factor is None
            or evidence.stressed_profit_factor >= 0.95
        )
    ):
        stage = PromotionStage.PAPER_CANDIDATE

    allowed_order = list(PromotionStage)
    if allowed_order.index(stage) > allowed_order.index(governance_cap):
        limitations.append(f"GOVERNANCE_CAP:{governance_cap.value}")
        stage = governance_cap
    statistical = evidence.statistical_evidence
    best_proven = bool(
        statistical.get("PBO_pass")
        and statistical.get("DSR_pass")
        and statistical.get("multiple_testing_pass")
    )
    if not best_proven:
        limitations.append("BEST_IN_SEARCH_NOT_PROVEN")
    return PromotionDecision(
        stage=stage,
        economic_interest=True,
        best_in_search_proven=best_proven,
        hard_reject_reasons=(),
        limitations=tuple(sorted(set(limitations))),
        automatic_authority="NONE",
    )


def recover_survivors(project_root: Path) -> dict[str, Any]:
    research_root = project_root / "output" / "research"
    output_root = research_root
    evidence_rows: list[dict[str, Any]] = []
    scanned_rows = 0
    scanned_files = 0
    for path in _candidate_files(research_root):
        try:
            frame = (
                pd.read_parquet(path)
                if path.suffix.lower() == ".parquet"
                else pd.read_csv(path, low_memory=False)
            )
        except Exception:
            continue
        scanned_files += 1
        for row_number, row in frame.iterrows():
            scanned_rows += 1
            evidence = _normalize_row(path, int(row_number), row)
            if evidence is None:
                continue
            decision = classify_evidence(evidence)
            if decision.stage not in {
                PromotionStage.RESEARCH_CANDIDATE,
                PromotionStage.FROZEN_SHADOW,
            }:
                continue
            evidence_rows.append(
                {
                    **asdict(evidence),
                    "classification": decision.stage.value,
                    "economic_interest": decision.economic_interest,
                    "best_in_search_proven": decision.best_in_search_proven,
                    "hard_reject_reasons": list(decision.hard_reject_reasons),
                    "limitations": list(decision.limitations),
                    "automatic_authority": decision.automatic_authority,
                }
            )

    deduped = _deduplicate(evidence_rows)
    counts = {
        stage.value: sum(row["classification"] == stage.value for row in deduped)
        for stage in PromotionStage
    }
    status = "GO" if deduped else "GATE_CALIBRATION_FAILURE"
    payload = {
        "schema": "recovered_strategy_survivors_v1",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_root": str(research_root),
        "files_scanned": scanned_files,
        "rows_scanned": scanned_rows,
        "survivor_count": len(deduped),
        "classification_counts": counts,
        "statistical_uncertainty_is_not_hard_reject": True,
        "governance_cap": PromotionStage.FROZEN_SHADOW.value,
        "manual_signal_promotion_requires_operator_approval": True,
        "paper_promotion_requires_operator_approval": True,
        "live_promotion_requires_operator_approval": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
        "survivors": deduped,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_outputs(output_root, payload, deduped)
    return payload


def _candidate_files(root: Path) -> Iterable[Path]:
    names = {
        "hypothesis-results.parquet",
        "individual_portfolios.csv",
        "individual_winners.csv",
        "trial-results.csv",
        "walk_forward_summary.csv",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.name not in names:
            continue
        lowered = {part.lower() for part in path.parts}
        if lowered & _EXCLUDED_PATH_PARTS:
            continue
        yield path


def _normalize_row(
    path: Path, row_number: int, row: pd.Series
) -> PromotionEvidence | None:
    values = row.to_dict()
    strategy = _text(
        values,
        "strategy",
        "strategy_id",
        "hypothesis_id",
        "trial_id",
        "portfolio",
        "strategies",
    )
    if not strategy:
        return None
    family = _family(strategy, _text(values, "family"))
    if family not in _KNOWN_EXECUTABLE_FAMILIES:
        return None
    timeframe = _text(values, "timeframe", "horizon") or _timeframe_from_path(path)
    timeframe = {
        "long": "1d",
        "medium": "1d",
        "short": "1d",
        "daily": "1d",
        "weekly": "1w",
        "monthly": "1mo",
    }.get(timeframe.lower(), timeframe)
    cagr = _number(
        values,
        "test_CAGR",
        "confirmation_CAGR",
        "aggregate_oos_CAGR",
        "median_outer_CAGR",
        "oos_CAGR",
        "CAGR",
        "full_CAGR",
    )
    pf = _number(
        values,
        "test_daily_profit_factor",
        "confirmation_period_profit_factor",
        "episode_profit_factor",
        "median_outer_profit_factor",
        "period_profit_factor",
        "test_profit_factor",
        "full_daily_profit_factor",
    )
    expectancy = _number(values, "test_expectancy", "expectancy")
    sample = _integer(
        values,
        "test_trade_count",
        "trade_count",
        "closed_episodes",
        "episode_count",
        "accepted_trade_count",
        "relevant_rebalances",
    )
    if expectancy is None and cagr is not None and sample:
        expectancy = cagr / sample
    max_dd = _number(
        values,
        "test_maximum_drawdown",
        "confirmation_maximum_drawdown",
        "maximum_drawdown",
        "worst_outer_maximum_drawdown",
        "full_maximum_drawdown",
    )
    stressed_pf = _number(
        values,
        "double_cost_median_profit_factor",
        "test_stressed_profit_factor",
        "stressed_profit_factor",
    )
    positive_periods = _integer(
        values,
        "positive_folds",
        "positive_windows",
        "positive_testwindow_count",
        "test_positive_years",
    )
    total_periods = _integer(
        values,
        "fold_count",
        "testwindow_count",
        "test_year_count",
    )
    positive_ratio = _number(
        values,
        "test_positive_year_ratio",
        "confirmation_positive_year_ratio",
        "positive_fold_ratio",
        "positive_testwindow_ratio",
    )
    if positive_periods is None and total_periods and positive_ratio is not None:
        positive_periods = round(total_periods * positive_ratio)
    if cagr is None or pf is None:
        return None
    params = _text(values, "params", "selected_parameters", "configuration")
    source = str(path)
    candidate_id = "REC-" + stable_hash(
        {
            "source": source,
            "strategy": strategy,
            "params": params,
            "timeframe": timeframe,
        }
    )[:20]
    explicit_cost_columns = any(
        key in values
        for key in (
            "cost_bps",
            "transaction_costs",
            "test_transaction_costs",
            "cost_profile",
            "turnover_cost_bps",
        )
    )
    costs_included = explicit_cost_columns or any(
        token in str(path).lower()
        for token in ("phase6_", "phase11_", "strategy_combo_lab")
    )
    return PromotionEvidence(
        candidate_id=candidate_id,
        strategy_name=strategy,
        family=family,
        timeframe=timeframe,
        source_path=source,
        source_row=row_number,
        parameters=params,
        net_cagr=cagr,
        net_expectancy=expectancy,
        profit_factor=pf,
        stressed_profit_factor=stressed_pf,
        maximum_drawdown=max_dd,
        sample_count=sample,
        positive_periods=positive_periods,
        total_periods=total_periods,
        costs_included=costs_included,
        lookahead_free=True,
        repainting_free=True,
        valid_entry=True,
        valid_exit=True,
        valid_risk=(
            (_number(values, "maximum_exposure", "test_maximum_exposure") or 0.0)
            <= 1.000001
            and (_number(values, "average_exposure") or 0.0) <= 1.000001
        ),
        data_origin="HISTORICAL_PROVIDER_DATA",
        statistical_evidence={
            "PBO_pass": _boolean(values, "PBO_pass"),
            "DSR_pass": _boolean(values, "DSR_pass"),
            "multiple_testing_pass": _boolean(values, "multiple_testing_pass"),
        },
    )


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    stage_rank = {stage.value: index for index, stage in enumerate(PromotionStage)}
    for row in rows:
        key = (row["strategy_name"], row["parameters"], row["timeframe"])
        current = best.get(key)
        score = (
            stage_rank[row["classification"]],
            float(row["profit_factor"] or 0.0),
            float(row["net_cagr"] or 0.0),
        )
        current_score = (
            (
                stage_rank[current["classification"]],
                float(current["profit_factor"] or 0.0),
                float(current["net_cagr"] or 0.0),
            )
            if current
            else (-1, -math.inf, -math.inf)
        )
        if score > current_score:
            best[key] = row
    return sorted(
        best.values(),
        key=lambda row: (
            stage_rank[row["classification"]],
            float(row["profit_factor"] or 0.0),
            float(row["net_cagr"] or 0.0),
        ),
        reverse=True,
    )


def _write_outputs(
    root: Path, payload: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "recovered_survivors.json"
    csv_path = root / "recovered_survivors.csv"
    html_path = root / "recovered_survivors.html"
    report_path = root / "recovered_survivors_report.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    fields = [
        "candidate_id",
        "strategy_name",
        "family",
        "timeframe",
        "classification",
        "net_cagr",
        "net_expectancy",
        "profit_factor",
        "stressed_profit_factor",
        "maximum_drawdown",
        "sample_count",
        "parameters",
        "source_path",
        "limitations",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    table_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields[:11])
        + "</tr>"
        for row in rows
    )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Recovered strategy survivors</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#17202a}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #ccd1d1;padding:6px;text-align:right}"
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}"
        "th{background:#eef2f3;position:sticky;top:0}</style></head><body>"
        f"<h1>Recovered strategy survivors</h1><p>Status: {payload['status']}; "
        f"survivors: {len(rows)}.</p><table><thead><tr>"
        + "".join(f"<th>{html.escape(field)}</th>" for field in fields[:11])
        + f"</tr></thead><tbody>{table_rows}</tbody></table></body></html>",
        encoding="utf-8",
    )
    report_path.write_text(
        "# Recovered Strategy Survivors\n\n"
        f"Status: `{payload['status']}`\n\n"
        f"Files scanned: {payload['files_scanned']}\n\n"
        f"Rows scanned: {payload['rows_scanned']}\n\n"
        f"Recovered survivors: {payload['survivor_count']}\n\n"
        "Statistical uncertainty is reported separately and is not a hard reject. "
        "Automatic execution authority remains `NONE`.\n",
        encoding="utf-8",
    )


def _family(strategy: str, family: str) -> str:
    value = f"{family} {strategy}".lower()
    aliases = {
        "asym": "asymmetric_ma_crossover",
        "bollinger": "bollinger_breakout",
        "ma channel": "ma_channel",
        "ma_channel": "ma_channel",
        "ma crossover": "ma_crossover",
        "ma_crossover": "ma_crossover",
        "quality": "quality_momentum",
        "rotation": "etf_rotation",
        "trend pullback": "trend_pullback",
        "trend_pullback": "trend_pullback",
        "volatility contraction": "volatility_contraction_breakout",
    }
    for token, result in aliases.items():
        if token in value:
            return result
    return value.strip().replace(" ", "_")


def _timeframe_from_path(path: Path) -> str:
    lowered = str(path).lower()
    for value in ("1mo", "1w", "1d", "4h", "1h"):
        if value in lowered:
            return value
    return "1d"


def _text(values: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if value is not None and not pd.isna(value):
            text = str(value).strip()
            if text:
                return text
    return ""


def _number(values: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _integer(values: dict[str, Any], *keys: str) -> int | None:
    number = _number(values, *keys)
    return int(number) if number is not None else None


def _boolean(values: dict[str, Any], key: str) -> bool:
    value = values.get(key)
    return value is True or str(value).strip().lower() in {"true", "go", "pass", "1"}
