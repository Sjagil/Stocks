from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.research.autopilot.contracts import stable_hash
from stocks.research.autopilot.generator import (
    generate_strategies,
)
from stocks.research.autopilot.statistics import (
    probability_of_backtest_overfitting,
)
from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger
from stocks.research.phase11_6 import (
    PARAMETERS,
    TARGET_HISTORY_START,
    TREND_STRATEGIES,
    _aggregate_prices,
    _metrics,
    load_pit_prices,
    nested_walk_forward_folds,
    strategy_signal,
)


AUTHORITY = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_SHADOW_GO": False,
    "STRATEGY_AUTHORITY": "NONE",
    "EXECUTION_AUTHORITY": "NONE",
    "PAPER_STRATEGY_AUTHORITY": "NONE",
    "LIVE_STRATEGY_AUTHORITY": "NONE",
    "BROKER_CALLS": 0,
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "broker_calls": 0,
}
COST_STRESS_BPS = (5.0, 10.0, 20.0, 30.0, 50.0)
TIMEFRAMES = ("1d", "1w", "1mo")


def run_macro_pair_validation(
    project_root: Path,
    *,
    max_identities: int = 500,
) -> dict[str, Any]:
    output = project_root / "output" / "research" / "macro_pairs"
    output.mkdir(parents=True, exist_ok=True)
    base = load_pit_prices(project_root, max_identities)
    if base.empty:
        raise RuntimeError("MACRO_PAIR_DATA_UNAVAILABLE_BLOCKED")
    macro_gate, gate_audit = _macro_risk_on_gate(project_root)
    if macro_gate.empty:
        raise RuntimeError("MACRO_HISTORY_UNAVAILABLE_BLOCKED")
    cache: dict[tuple[str, str, str, float, bool], pd.Series] = {}
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    excess_returns: dict[tuple[str, str], list[pd.Series]] = {}
    validation_matrix: dict[str, dict[str, float]] = {}
    outer_matrix: dict[str, dict[str, float]] = {}
    for timeframe in TIMEFRAMES:
        data = _aggregate_prices(base, timeframe)
        start = max(TARGET_HISTORY_START, data["date"].min(), macro_gate.index.min())
        end = min(data["date"].max(), macro_gate.index.max())
        folds = nested_walk_forward_folds(start, end, timeframe)
        fold_rows.extend(folds.to_dict("records"))
        for strategy in TREND_STRATEGIES:
            for fold in folds.to_dict("records"):
                scored = []
                for parameters in PARAMETERS[strategy]:
                    baseline = _cached_returns(
                        cache,
                        data,
                        timeframe,
                        strategy,
                        parameters,
                        10.0,
                        macro_gate,
                        use_macro=False,
                    )
                    validation = _slice(
                        baseline,
                        fold["validation_start"],
                        fold["validation_end"],
                    )
                    metrics = _metrics(validation)
                    scored.append(
                        (
                            metrics["profit_factor"],
                            metrics["CAGR"],
                            _canonical(parameters),
                            parameters,
                            metrics,
                        )
                    )
                eligible = [
                    row
                    for row in scored
                    if math.isfinite(row[0]) and math.isfinite(row[1])
                ]
                if not eligible:
                    continue
                selected = max(
                    eligible,
                    key=lambda row: (row[0], row[1], row[2]),
                )
                parameter_hash = stable_hash(
                    {
                        "strategy": strategy,
                        "timeframe": timeframe,
                        "parameters": selected[3],
                    }
                )
                selection_rows.append(
                    {
                        "fold_id": fold["fold_id"],
                        "strategy": strategy,
                        "timeframe": timeframe,
                        "selected_parameters": selected[2],
                        "selected_parameter_hash": parameter_hash,
                        "selection_source": (
                            "NO_MACRO_INNER_VALIDATION_ONLY"
                        ),
                        "macro_variant_in_parameter_selection": False,
                        "validation_profit_factor": selected[0],
                        "validation_CAGR": selected[1],
                    }
                )
                for cost_bps in COST_STRESS_BPS:
                    baseline = _cached_returns(
                        cache,
                        data,
                        timeframe,
                        strategy,
                        selected[3],
                        cost_bps,
                        macro_gate,
                        use_macro=False,
                    )
                    macro = _cached_returns(
                        cache,
                        data,
                        timeframe,
                        strategy,
                        selected[3],
                        cost_bps,
                        macro_gate,
                        use_macro=True,
                    )
                    baseline_outer = _slice(
                        baseline,
                        fold["outer_test_start"],
                        fold["outer_test_end"],
                    )
                    macro_outer = _slice(
                        macro,
                        fold["outer_test_start"],
                        fold["outer_test_end"],
                    )
                    baseline_metrics = _metrics(baseline_outer)
                    macro_metrics = _metrics(macro_outer)
                    pair_id = f"{strategy}__{timeframe}"
                    outer_rows.append(
                        {
                            **fold,
                            "pair_id": pair_id,
                            "strategy": strategy,
                            "timeframe": timeframe,
                            "macro_treatment": "PIT_MACRO_RISK_ON_GATE_V1",
                            "selected_parameters": selected[2],
                            "selected_parameter_hash": parameter_hash,
                            "cost_bps": cost_bps,
                            "baseline_CAGR": baseline_metrics["CAGR"],
                            "macro_CAGR": macro_metrics["CAGR"],
                            "delta_CAGR": (
                                macro_metrics["CAGR"]
                                - baseline_metrics["CAGR"]
                            ),
                            "baseline_profit_factor": baseline_metrics[
                                "profit_factor"
                            ],
                            "macro_profit_factor": macro_metrics[
                                "profit_factor"
                            ],
                            "delta_profit_factor": (
                                macro_metrics["profit_factor"]
                                - baseline_metrics["profit_factor"]
                            ),
                            "baseline_Sharpe": baseline_metrics["Sharpe"],
                            "macro_Sharpe": macro_metrics["Sharpe"],
                            "delta_Sharpe": (
                                macro_metrics["Sharpe"]
                                - baseline_metrics["Sharpe"]
                            ),
                            "baseline_maximum_drawdown": baseline_metrics[
                                "maximum_drawdown"
                            ],
                            "macro_maximum_drawdown": macro_metrics[
                                "maximum_drawdown"
                            ],
                            "drawdown_improvement": (
                                macro_metrics["maximum_drawdown"]
                                - baseline_metrics["maximum_drawdown"]
                            ),
                            "identical_parameters": True,
                            "identical_universe": True,
                            "identical_cost_model": True,
                            "macro_used_for_selection": False,
                        }
                    )
                    if cost_bps == 10.0:
                        excess_returns.setdefault(
                            (strategy, timeframe),
                            [],
                        ).append(macro_outer - baseline_outer)
                        column_base = f"{pair_id}__BASELINE"
                        column_macro = f"{pair_id}__MACRO"
                        validation_matrix.setdefault(
                            fold["fold_id"],
                            {},
                        )[column_base] = float(selected[4]["Sharpe"])
                        macro_validation = _metrics(
                            _slice(
                                _cached_returns(
                                    cache,
                                    data,
                                    timeframe,
                                    strategy,
                                    selected[3],
                                    10.0,
                                    macro_gate,
                                    use_macro=True,
                                ),
                                fold["validation_start"],
                                fold["validation_end"],
                            )
                        )
                        validation_matrix[fold["fold_id"]][column_macro] = float(
                            macro_validation["Sharpe"]
                        )
                        outer_matrix.setdefault(
                            fold["fold_id"],
                            {},
                        )[column_base] = float(baseline_metrics["Sharpe"])
                        outer_matrix[fold["fold_id"]][column_macro] = float(
                            macro_metrics["Sharpe"]
                        )
    outer = pd.DataFrame(outer_rows)
    selection = pd.DataFrame(selection_rows)
    folds_frame = pd.DataFrame(fold_rows).drop_duplicates("fold_id")
    summaries = _pair_summaries(outer, excess_returns)
    registry_pairs = _registry_pair_inventory(project_root)
    pbo = probability_of_backtest_overfitting(
        pd.DataFrame.from_dict(validation_matrix, orient="index"),
        pd.DataFrame.from_dict(outer_matrix, orient="index"),
    )
    pbo["global_trial_count"] = int(
        len(TREND_STRATEGIES)
        * len(TIMEFRAMES)
        * 2
        * len(COST_STRESS_BPS)
    )
    retained = [
        row for row in summaries if row["macro_variant_status"] == "RETAINED"
    ]
    decision = {
        "schema": "macro_pair_financial_decision_v1",
        "status": "GO",
        "decision": (
            "MACRO_VARIANTS_RETAINED_FOR_FURTHER_RESEARCH"
            if retained
            else "NO_MACRO_VARIANT_DEMONSTRATED_OOS_VALUE_ADD"
        ),
        "retained_pair_ids": [row["pair_id"] for row in retained],
        "retained_count": len(retained),
        "financial_finalist_go": False,
        "financial_finalist_reason": (
            "MACRO_PAIR_VALIDATION_IS_INCREMENTAL_EVIDENCE_NOT_FORWARD_HOLDOUT"
        ),
        **AUTHORITY,
    }
    _write_frame(output / "fold-registry.parquet", folds_frame)
    _write_frame(output / "parameter-selection.parquet", selection)
    _write_frame(output / "paired-outer-results.parquet", outer)
    _write_frame(output / "pair-summary.parquet", pd.DataFrame(summaries))
    _write_json(output / "registry-pair-inventory.json", registry_pairs)
    _write_json(output / "multiple-testing.json", pbo)
    _write_json(output / "retained-macro-variants.json", {"rows": retained})
    _write_json(output / "decision.json", decision)
    status = {
        "schema": "macro_pair_validation_status_v1",
        "status": "GO" if not outer.empty else "BLOCKED",
        "identity_count": int(base["security_id"].nunique()),
        "financial_pair_count": len(summaries),
        "registry_pair_count": len(registry_pairs["pairs"]),
        "outer_result_count": len(outer),
        "cost_stress_bps": list(COST_STRESS_BPS),
        "macro_gate_audit": gate_audit,
        "retained_macro_variant_count": len(retained),
        "multiple_testing": pbo,
        "financial_decision": decision["decision"],
        **AUTHORITY,
    }
    _write_json(output / "status.json", status)
    return status


def _macro_risk_on_gate(
    project_root: Path,
) -> tuple[pd.Series, dict[str, Any]]:
    path = project_root / "output" / "macro" / "history.json"
    if not path.exists():
        return pd.Series(dtype=bool), {"status": "UNAVAILABLE"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: dict[pd.Timestamp, bool] = {}
    for row in payload.get("history", []):
        timestamp = pd.Timestamp(row["as_of"]).tz_localize(None).normalize()
        records[timestamp] = (
            row.get("regime", {}).get("market_regime") == "RISK_ON"
        )
    if not records:
        return pd.Series(dtype=bool), {"status": "UNAVAILABLE"}
    series = pd.Series(records, dtype=bool).sort_index()
    return series, {
        "status": "GO",
        "component": "macro_risk_on",
        "source": str(path.relative_to(project_root)).replace("\\", "/"),
        "first_as_of": series.index.min().date().isoformat(),
        "last_as_of": series.index.max().date().isoformat(),
        "observation_count": len(series),
        "positive_ratio": float(series.mean()),
        "point_in_time": True,
        "future_returns_used": False,
    }


def _cached_returns(
    cache: dict[tuple[str, str, str, float, bool], pd.Series],
    frame: pd.DataFrame,
    timeframe: str,
    strategy: str,
    parameters: Mapping[str, float],
    cost_bps: float,
    macro_gate: pd.Series,
    *,
    use_macro: bool,
) -> pd.Series:
    key = (
        timeframe,
        strategy,
        _canonical(parameters),
        cost_bps,
        use_macro,
    )
    if key not in cache:
        cache[key] = _portfolio_returns(
            frame,
            strategy,
            parameters,
            cost_bps,
            macro_gate=macro_gate if use_macro else None,
        )
    return cache[key]


def _portfolio_returns(
    frame: pd.DataFrame,
    strategy: str,
    parameters: Mapping[str, float],
    cost_bps: float,
    *,
    macro_gate: pd.Series | None,
) -> pd.Series:
    pieces = []
    for _, group in frame.groupby("security_id", sort=False):
        work = group.sort_values("date").copy()
        decision_signal = strategy_signal(
            work,
            strategy,
            parameters,
        ).fillna(False)
        if macro_gate is not None:
            decision_dates = pd.DatetimeIndex(work["date"])
            gate = (
                macro_gate.reindex(
                    macro_gate.index.union(decision_dates)
                )
                .astype("boolean")
                .sort_index()
                .ffill()
                .reindex(decision_dates)
                .fillna(False)
                .astype(bool)
            )
            decision_signal &= pd.Series(
                gate.to_numpy(),
                index=decision_signal.index,
            )
        signal = decision_signal.shift(1, fill_value=False).astype(float)
        returns = work["close"].astype(float).pct_change().fillna(0.0)
        turnover = signal.diff().abs().fillna(signal.abs())
        work["strategy_return"] = (
            signal * returns - turnover * cost_bps / 10_000.0
        )
        pieces.append(work[["date", "strategy_return"]])
    details = pd.concat(pieces, ignore_index=True)
    return details.groupby("date")["strategy_return"].mean().sort_index()


def _pair_summaries(
    outer: pd.DataFrame,
    excess_returns: dict[tuple[str, str], list[pd.Series]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if outer.empty:
        return rows
    trial_count = int(
        len(TREND_STRATEGIES) * len(TIMEFRAMES) * 2 * len(COST_STRESS_BPS)
    )
    for (strategy, timeframe), group in outer.groupby(
        ["strategy", "timeframe"]
    ):
        normal = group.loc[group["cost_bps"].eq(10.0)]
        stress = group.loc[group["cost_bps"].eq(20.0)]
        excess = pd.concat(
            excess_returns.get((strategy, timeframe), [])
        ).sort_index()
        dsr = _deflated_sharpe_probability(excess, trial_count)
        win_ratio = float(normal["delta_CAGR"].gt(0).mean())
        retained = (
            len(normal) >= 5
            and float(normal["delta_CAGR"].median()) > 0
            and float(normal["delta_Sharpe"].median()) > 0
            and float(normal["macro_profit_factor"].median())
            > float(normal["baseline_profit_factor"].median())
            and float(stress["macro_profit_factor"].median()) > 1.0
            and win_ratio >= 0.60
            and dsr is not None
            and dsr >= 0.95
        )
        rows.append(
            {
                "pair_id": f"{strategy}__{timeframe}",
                "strategy": strategy,
                "timeframe": timeframe,
                "fold_count": len(normal),
                "positive_oos_improvement_ratio": win_ratio,
                "median_delta_CAGR": float(normal["delta_CAGR"].median()),
                "median_delta_Sharpe": float(normal["delta_Sharpe"].median()),
                "median_delta_profit_factor": float(
                    normal["delta_profit_factor"].median()
                ),
                "baseline_median_profit_factor": float(
                    normal["baseline_profit_factor"].median()
                ),
                "macro_median_profit_factor": float(
                    normal["macro_profit_factor"].median()
                ),
                "macro_20bps_median_profit_factor": float(
                    stress["macro_profit_factor"].median()
                ),
                "median_drawdown_improvement": float(
                    normal["drawdown_improvement"].median()
                ),
                "excess_deflated_sharpe_probability": dsr,
                "global_trial_count": trial_count,
                "macro_variant_status": (
                    "RETAINED" if retained else "REJECTED_NO_ROBUST_OOS_UPLIFT"
                ),
            }
        )
    return rows


def _registry_pair_inventory(
    project_root: Path | None = None,
) -> dict[str, Any]:
    baselines: dict[str, dict[str, Any]] = {
        item.strategy_id: {
            "strategy_id": item.strategy_id,
            "strategy_hash": item.strategy_hash,
            "parameters": item.parameters,
            "family": item.family,
        }
        for item in generate_strategies(budget=100)
    }
    if project_root is not None:
        layout = AutopilotLayout(project_root)
        if layout.database.exists():
            ledger = ResearchLedger(layout)
            try:
                for row in ledger.strategies():
                    baselines[str(row["strategy_id"])] = {
                        "strategy_id": str(row["strategy_id"]),
                        "strategy_hash": str(row["strategy_hash"]),
                        "parameters": dict(row["parameters"]),
                        "family": str(row["family"]),
                    }
            finally:
                ledger.close()
    pairs = []
    for baseline in sorted(
        baselines.values(),
        key=lambda item: item["strategy_id"],
    ):
        macro_hash = stable_hash(
            {
                "baseline_strategy_hash": baseline["strategy_hash"],
                "parameters": baseline["parameters"],
                "macro_filter": "macro_risk_on",
            }
        )
        pairs.append(
            {
                "baseline_strategy_id": baseline["strategy_id"],
                "macro_strategy_id": f"MACRO-PAIR-{macro_hash[:24]}",
                "baseline_strategy_hash": baseline["strategy_hash"],
                "macro_strategy_hash": macro_hash,
                "family": baseline["family"],
                "identical_non_macro_parameters": True,
                "macro_filter": "macro_risk_on",
                "financial_engine_mapping": (
                    "PHASE11_6_REGRESSION_SHORTLIST_ONLY"
                ),
                "pair_status": "STRICT_PAIR_REGISTERED",
                "automatic_promotion": False,
            }
        )
    return {
        "schema": "macro_registry_pair_inventory_v1",
        "status": "GO",
        "pairs": pairs,
        "pair_count": len(pairs),
        "financial_evidence_scope": (
            "ONLY_PHASE11_6_REGRESSION_SHORTLIST_HAS_COMPLETE_PIT_PRICE_MAPPING"
        ),
        **AUTHORITY,
    }


def _deflated_sharpe_probability(
    returns: pd.Series,
    trial_count: int,
) -> float | None:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(values) < 126 or values.std(ddof=1) <= 0:
        return None
    periods = max(
        1.0,
        len(values)
        / max((values.index.max() - values.index.min()).days / 365.2425, 1 / 365),
    )
    sharpe = float(values.mean() / values.std(ddof=1) * np.sqrt(periods))
    probability = min(
        0.999999,
        max(0.500001, 1.0 - 1.0 / max(2, trial_count)),
    )
    threshold = max(0.0, NormalDist().inv_cdf(probability))
    standard_error = math.sqrt(
        max(
            1e-12,
            (1.0 + 0.5 * sharpe * sharpe) / max(1, len(values) - 1),
        )
    )
    return float(NormalDist().cdf((sharpe - threshold) / standard_error))


def _slice(
    values: pd.Series,
    start: Any,
    end: Any,
) -> pd.Series:
    return values.loc[
        (values.index >= pd.Timestamp(start))
        & (values.index <= pd.Timestamp(end))
    ]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: Any) -> None:
    enriched = (
        {
            **payload,
            "generated_at": datetime.now(UTC).isoformat(),
            "content_hash": stable_hash(payload),
        }
        if isinstance(payload, dict)
        else payload
    )
    path.write_text(
        json.dumps(enriched, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)
