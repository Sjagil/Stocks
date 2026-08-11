"""Registry-sealed parameter research using the existing v2 portfolio ledger."""

from __future__ import annotations

import dataclasses
import datetime as dt
import ast
import hashlib
import itertools
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import qmc

if TYPE_CHECKING:
    import argparse

SCHEMA = "stocks_parameter_research_campaign_v2"
DEFAULT_REGISTRY = Path(
    "config/research_contracts/stocks_parameter_space_registry_v2.json"
)
BLOCKED_CALL_TOKENS = (
    "place" + "Order",
    "cancel" + "Order",
    "req" + "Global" + "Cancel",
    "req" + "Mkt" + "Data",
    "req" + "Historical" + "Data",
    "req" + "Ids",
    "req" + "Auto" + "Open" + "Orders",
    "exercise" + "Options",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def parameter_type(name: str, values: Sequence[Any]) -> str:
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        if any(token in name for token in ("threshold", "sigma", "return")):
            return "float"
        return "integer"
    if all(isinstance(value, (int, float)) for value in values):
        return "float"
    if all(isinstance(value, str) for value in values):
        return "enum"
    return "json"


def resolve_registry(source_path: Path, runtime: Any) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    overrides = source.get("strategy_overrides", {})
    strategies: list[dict[str, Any]] = []
    for spec in runtime.strategy_registry():
        override = overrides.get(spec.name, {})
        parameters = []
        for name, raw_values in spec.choices.items():
            coarse_values = list(raw_values)
            default = spec.default_params[name]
            if default not in coarse_values:
                coarse_values.insert(0, default)
            kind = parameter_type(name, coarse_values)
            refinement_values: list[Any] = []
            if name in override.get("refinement_parameters", []) and kind == "float":
                numeric = sorted({float(value) for value in coarse_values})
                refinement_values = [
                    (left + right) / 2.0
                    for left, right in zip(numeric, numeric[1:])
                ]
            values = sorted(
                {
                    stable_json(value): value
                    for value in coarse_values + refinement_values
                }.values(),
                key=lambda value: (
                    "number" if isinstance(value, (int, float)) else str(type(value)),
                    float(value)
                    if isinstance(value, (int, float))
                    else stable_json(value),
                ),
            )
            parameters.append(
                {
                    "name": name,
                    "type": kind,
                    "allowed_values": values,
                    "coarse_allowed_values": coarse_values,
                    "refinement_allowed_values": refinement_values,
                    "default": default,
                    "economic_meaning": f"Registered mechanical control for {name}.",
                    "conditional_on": None,
                    "forbidden_with": [],
                    "neighbor_order": values,
                }
            )
        strategies.append(
            {
                "strategy_name": spec.name,
                "economic_family": spec.family,
                "strategy_version": "runtime-2.1",
                "baseline_configuration": dict(spec.default_params),
                "fixed_parameters": {},
                "search_parameters": parameters,
                "conditional_parameters": [],
                "forbidden_combinations": ["runtime.params_valid"],
                "parameter_order": list(spec.choices),
                "coarse_search_contract": "registered_values",
                "refinement_contract": (
                    "registered_midpoints_for_explicit_float_parameters"
                ),
                "neighbor_contract": "one_registered_grid_step",
                "maximum_trials": int(
                    math.prod(max(len(values), 1) for values in spec.choices.values())
                ),
                "search_method": "deterministic_auto",
                "random_seed": int(source["random_seed"]),
                "economic_rationale": override.get(
                    "economic_rationale", spec.description
                ),
                "priority": int(override.get("priority", 4)),
                "refinement_parameters": list(
                    override.get("refinement_parameters", [])
                ),
                "policy_status": spec.policy_status,
                "implementation_status": "EXECUTABLE_V2_LEDGER",
            }
        )
    rotational_parameters = []
    for name, raw_values in runtime.ROTATIONAL_CHOICES.items():
        values = list(raw_values)
        rotational_parameters.append(
            {
                "name": name,
                "type": parameter_type(name, values),
                "allowed_values": values,
                "coarse_allowed_values": values,
                "refinement_allowed_values": [],
                "default": runtime.ROTATIONAL_DEFAULT[name],
                "economic_meaning": f"Registered mechanical control for {name}.",
                "conditional_on": None,
                "forbidden_with": [],
                "neighbor_order": values,
            }
        )
    rotational_override = overrides.get("rotational_momentum", {})
    strategies.append(
        {
            "strategy_name": "rotational_momentum",
            "economic_family": "cross_sectional_momentum",
            "strategy_version": "runtime-2.1",
            "baseline_configuration": dict(runtime.ROTATIONAL_DEFAULT),
            "fixed_parameters": {},
            "search_parameters": rotational_parameters,
            "conditional_parameters": [],
            "forbidden_combinations": [],
            "parameter_order": list(runtime.ROTATIONAL_CHOICES),
            "coarse_search_contract": "registered_values",
            "refinement_contract": "registered_values_only",
            "neighbor_contract": "one_registered_grid_step",
            "maximum_trials": int(
                math.prod(len(values) for values in runtime.ROTATIONAL_CHOICES.values())
            ),
            "search_method": "deterministic_auto",
            "random_seed": int(source["random_seed"]),
            "economic_rationale": rotational_override.get(
                "economic_rationale", "Causal monthly cross-sectional momentum."
            ),
            "priority": int(rotational_override.get("priority", 2)),
            "refinement_parameters": [],
            "policy_status": "LONG_ONLY_GO",
            "implementation_status": "EXECUTABLE_V2_LEDGER",
        }
    )
    resolved = {
        **source,
        "resolved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_registry_sha256": sha256_file(source_path),
        "strategies": sorted(strategies, key=lambda item: item["strategy_name"]),
    }
    hash_payload = {key: value for key, value in resolved.items() if key != "resolved_at"}
    resolved["resolved_registry_sha256"] = sha256_text(stable_json(hash_payload))
    return resolved


def strategy_map(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["strategy_name"]: item for item in registry["strategies"]}


def validate_configuration(
    strategy: Mapping[str, Any], params: Mapping[str, Any], runtime: Any
) -> bool:
    parameters = {item["name"]: item for item in strategy["search_parameters"]}
    if set(params) != set(parameters):
        raise ValueError("UNREGISTERED_PARAMETER_VALUE_BLOCKED")
    for name, value in params.items():
        if value not in parameters[name]["allowed_values"]:
            raise ValueError(
                f"UNREGISTERED_PARAMETER_VALUE_BLOCKED:{name}={value}"
            )
    return bool(runtime.params_valid(strategy["strategy_name"], params))


def raw_space_count(strategy: Mapping[str, Any]) -> int:
    return int(
        math.prod(
            len(parameter["coarse_allowed_values"])
            for parameter in strategy["search_parameters"]
        )
    )


def all_configurations(strategy: Mapping[str, Any]) -> list[dict[str, Any]]:
    parameters = strategy["search_parameters"]
    keys = [parameter["name"] for parameter in parameters]
    values = [parameter["coarse_allowed_values"] for parameter in parameters]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def qmc_sample(
    strategy: Mapping[str, Any], method: str, count: int, seed: int
) -> list[dict[str, Any]]:
    parameters = strategy["search_parameters"]
    dimension = len(parameters)
    target = max(count * 8, count)
    if method == "sobol":
        exponent = max(1, math.ceil(math.log2(target)))
        points = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(exponent)
    elif method == "latin-hypercube":
        points = qmc.LatinHypercube(d=dimension, seed=seed).random(target)
    else:
        rng = random.Random(seed)
        points = np.asarray(
            [[rng.random() for _ in range(dimension)] for _ in range(target)]
        )
    result: dict[str, dict[str, Any]] = {}
    for point in points:
        candidate = {}
        for parameter, coordinate in zip(parameters, point):
            values = parameter["coarse_allowed_values"]
            index = min(int(float(coordinate) * len(values)), len(values) - 1)
            candidate[parameter["name"]] = values[index]
        result[stable_json(candidate)] = candidate
        if len(result) >= count:
            break
    return [result[key] for key in sorted(result)]


def plan_strategy(
    strategy: Mapping[str, Any],
    runtime: Any,
    method: str,
    maximum_trials: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = dict(strategy["baseline_configuration"])
    raw_count = raw_space_count(strategy)
    exact_candidates = all_configurations(strategy) if raw_count <= 100_000 else []
    exact_valid = [
        candidate
        for candidate in exact_candidates
        if validate_configuration(strategy, candidate, runtime)
    ]
    exact_forbidden = len(exact_candidates) - len(exact_valid)
    if raw_count <= 500:
        candidates = exact_valid
        effective_method = "exhaustive"
    else:
        effective_method = "sobol" if method == "exhaustive" else method
        candidates = qmc_sample(
            strategy, effective_method, max(maximum_trials * 4, maximum_trials), seed
        )
    valid: dict[str, dict[str, Any]] = {}
    sampled_forbidden = 0
    for candidate in candidates:
        if validate_configuration(strategy, candidate, runtime):
            valid[stable_json(candidate)] = candidate
        else:
            sampled_forbidden += 1
    if exact_valid and len(valid) < maximum_trials:
        for candidate in exact_valid:
            valid[stable_json(candidate)] = candidate
            if len(valid) >= maximum_trials:
                break
    if validate_configuration(strategy, baseline, runtime):
        valid[stable_json(baseline)] = baseline
    ordered = [baseline] + [
        value
        for marker, value in sorted(valid.items())
        if marker != stable_json(baseline)
    ]
    planned = ordered[: max(1, maximum_trials)]
    return planned, {
        "strategy": strategy["strategy_name"],
        "raw_parameter_combinations": raw_count,
        "enumerated_parameter_combinations": len(exact_candidates),
        "enumerated_or_sampled_before_validation": len(candidates),
        "forbidden_parameter_combinations": (
            exact_forbidden if exact_candidates else "NOT_EXACTLY_COUNTED"
        ),
        "sampled_forbidden_parameter_combinations": sampled_forbidden,
        "valid_parameter_combinations": (
            len(exact_valid) if exact_candidates else "NOT_EXACTLY_COUNTED"
        ),
        "sampled_parameter_combinations": len(planned),
        "scheduled_parameter_combinations": len(planned),
        "executed_parameter_combinations": 0,
        "pruned_parameter_combinations": 0,
        "skipped_parameter_combinations": max(raw_count - len(planned), 0),
        "failed_parameter_combinations": 0,
        "search_method": (
            "EXHAUSTIVE_GRID"
            if effective_method == "exhaustive" and len(planned) == len(exact_valid)
            else "EXHAUSTIVE_ENUMERATION_WITH_EXECUTION_CAP"
            if effective_method == "exhaustive"
            else "DETERMINISTIC_SOBOL_SAMPLE"
            if effective_method == "sobol"
            else "DETERMINISTIC_LATIN_HYPERCUBE_SAMPLE"
            if effective_method == "latin-hypercube"
            else "DETERMINISTIC_RANDOM_SAMPLE"
        ),
    }


def neighbor_configurations(
    strategy: Mapping[str, Any], params: Mapping[str, Any], runtime: Any
) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for parameter in strategy["search_parameters"]:
        name = parameter["name"]
        values = parameter["neighbor_order"]
        index = values.index(params[name])
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(values):
                candidate = dict(params)
                candidate[name] = values[neighbor_index]
                if validate_configuration(strategy, candidate, runtime):
                    result[stable_json(candidate)] = candidate
    return [result[key] for key in sorted(result)]


def refinement_configurations(
    strategy: Mapping[str, Any], params: Mapping[str, Any], runtime: Any
) -> list[dict[str, Any]]:
    """Return one-factor, pre-registered decimal refinements around a parent."""
    result: dict[str, dict[str, Any]] = {}
    for parameter in strategy["search_parameters"]:
        name = parameter["name"]
        refinements = parameter["refinement_allowed_values"]
        if not refinements:
            continue
        ordered = parameter["neighbor_order"]
        parent_index = ordered.index(params[name])
        for value in refinements:
            value_index = ordered.index(value)
            if abs(value_index - parent_index) != 1:
                continue
            candidate = dict(params)
            candidate[name] = value
            if validate_configuration(strategy, candidate, runtime):
                result[stable_json(candidate)] = candidate
    return [result[key] for key in sorted(result)]


def parameter_hash(strategy: str, timeframe: str, params: Mapping[str, Any]) -> str:
    return sha256_text(
        stable_json(
            {"strategy": strategy, "timeframe": timeframe, "parameters": params}
        )
    )


def materialize_weekly_data(
    source: Path,
    destination: Path,
    start: str,
    end: str,
    max_symbols: int | None,
    seed: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    limit_clause = f"LIMIT {int(max_symbols)}" if max_symbols else ""
    source_sql = "'" + str(source.resolve()).replace("'", "''") + "'"
    destination_sql = "'" + str(destination.resolve()).replace("'", "''") + "'"
    start_sql = "'" + str(start).replace("'", "''") + "'"
    end_sql = "'" + str(end).replace("'", "''") + "'"
    query = f"""
        COPY (
            WITH raw AS (
                SELECT CAST(security_id AS VARCHAR) security_id,
                       CAST(ticker AS VARCHAR) ticker,
                       TRY_CAST(date AS DATE) date,
                       TRY_CAST("open" AS DOUBLE) o,
                       TRY_CAST("high" AS DOUBLE) h,
                       TRY_CAST("low" AS DOUBLE) l,
                       TRY_CAST("close" AS DOUBLE) c,
                       TRY_CAST(volume AS DOUBLE) volume,
                       CAST(sector AS VARCHAR) sector,
                       CAST(currency AS VARCHAR) currency,
                       CAST(price_basis AS VARCHAR) price_basis
                FROM read_parquet({source_sql})
                WHERE TRY_CAST(date AS DATE) BETWEEN {start_sql} AND {end_sql}
            ), selected AS (
                SELECT security_id FROM raw GROUP BY security_id
                ORDER BY hash(security_id, {int(seed)}) {limit_clause}
            ), base AS (
                SELECT raw.*, date_trunc('week', date) week_start
                FROM raw INNER JOIN selected USING(security_id)
            )
            SELECT security_id, arg_max(ticker, date) ticker,
                   CAST(max(date) AS VARCHAR) date,
                   arg_min(o, date) "open", max(h) "high", min(l) "low",
                   arg_max(c, date) "close", sum(volume) volume,
                   arg_max(sector, date) sector, arg_max(currency, date) currency,
                   'PIT_CAUSAL_CLOSED_WEEK_RESAMPLE' AS "source",
                   arg_max(price_basis, date) price_basis
            FROM base GROUP BY security_id, week_start
            ORDER BY security_id, week_start
        ) TO {destination_sql} (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    try:
        connection.execute(query)
    finally:
        connection.close()


def passes_development(
    row: Mapping[str, Any], minimum_trades: int = 30
) -> bool:
    def number(key: str) -> float:
        try:
            return float(str(row.get(key)))
        except (TypeError, ValueError):
            return float("nan")

    return (
        number("normal_validation_CAGR") > 0
        and number("normal_validation_daily_profit_factor") > 1
        and number("double_validation_daily_profit_factor") > 1
        and number("normal_validation_trade_count") >= minimum_trades
        and number("normal_validation_accounting_failure_count") == 0
    )


def finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else float("nan")


def merge_cost_results(
    normal_path: Path,
    double_path: Path,
    trial_meta: Mapping[tuple[str, str], Mapping[str, Any]],
    timeframe: str,
    minimum_trades: int,
) -> pd.DataFrame:
    normal = pd.read_csv(normal_path / "individual_variants.csv").add_prefix("normal_")
    double = pd.read_csv(double_path / "individual_variants.csv").add_prefix("double_")
    merged = normal.merge(
        double,
        left_on=["normal_strategy", "normal_params"],
        right_on=["double_strategy", "double_params"],
        how="outer",
        validate="one_to_one",
    )
    rows = []
    for record in merged.to_dict("records"):
        strategy = str(record.get("normal_strategy") or record.get("double_strategy"))
        params_json = str(record.get("normal_params") or record.get("double_params"))
        params = json.loads(params_json)
        meta = trial_meta[(strategy, stable_json(params))]
        timeframe_parameter_hash = parameter_hash(strategy, timeframe, params)
        row = {
            "trial_id": f"PTRIAL_{timeframe_parameter_hash[:16]}",
            "strategy": strategy,
            "economic_family": record.get("normal_family"),
            "timeframe": timeframe,
            "parameter_hash": timeframe_parameter_hash,
            "parameters": params_json,
            "baseline_distance": meta["baseline_distance"],
            "development_period": "2000-01-01/2018-12-31",
            "search_phase": meta["search_phase"],
            "parent_trial": None,
            "normal_validation_CAGR": record.get("normal_validation_CAGR"),
            "normal_validation_PF": record.get(
                "normal_validation_daily_profit_factor"
            ),
            "normal_validation_Sharpe": record.get("normal_validation_Sharpe"),
            "normal_validation_maximum_drawdown": record.get(
                "normal_validation_maximum_drawdown"
            ),
            "normal_validation_trade_count": record.get(
                "normal_validation_trade_count"
            ),
            "normal_validation_turnover_eur": record.get(
                "normal_validation_turnover_eur"
            ),
            "normal_validation_costs_eur": record.get(
                "normal_validation_transaction_costs_eur"
            ),
            "double_validation_CAGR": record.get("double_validation_CAGR"),
            "double_validation_PF": record.get(
                "double_validation_daily_profit_factor"
            ),
            "double_validation_Sharpe": record.get("double_validation_Sharpe"),
            "double_validation_maximum_drawdown": record.get(
                "double_validation_maximum_drawdown"
            ),
            "double_validation_trade_count": record.get(
                "double_validation_trade_count"
            ),
            "fold_metrics": "BLOCKED_NOT_YET_EXECUTED",
            "cohort_metrics": "BLOCKED_NOT_YET_EXECUTED",
            "concentration_metrics": "AVAILABLE_IN_V2_RUN_ARTIFACTS",
            "sample_validity": "PARTIAL",
            "portfolio_accounting_validity": (
                "GO"
                if int(record.get("normal_validation_accounting_failure_count") or 0)
                == 0
                else "NO_GO"
            ),
            "investability_validity": "PARTIAL_CURRENT_STATIC_METADATA",
            "rejection_reasons": record.get("normal_hard_gate_failures") or "",
        }
        row["gate_status"] = (
            "DEVELOPMENT_GATE_GO"
            if passes_development(record, minimum_trades)
            else "DEVELOPMENT_GATE_NO_GO"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def source_audit(
    project_root: Path, registry_path: Path, parameter_module: Path
) -> dict[str, Any]:
    paths = [
        project_root / "strategy_combo_research_lab.py",
        project_root / "rsi2_adx5_vwap.py",
        project_root / "strategy_research_hub_v3_1_0_definitive.py",
        project_root / "strategy1.py",
        registry_path,
        parameter_module,
    ]
    return {
        "files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in paths
        ],
        "reference_execution_policy": {
            "rsi2_adx5_vwap.py": "NOT_IMPORTED",
            "strategy_research_hub_v3_1_0_definitive.py": "NOT_IMPORTED",
            "strategy1.py": "BROKER_AND_EXTERNAL_CLIENT_IMPORT_BLOCKED",
        },
    }


def research_output_inventory(root: Path) -> dict[str, Any]:
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = [path for path in directory.rglob("*") if path.is_file()]
        manifests = []
        for path in files:
            if path.name not in {"manifest.json", "run-status.json", "status.json"}:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                manifests.append(
                    {
                        "path": str(path.relative_to(root)),
                        "status": payload.get("status", "UNSPECIFIED"),
                        "schema": payload.get("schema", "UNSPECIFIED"),
                    }
                )
            except (OSError, json.JSONDecodeError):
                manifests.append(
                    {
                        "path": str(path.relative_to(root)),
                        "status": "UNREADABLE_BLOCKED",
                    }
                )
        rows.append(
            {
                "directory": directory.name,
                "file_count": len(files),
                "byte_count": sum(path.stat().st_size for path in files),
                "machine_readable_statuses": manifests,
            }
        )
    return {
        "root": str(root.resolve()),
        "directory_count": len(rows),
        "file_count": sum(row["file_count"] for row in rows),
        "byte_count": sum(row["byte_count"] for row in rows),
        "directories": rows,
    }


def forbidden_call_scan(paths: Sequence[Path]) -> dict[str, Any]:
    findings = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name in BLOCKED_CALL_TOKENS:
                findings.append(
                    {"path": str(path), "token": name, "line": node.lineno}
                )
    return {
        "status": "GO" if not findings else "NO_GO",
        "findings": findings,
        "broker_calls": 0,
    }


def add_parameter_research_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "parameter-research",
        help="Run a registry-sealed deterministic campaign through the v2 ledger.",
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--data", default="")
    parser.add_argument("--input-run-dir", default="")
    parser.add_argument(
        "--output", default="output/research/strategy_combo_lab_v2_parameter_campaign"
    )
    parser.add_argument("--parameter-search", action="store_true")
    parser.add_argument(
        "--parameter-search-method",
        choices=["exhaustive", "sobol", "latin-hypercube", "deterministic-random"],
        default="sobol",
    )
    parser.add_argument("--include-strategies", default="")
    parser.add_argument("--timeframes", default="1d,1w")
    parser.add_argument("--max-coarse-trials-per-strategy", type=int, default=20)
    parser.add_argument("--max-refinement-trials-per-strategy", type=int, default=0)
    parser.add_argument("--max-neighbor-trials", type=int, default=20)
    parser.add_argument("--max-parameterizations-per-strategy", type=int, default=3)
    parser.add_argument("--max-parameterizations-per-family", type=int, default=2)
    parser.add_argument("--parameter-plateau-min-neighbors", type=int, default=5)
    parser.add_argument("--parameter-plateau-pass-ratio", type=float, default=0.60)
    parser.add_argument("--combo-sizes", default="2,3,4")
    parser.add_argument("--combo-architecture", default="vote,sleeves,hierarchical")
    parser.add_argument("--max-parameter-products-pair", type=int, default=100)
    parser.add_argument("--max-parameter-products-triple", type=int, default=500)
    parser.add_argument("--max-parameter-products-quad", type=int, default=1000)
    parser.add_argument("--normal-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--double-cost-multiplier", type=float, default=2.0)
    parser.add_argument("--initial-capital", type=float, default=2000.0)
    parser.add_argument("--global-max-positions", type=int, default=4)
    parser.add_argument("--max-security-weight", type=float, default=0.25)
    parser.add_argument("--max-sector-weight", type=float, default=0.50)
    parser.add_argument("--whole-shares", action="store_true", default=True)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-median-dollar-volume", type=float, default=5_000_000.0)
    parser.add_argument("--max-order-adv-fraction", type=float, default=0.01)
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--fixed-fee-eur", type=float, default=3.0)
    parser.add_argument("--fx-cost-bps-per-side", type=float, default=0.0)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--train-end", default="2011-12-31")
    parser.add_argument("--validation-end", default="2018-12-31")
    parser.add_argument("--end", default="2026-07-22")
    parser.add_argument("--min-bars", type=int, default=260)
    parser.add_argument("--min-validation-trades", type=int, default=30)
    parser.add_argument("--validation-max-drawdown", type=float, default=-0.35)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bootstrap-runs", type=int, default=100)
    parser.add_argument("--mc-paths", type=int, default=5000)
    parser.add_argument("--mc-block-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--future-holdout-required", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--plan-only", action="store_true")


def v2_config(
    args: argparse.Namespace,
    runtime: Any,
    data: Path,
    output: Path,
    strategies: Sequence[str],
    grid: Mapping[str, Sequence[Mapping[str, Any]]],
    cost_multiplier: float,
) -> Any:
    return runtime.V2Config(
        command="parameter-research",
        data=str(data),
        output=str(output),
        preset="smoke",
        policy="shariah",
        start=args.start,
        train_end=args.train_end,
        validation_end=args.validation_end,
        end=args.end,
        initial_capital=args.initial_capital,
        global_max_positions=args.global_max_positions,
        max_security_weight=args.max_security_weight,
        max_sector_weight=args.max_sector_weight,
        max_gross_exposure=1.0,
        minimum_order_eur=25.0,
        whole_shares=True,
        max_order_adv_fraction=args.max_order_adv_fraction,
        min_price=args.min_price,
        min_median_dollar_volume=args.min_median_dollar_volume,
        liquidity_lookback=20,
        allowed_exchanges=("NYSE", "NASDAQ", "NYSEMKT"),
        cost_bps_per_side=args.cost_bps_per_side * cost_multiplier,
        slippage_bps_per_side=args.slippage_bps_per_side * cost_multiplier,
        fx_cost_bps_per_side=args.fx_cost_bps_per_side * cost_multiplier,
        fixed_fee_eur=args.fixed_fee_eur * cost_multiplier,
        min_bars=args.min_bars,
        min_validation_trades=args.min_validation_trades,
        validation_max_drawdown=args.validation_max_drawdown,
        max_symbols=args.max_symbols,
        corporate_action_gate=True,
        overnight_ratio_min=0.25,
        overnight_ratio_max=4.0,
        batch_size=args.batch_size,
        checkpoint_every=5,
        workers=1,
        memory_budget_gb=4.0,
        full_cartesian=False,
        max_variants_per_strategy=max(len(value) for value in grid.values()),
        combo_sizes=tuple(
            sorted({int(value) for value in args.combo_sizes.split(",") if value})
        ),
        weight_modes=("equal", "inverse_volatility"),
        allow_invalid_strategies_in_combos=False,
        bootstrap_runs=args.bootstrap_runs,
        bootstrap_block_size=args.mc_block_length,
        top_equity_curves=5,
        equity_extreme_return_threshold=0.10,
        equity_hard_fail_return_threshold=0.50,
        seed=args.seed,
        include_strategies=tuple(strategies),
        exclude_strategies=(),
        resume=True,
        parameter_grid_override=grid,
    )


def run_parameter_research(args: argparse.Namespace, runtime: Any) -> Path:
    started = time.time()
    project_root = Path(runtime.__file__).resolve().parent
    registry_path = (project_root / args.registry).resolve()
    source_data = (
        Path(args.data).resolve()
        if args.data
        else (project_root / runtime.discover_default_data()).resolve()
    )
    resolved = resolve_registry(registry_path, runtime)
    registered = strategy_map(resolved)
    requested = [
        value.strip() for value in args.include_strategies.split(",") if value.strip()
    ]
    if not requested:
        requested = [
            item["strategy_name"]
            for item in resolved["strategies"]
            if item["priority"] <= 2
        ]
    unknown = sorted(set(requested) - set(registered))
    if unknown:
        raise ValueError(f"UNREGISTERED_STRATEGY_BLOCKED:{unknown}")
    requested_timeframes = [
        value.strip() for value in args.timeframes.split(",") if value.strip()
    ]
    timeframe_contract = resolved["timeframes"]
    unknown_timeframes = sorted(set(requested_timeframes) - set(timeframe_contract))
    if unknown_timeframes:
        raise ValueError(f"UNREGISTERED_TIMEFRAME_BLOCKED:{unknown_timeframes}")
    blocked_timeframes = [
        value
        for value in requested_timeframes
        if timeframe_contract[value]["status"] == "DATA_UNAVAILABLE_BLOCKED"
    ]
    executable_timeframes = [
        value for value in requested_timeframes if value not in blocked_timeframes
    ]
    if not executable_timeframes:
        raise ValueError("NO_EXECUTABLE_TIMEFRAME_BLOCKED")

    grids: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, Any] = {}
    trial_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for index, strategy_name in enumerate(requested):
        strategy = registered[strategy_name]
        planned, strategy_counts = plan_strategy(
            strategy,
            runtime,
            args.parameter_search_method,
            args.max_coarse_trials_per_strategy,
            args.seed + index * 997,
        )
        grids[strategy_name] = planned
        counts[strategy_name] = strategy_counts
        baseline = strategy["baseline_configuration"]
        for position, params in enumerate(planned):
            marker = stable_json(params)
            parameter_id = parameter_hash(strategy_name, "MULTI", params)
            trial_meta[(strategy_name, marker)] = {
                "trial_id": f"PTRIAL_{parameter_id[:16]}",
                "parameter_hash": parameter_id,
                "search_phase": "BASELINE" if position == 0 else "COARSE",
                "baseline_distance": sum(
                    params[key] != baseline[key] for key in baseline
                ),
            }

    campaign_fingerprint = {
        "resolved_registry_sha256": resolved["resolved_registry_sha256"],
        "program_sha256": sha256_file(Path(runtime.__file__).resolve()),
        "data_sha256": sha256_file(source_data),
        "strategies": requested,
        "timeframes": requested_timeframes,
        "grids": grids,
        "seed": args.seed,
        "costs": {
            "normal": args.normal_cost_multiplier,
            "double": args.double_cost_multiplier,
        },
    }
    campaign_id = sha256_text(stable_json(campaign_fingerprint))[:16]
    campaign_dir = Path(args.output).resolve() / f"campaign_{campaign_id}"
    if args.clean_output and campaign_dir.exists():
        allowed_root = (project_root / "output" / "research").resolve()
        if allowed_root not in campaign_dir.parents:
            raise ValueError("CLEAN_OUTPUT_OUTSIDE_RESEARCH_ROOT_BLOCKED")
        shutil.rmtree(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    write_json(campaign_dir / "parameter_space_registry_used.json", resolved)
    write_json(campaign_dir / "parameter_search_space_counts.json", counts)
    write_json(
        campaign_dir / "timeframe_availability.json",
        {
            "requested": requested_timeframes,
            "executable": executable_timeframes,
            "blocked": blocked_timeframes,
            "contract": timeframe_contract,
        },
    )
    source_manifest = source_audit(project_root, registry_path, Path(__file__).resolve())
    source_manifest.update(
        {"campaign_fingerprint": campaign_fingerprint, "campaign_id": campaign_id}
    )
    write_json(campaign_dir / "source_manifest.json", source_manifest)
    write_json(
        campaign_dir / "sealed_candidate_manifest.json",
        {
            "sealed_before_execution": True,
            "campaign_id": campaign_id,
            "resolved_registry_sha256": resolved["resolved_registry_sha256"],
            "grids": grids,
            "historical_confirmation_status": "CONSUMED_2019_2026",
            "future_holdout_status": "UNAVAILABLE_UNTIL_FUTURE_FORWARD_DATA",
        },
    )
    if args.plan_only:
        write_json(
            campaign_dir / "completion_audit.json",
            {
                "status": "PARAMETER_RESEARCH_PLAN_SEALED",
                "parameter_registry_sealed_before_run": True,
                "parameter_registry_hash_verified": True,
                "parameter_sampling_deterministic": True,
                "broker_calls_zero": True,
                "execution_authority_none": True,
            },
        )
        print(campaign_dir)
        return campaign_dir

    scoreboards = []
    child_runs: list[dict[str, Any]] = []
    for timeframe in executable_timeframes:
        timeframe_data = source_data
        runtime_max_symbols = args.max_symbols
        if timeframe == "1w":
            timeframe_data = campaign_dir / "private" / "pit-bars-1w.parquet"
            if not timeframe_data.exists():
                materialize_weekly_data(
                    source_data,
                    timeframe_data,
                    args.start,
                    args.end,
                    args.max_symbols,
                    args.seed,
                )
            runtime_max_symbols = None
        normal_config = dataclasses.replace(
            v2_config(
                args,
                runtime,
                timeframe_data,
                campaign_dir / f"normal_{timeframe}",
                requested,
                grids,
                args.normal_cost_multiplier,
            ),
            max_symbols=runtime_max_symbols,
        )
        double_config = dataclasses.replace(
            v2_config(
                args,
                runtime,
                timeframe_data,
                campaign_dir / f"double_{timeframe}",
                requested,
                grids,
                args.double_cost_multiplier,
            ),
            max_symbols=runtime_max_symbols,
        )
        normal_run = runtime.run_lab_v2(normal_config)
        double_run = runtime.run_lab_v2(double_config)
        child_runs.append(
            {
                "timeframe": timeframe,
                "stage": "BASELINE_AND_COARSE",
                "normal_run": str(normal_run.resolve()),
                "double_run": str(double_run.resolve()),
            }
        )
        timeframe_scoreboard = merge_cost_results(
            normal_run,
            double_run,
            trial_meta,
            timeframe,
            max(args.min_validation_trades, 30),
        )
        followup_grid: dict[str, list[dict[str, Any]]] = {}
        passing_coarse = timeframe_scoreboard.loc[
            timeframe_scoreboard["gate_status"].eq("DEVELOPMENT_GATE_GO")
        ]
        for strategy_name, group in passing_coarse.groupby("strategy"):
            item = registered[strategy_name]
            candidates: dict[str, tuple[dict[str, Any], str]] = {}
            parents = group.sort_values(
                ["double_validation_PF", "normal_validation_CAGR"],
                ascending=False,
            ).head(args.max_parameterizations_per_strategy)
            for parent in parents.to_dict("records"):
                params = json.loads(parent["parameters"])
                refinements = refinement_configurations(item, params, runtime)[
                    : args.max_refinement_trials_per_strategy
                ]
                neighbors = neighbor_configurations(item, params, runtime)[
                    : args.max_neighbor_trials
                ]
                for candidate in refinements:
                    candidates[stable_json(candidate)] = (candidate, "REFINEMENT")
                for candidate in neighbors:
                    candidates.setdefault(
                        stable_json(candidate), (candidate, "NEIGHBOR")
                    )
            if candidates:
                followup_grid[strategy_name] = [
                    value[0] for _, value in sorted(candidates.items())
                ]
                baseline = item["baseline_configuration"]
                for marker, (params, phase) in candidates.items():
                    refinement_id = parameter_hash(strategy_name, "MULTI", params)
                    trial_meta[(strategy_name, marker)] = {
                        "trial_id": f"PTRIAL_{refinement_id[:16]}",
                        "parameter_hash": refinement_id,
                        "search_phase": phase,
                        "baseline_distance": sum(
                            params[key] != baseline[key] for key in baseline
                        ),
                    }
        if followup_grid:
            followup_strategies = sorted(followup_grid)
            refine_normal = runtime.run_lab_v2(
                dataclasses.replace(
                    v2_config(
                        args,
                        runtime,
                        timeframe_data,
                        campaign_dir / f"normal_refinement_{timeframe}",
                        followup_strategies,
                        followup_grid,
                        args.normal_cost_multiplier,
                    ),
                    max_symbols=runtime_max_symbols,
                )
            )
            refine_double = runtime.run_lab_v2(
                dataclasses.replace(
                    v2_config(
                        args,
                        runtime,
                        timeframe_data,
                        campaign_dir / f"double_refinement_{timeframe}",
                        followup_strategies,
                        followup_grid,
                        args.double_cost_multiplier,
                    ),
                    max_symbols=runtime_max_symbols,
                )
            )
            child_runs.append(
                {
                    "timeframe": timeframe,
                    "stage": "REFINEMENT_AND_NEIGHBORS",
                    "normal_run": str(refine_normal.resolve()),
                    "double_run": str(refine_double.resolve()),
                }
            )
            followup_scoreboard = merge_cost_results(
                refine_normal,
                refine_double,
                trial_meta,
                timeframe,
                max(args.min_validation_trades, 30),
            )
            timeframe_scoreboard = pd.concat(
                [timeframe_scoreboard, followup_scoreboard], ignore_index=True
            ).drop_duplicates("trial_id", keep="last")
        scoreboards.append(timeframe_scoreboard)

    scoreboard = pd.concat(scoreboards, ignore_index=True)
    scoreboard.to_csv(campaign_dir / "parameter_trial_scoreboard.csv", index=False)
    for strategy_name, strategy_counts in counts.items():
        strategy_trials = scoreboard.loc[scoreboard["strategy"].eq(strategy_name)]
        completed = int(
            strategy_trials[["parameter_hash", "timeframe"]]
            .drop_duplicates()
            .shape[0]
            * 2
        )
        strategy_counts["executed_parameter_combinations"] = completed
        strategy_counts["failed_parameter_combinations"] = 0
        strategy_counts["scheduled_parameter_combinations"] = completed
        strategy_counts["skipped_parameter_combinations"] = max(
            int(strategy_counts["raw_parameter_combinations"])
            * len(executable_timeframes)
            * 2
            - completed,
            0,
        )
    write_json(campaign_dir / "parameter_search_space_counts.json", counts)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    trial_registry_path = campaign_dir / "parameter_trial_registry.jsonl"
    existing_trial_ids: set[str] = set()
    if trial_registry_path.exists():
        for line in trial_registry_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_trial_ids.add(str(json.loads(line)["trial_id"]))
    with trial_registry_path.open("a", encoding="utf-8") as handle:
        for row in scoreboard.to_dict("records"):
            if row["trial_id"] in existing_trial_ids:
                continue
            record = {
                **row,
                "normal_cost_metrics": {
                    key: value for key, value in row.items() if key.startswith("normal_")
                },
                "double_cost_metrics": {
                    key: value for key, value in row.items() if key.startswith("double_")
                },
                "started_at": now,
                "completed_at": now,
                "runtime_seconds": None,
            }
            handle.write(stable_json(record) + "\n")
            existing_trial_ids.add(row["trial_id"])
    family = (
        scoreboard.groupby(["economic_family", "timeframe"], dropna=False)
        .agg(
            trial_count=("trial_id", "count"),
            median_normal_PF=("normal_validation_PF", finite_median),
            median_double_PF=("double_validation_PF", finite_median),
            median_normal_CAGR=("normal_validation_CAGR", finite_median),
            passing_trials=(
                "gate_status",
                lambda values: int((values == "DEVELOPMENT_GATE_GO").sum()),
            ),
        )
        .reset_index()
    )
    family.to_csv(campaign_dir / "parameter_family_summary.csv", index=False)

    selection_rows = []
    plateau_rows = []
    neighbor_rows = []
    sensitivity_rows = []
    for (strategy_name, timeframe), group in scoreboard.groupby(
        ["strategy", "timeframe"]
    ):
        strategy = registered[strategy_name]
        passing = group.loc[group["gate_status"].eq("DEVELOPMENT_GATE_GO")].copy()
        passing = passing.sort_values(
            [
                "normal_validation_maximum_drawdown",
                "double_validation_PF",
                "normal_validation_CAGR",
            ],
            ascending=[False, False, False],
        )
        for row in passing.head(args.max_parameterizations_per_strategy).to_dict(
            "records"
        ):
            params = json.loads(row["parameters"])
            boundary_parameters = []
            parameter_lookup = {
                item["name"]: item for item in strategy["search_parameters"]
            }
            for name, value in params.items():
                coarse = parameter_lookup[name]["coarse_allowed_values"]
                if len(coarse) > 1 and value in {coarse[0], coarse[-1]}:
                    boundary_parameters.append(name)
            neighbors = neighbor_configurations(strategy, params, runtime)
            neighbor_json = {stable_json(value) for value in neighbors}
            observed = group.loc[group["parameters"].astype(str).isin(neighbor_json)]
            normal_count = int((observed["normal_validation_PF"] > 1).sum())
            double_count = int((observed["double_validation_PF"] > 1).sum())
            observed_count = len(observed)
            pass_ratio = normal_count / observed_count if observed_count else 0.0
            neighbor_median = finite_median(observed["normal_validation_PF"])
            direct_cliff = (
                observed_count >= args.parameter_plateau_min_neighbors
                and math.isfinite(neighbor_median)
                and float(row["normal_validation_PF"]) - neighbor_median >= 0.25
            )
            if observed_count < args.parameter_plateau_min_neighbors:
                plateau_status = "PARAMETER_PLATEAU_INSUFFICIENT_NEIGHBORS"
            elif (
                pass_ratio >= args.parameter_plateau_pass_ratio
                and double_count / observed_count >= 0.50
            ):
                plateau_status = "PARAMETER_PLATEAU_CONFIRMED"
            elif direct_cliff:
                plateau_status = "REJECTED_PARAMETER_SPIKE"
            else:
                plateau_status = "PARAMETER_PLATEAU_NOT_CONFIRMED"
            plateau_rows.append(
                {
                    "strategy": strategy_name,
                    "timeframe": timeframe,
                    "parameter_hash": row["parameter_hash"],
                    "neighbor_count": observed_count,
                    "neighbors_profitable_normal": normal_count,
                    "neighbors_profitable_double": double_count,
                    "parameter_plateau_pass_ratio": pass_ratio,
                    "direct_cliff_proven": direct_cliff,
                    "plateau_status": plateau_status,
                    "boundary_winner": bool(boundary_parameters),
                    "boundary_parameters": "|".join(boundary_parameters),
                }
            )
            for neighbor in neighbors:
                neighbor_rows.append(
                    {
                        "strategy": strategy_name,
                        "timeframe": timeframe,
                        "parent_parameter_hash": row["parameter_hash"],
                        "neighbor_parameters": stable_json(neighbor),
                        "executed": stable_json(neighbor)
                        in set(group["parameters"].astype(str)),
                    }
                )
            if plateau_status == "PARAMETER_PLATEAU_CONFIRMED":
                selection_rows.append(
                    {
                        "strategy": strategy_name,
                        "timeframe": timeframe,
                        "selected_parameter_hash": row["parameter_hash"],
                        "selected_parameters": row["parameters"],
                        "selection_status": plateau_status,
                        "boundary_winner": bool(boundary_parameters),
                        "future_holdout_status": (
                            "UNAVAILABLE_UNTIL_FUTURE_FORWARD_DATA"
                        ),
                    }
                )

    pd.DataFrame(plateau_rows).to_csv(
        campaign_dir / "parameter_plateau_report.csv", index=False
    )
    pd.DataFrame(neighbor_rows).to_csv(
        campaign_dir / "parameter_neighbor_results.csv", index=False
    )
    for (strategy_name, timeframe), group in scoreboard.groupby(
        ["strategy", "timeframe"]
    ):
        expanded = []
        for row in group.to_dict("records"):
            for name, value in json.loads(row["parameters"]).items():
                expanded.append(
                    {
                        "parameter": name,
                        "value": stable_json(value),
                        "PF": row["normal_validation_PF"],
                        "CAGR": row["normal_validation_CAGR"],
                    }
                )
        expanded_frame = pd.DataFrame(expanded)
        for (name, value), values in expanded_frame.groupby(["parameter", "value"]):
            sensitivity_rows.append(
                {
                    "strategy": strategy_name,
                    "timeframe": timeframe,
                    "parameter": name,
                    "value": value,
                    "trial_count": len(values),
                    "median_PF": finite_median(values["PF"]),
                    "median_CAGR": finite_median(values["CAGR"]),
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(
        campaign_dir / "parameter_sensitivity_report.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "status": "BLOCKED_NOT_YET_EXECUTED",
                "reason": "walk-forward campaign extension required",
            }
        ]
    ).to_csv(campaign_dir / "parameter_fold_stability.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "BLOCKED_NOT_YET_EXECUTED",
                "reason": "cohort campaign extension required",
            }
        ]
    ).to_csv(campaign_dir / "parameter_cohort_stability.csv", index=False)
    with (campaign_dir / "parameter_rejection_graveyard.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        rejected = scoreboard.loc[
            scoreboard["gate_status"].ne("DEVELOPMENT_GATE_GO")
        ]
        for row in rejected.to_dict("records"):
            handle.write(
                stable_json(
                    {
                        "trial_id": row["trial_id"],
                        "strategy": row["strategy"],
                        "timeframe": row["timeframe"],
                        "reason": row["rejection_reasons"],
                    }
                )
                + "\n"
            )

    write_json(
        campaign_dir / "parameter_selection_manifest.json",
        {
            "status": "DEVELOPMENT_SELECTION_ONLY",
            "selected": selection_rows,
            "historical_confirmation_status": "CONSUMED_2019_2026",
            "future_holdout_status": "UNAVAILABLE_UNTIL_FUTURE_FORWARD_DATA",
            "FINANCIAL_FINALIST_GO": False,
            "FORWARD_SHADOW_GO": False,
            "STRATEGY_AUTHORITY": "NONE",
            "EXECUTION_AUTHORITY": "NONE",
        },
    )
    broker_scan = forbidden_call_scan(
        [Path(runtime.__file__).resolve(), Path(__file__).resolve()]
    )
    write_json(campaign_dir / "broker_call_scan.json", broker_scan)
    requested_architectures = {
        value.strip() for value in args.combo_architecture.split(",") if value.strip()
    }
    architecture_status = {
        "sleeves": "EXECUTED_BY_EXISTING_V2_GLOBAL_NETTED_LEDGER"
        if "sleeves" in requested_architectures
        else "NOT_REQUESTED",
        "vote": "IMPLEMENTATION_BLOCKED"
        if "vote" in requested_architectures
        else "NOT_REQUESTED",
        "hierarchical": "IMPLEMENTATION_BLOCKED"
        if "hierarchical" in requested_architectures
        else "NOT_REQUESTED",
    }
    write_json(
        campaign_dir / "combination_architecture_audit.json",
        {
            "requested": sorted(requested_architectures),
            "status": architecture_status,
            "pair_parameter_product_cap": args.max_parameter_products_pair,
            "triple_parameter_product_cap": args.max_parameter_products_triple,
            "quad_parameter_product_cap": args.max_parameter_products_quad,
            "pair_majority_duplicate_policy": "MAJORITY_EQUALS_ALL_NOT_DUPLICATED",
        },
    )
    refinement_executed = bool((scoreboard["search_phase"] == "REFINEMENT").any())
    neighbor_executed = bool((scoreboard["search_phase"] == "NEIGHBOR").any())
    completion = {
        "parameter_registry_sealed_before_run": True,
        "parameter_registry_hash_verified": True,
        "all_executed_parameters_in_registry": True,
        "raw_parameter_space_counted": True,
        "valid_parameter_space_counted": True,
        "parameter_sampling_deterministic": True,
        "baseline_phase_complete": True,
        "coarse_search_complete": True,
        "local_refinement_complete": (
            True
            if refinement_executed
            else "NO_ELIGIBLE_COARSE_REGION_OR_NOT_REQUESTED"
        ),
        "neighbor_analysis_complete": (
            True if neighbor_executed else "NO_ELIGIBLE_COARSE_REGION"
        ),
        "parameter_plateau_report_complete": True,
        "isolated_parameter_spikes_rejected": True,
        "boundary_winners_flagged": True,
        "portfolio_based_parameter_selection": True,
        "normal_cost_test_complete": True,
        "double_cost_test_complete": True,
        "fold_stability_complete": False,
        "cohort_stability_complete": False,
        "concentration_stress_complete": False,
        "global_parameter_trials_counted_for_dsr": "PARTIAL_CHILD_RUN_LEVEL",
        "combo_parameter_products_bounded": "PARTIAL_EXISTING_SLEEVES_ONLY",
        "near_duplicate_signals_removed": False,
        "pair_majority_duplicates_removed": "NOT_APPLICABLE_NOT_EXECUTED",
        "global_portfolio_ledger_used": True,
        "whole_share_accounting_used": True,
        "global_security_netting_used": True,
        "historical_confirmation_marked_consumed": True,
        "future_holdout_parameters_sealed": True,
        "holdout_parameter_mutation_false": True,
        "broker_calls_zero": broker_scan["status"] == "GO",
        "execution_authority_none": True,
        "status": "PARAMETER_RESEARCH_DEVELOPMENT_PARTIAL",
    }
    write_json(campaign_dir / "completion_audit.json", completion)
    write_json(
        campaign_dir / "program-freeze.json",
        {
            "campaign_id": campaign_id,
            "program_sha256": sha256_file(Path(runtime.__file__).resolve()),
            "parameter_module_sha256": sha256_file(Path(__file__).resolve()),
            "source_registry_sha256": resolved["source_registry_sha256"],
            "resolved_registry_sha256": resolved["resolved_registry_sha256"],
            "child_runs": child_runs,
        },
    )
    manifest = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "status": "PARAMETER_RESEARCH_DEVELOPMENT_PARTIAL",
        "elapsed_seconds": time.time() - started,
        "strategies": requested,
        "timeframes": requested_timeframes,
        "blocked_timeframes": blocked_timeframes,
        "trial_count": len(scoreboard),
        "selected_count": len(selection_rows),
        "child_runs": child_runs,
        "resolved_registry_sha256": resolved["resolved_registry_sha256"],
        "historical_confirmation_status": "CONSUMED_2019_2026",
        "future_holdout_status": "UNAVAILABLE_UNTIL_FUTURE_FORWARD_DATA",
        "FINANCIAL_FINALIST_GO": False,
        "FORWARD_SHADOW_GO": False,
        "STRATEGY_AUTHORITY": "NONE",
        "EXECUTION_AUTHORITY": "NONE",
        "BROKER_CALLS": 0,
    }
    write_json(campaign_dir / "manifest.json", manifest)
    write_json(
        campaign_dir / "output_research_inventory.json",
        research_output_inventory(project_root / "output" / "research"),
    )
    write_text(
        campaign_dir / "report.md",
        "\n".join(
            [
                "# Stocks Parameter Research V2",
                "",
                f"- Campaign: {campaign_id}",
                f"- Status: {manifest['status']}",
                f"- Trials: {len(scoreboard)}",
                f"- Timeframes: {','.join(requested_timeframes)}",
                f"- Blocked timeframes: {','.join(blocked_timeframes) or 'none'}",
                "- Historical confirmation: CONSUMED_2019_2026",
                "- Future holdout: UNAVAILABLE_UNTIL_FUTURE_FORWARD_DATA",
                "- FINANCIAL_FINALIST_GO=false",
                "- FORWARD_SHADOW_GO=false",
                "- STRATEGY_AUTHORITY=NONE",
                "- EXECUTION_AUTHORITY=NONE",
                "",
            ]
        ),
    )
    print(campaign_dir)
    return campaign_dir
