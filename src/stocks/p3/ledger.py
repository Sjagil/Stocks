from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from stocks.p3.contracts import StrategyDNA, UnifiedTrialRecord, stable_hash
from stocks.p3.io import atomic_write_jsonl, read_json


class UnifiedTrialLedger:
    """Append-only P3 consolidation of native deterministic and AI trial sources."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.path = (
            project_root
            / "data"
            / "research"
            / "p3"
            / "private"
            / "unified_evidence.sqlite3"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS unified_trials (
                trial_id TEXT PRIMARY KEY,
                trial_hash TEXT NOT NULL UNIQUE,
                source_namespace TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_namespace, source_record_id)
            );
            CREATE TABLE IF NOT EXISTS import_runs (
                import_hash TEXT PRIMARY KEY,
                source_counts_json TEXT NOT NULL,
                completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def import_records(self, records: Iterable[UnifiedTrialRecord]) -> dict[str, int]:
        inserted = 0
        existing = 0
        with self.connection:
            for record in records:
                payload = record.as_record()
                prior = self.connection.execute(
                    """
                    SELECT trial_hash FROM unified_trials
                    WHERE source_namespace=? AND source_record_id=?
                    """,
                    (record.source_namespace, record.source_record_id),
                ).fetchone()
                if prior is not None:
                    if prior["trial_hash"] != record.trial_hash:
                        raise ValueError(
                            "UNIFIED_TRIAL_IMMUTABILITY_CONFLICT:"
                            f"{record.source_namespace}:{record.source_record_id}"
                        )
                    existing += 1
                    continue
                self.connection.execute(
                    """
                    INSERT INTO unified_trials(
                        trial_id, trial_hash, source_namespace, source_record_id,
                        record_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.trial_id,
                        record.trial_hash,
                        record.source_namespace,
                        record.source_record_id,
                        json.dumps(
                            payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            default=str,
                        ),
                    ),
                )
                inserted += 1
        return {"inserted": inserted, "existing": existing}

    def records(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT record_json FROM unified_trials
            ORDER BY source_namespace, source_record_id
            """
        ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            """
            SELECT source_namespace, count(*) AS count
            FROM unified_trials GROUP BY source_namespace
            ORDER BY source_namespace
            """
        ).fetchall()
        counts = {str(row["source_namespace"]): int(row["count"]) for row in rows}
        counts["TOTAL"] = sum(counts.values())
        return counts

    def record_import_run(self) -> str:
        counts = self.counts()
        import_hash = stable_hash(counts)
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO import_runs(import_hash, source_counts_json)
                VALUES (?, ?)
                """,
                (import_hash, json.dumps(counts, sort_keys=True)),
            )
        return import_hash

    def export_public_jsonl(self) -> Path:
        return atomic_write_jsonl(
            self.project_root / "output" / "research" / "trial-ledger.jsonl",
            self.records(),
        )


def native_strategy_dna(project_root: Path) -> list[StrategyDNA]:
    path = (
        project_root
        / "data"
        / "research"
        / "autopilot"
        / "private"
        / "research_autopilot.sqlite3"
    )
    if not path.exists():
        return []
    connection = _read_only_connection(path)
    try:
        rows = connection.execute(
            """
            SELECT strategy_id, strategy_hash, family, payload_json,
                   'CANONICAL_AUTOPILOT' AS source_registry
            FROM strategies
            UNION ALL
            SELECT strategy_id, strategy_hash, family, payload_json,
                   'BULK_STRATEGY_DNA' AS source_registry
            FROM bulk_strategy_dna
            ORDER BY strategy_id
            """
        ).fetchall()
    finally:
        connection.close()
    return [_strategy_dna_from_row(row) for row in rows]


def native_trial_records(
    project_root: Path,
    dna_by_strategy: dict[str, StrategyDNA],
) -> list[UnifiedTrialRecord]:
    path = (
        project_root
        / "data"
        / "research"
        / "autopilot"
        / "private"
        / "research_autopilot.sqlite3"
    )
    if not path.exists():
        return []
    connection = _read_only_connection(path)
    try:
        standard_rows = connection.execute(
            """
            SELECT t.*, s.family, s.payload_json AS strategy_payload_json
            FROM trials t JOIN strategies s ON s.strategy_id=t.strategy_id
            ORDER BY t.created_at, t.trial_id
            """
        ).fetchall()
        bulk_rows = connection.execute(
            """
            SELECT t.*, s.family, s.payload_json AS strategy_payload_json
            FROM bulk_trials t
            JOIN bulk_strategy_dna s ON s.strategy_id=t.strategy_id
            ORDER BY t.created_at, t.trial_id
            """
        ).fetchall()
    finally:
        connection.close()
    records = [
        _native_trial(row, "AUTOPILOT_STANDARD", dna_by_strategy)
        for row in standard_rows
    ]
    records.extend(
        _native_trial(row, "AUTOPILOT_BULK", dna_by_strategy)
        for row in bulk_rows
    )
    return records


def ai_trial_records(project_root: Path) -> list[UnifiedTrialRecord]:
    root = project_root / "output" / "ai" / "experiments"
    records: list[UnifiedTrialRecord] = []
    for path in sorted(root.glob("EXP-*.json")):
        payload = read_json(path)
        if not payload:
            continue
        hypothesis = payload.get("hypothesis") or {}
        split = payload.get("dataset_split") or {}
        metrics = payload.get("metrics") or payload.get("results") or {}
        blockers = payload.get("blockers") or payload.get("rejection_reasons") or []
        if isinstance(blockers, str):
            blockers = [blockers]
        model_id = payload.get("model_id") or payload.get("model_record_id")
        records.append(
            UnifiedTrialRecord(
                source_namespace="AI_EXPERIMENT",
                source_record_id=str(payload.get("experiment_id") or path.stem),
                research_family=str(
                    payload.get("research_family")
                    or hypothesis.get("family")
                    or "AI_MODEL"
                ),
                hypothesis_id=str(
                    payload.get("hypothesis_id")
                    or hypothesis.get("hypothesis_id")
                    or "AI_HYPOTHESIS_UNSPECIFIED"
                ),
                strategy_id=payload.get("strategy_id"),
                strategy_spec_hash=payload.get("strategy_spec_hash"),
                model_id=None if model_id is None else str(model_id),
                parameters=dict(payload.get("parameters") or {}),
                timeframes=_tuple(payload.get("timeframes") or payload.get("timeframe")),
                universe=_tuple(payload.get("universe") or payload.get("symbols")),
                features=_tuple(payload.get("feature_versions") or payload.get("features")),
                regime_filter=str(payload.get("regime_filter") or "NONE"),
                entry="AI_ADVISORY_ONLY",
                exit="AI_ADVISORY_ONLY",
                stop="AI_HAS_NO_STOP_AUTHORITY",
                target="AI_HAS_NO_TARGET_AUTHORITY",
                holding_rule=str(payload.get("holding_rule") or "MODEL_HORIZON"),
                cost_version=str(payload.get("cost_version") or "NATIVE_COST_MODEL"),
                fill_model_version=str(payload.get("fill_model_version") or "NO_EXECUTION"),
                data_hash=str(
                    payload.get("data_hash")
                    or split.get("dataset_hash")
                    or "UNAVAILABLE"
                ),
                cutoff=str(payload.get("cutoff") or split.get("cutoff") or "UNAVAILABLE"),
                code_hash=str(payload.get("code_hash") or "UNAVAILABLE"),
                seed=_optional_int(payload.get("seed")),
                created_at=str(payload.get("created_at") or payload.get("generated_at") or path.stat().st_mtime_ns),
                status=str(payload.get("status") or "INSUFFICIENT_EVIDENCE"),
                rejection_reason=tuple(str(item) for item in blockers),
                metrics=dict(metrics) if isinstance(metrics, dict) else {},
                provenance={
                    "artifact": str(path.relative_to(project_root)).replace("\\", "/"),
                    "content_hash": payload.get("content_hash"),
                    "dataset_split": split,
                },
            )
        )
    return records


def _strategy_dna_from_row(row: sqlite3.Row) -> StrategyDNA:
    payload = json.loads(row["payload_json"])
    source = str(row["source_registry"])
    is_standard = source == "CANONICAL_AUTOPILOT"
    entry_components = _tuple(payload.get("entry_components") or payload.get("indicator_components"))
    confirmation = _tuple(payload.get("confirmation_components"))
    regime = _tuple(payload.get("regime_components"))
    exit_components = _tuple(payload.get("exit_components"))
    universe = _tuple(payload.get("asset_scope") or payload.get("symbols") or payload.get("asset_class"))
    entry_timeframe = str(payload.get("entry_timeframe") or payload.get("timeframe") or "UNSPECIFIED")
    setup_timeframe = payload.get("confirmation_timeframe")
    context = _tuple(
        [payload.get("regime_timeframe"), payload.get("confirmation_timeframe")]
    )
    missing = []
    for field, value in {
        "economic_hypothesis": payload.get("hypothesis"),
        "stop_rule": payload.get("stop_rule"),
        "target_rule": payload.get("target_rule"),
    }.items():
        if not value:
            missing.append(field)
    return StrategyDNA(
        strategy_id=str(row["strategy_id"]),
        native_strategy_hash=str(row["strategy_hash"]),
        strategy_family=str(payload.get("family") or row["family"] or "UNCLASSIFIED"),
        economic_hypothesis=str(
            payload.get("hypothesis")
            or f"NATIVE_{str(payload.get('formula') or row['family']).upper()}_HYPOTHESIS_NOT_RECORDED"
        ),
        direction="LONG_ONLY" if payload.get("long_only", True) else "LONG_SHORT",
        universe_scope=universe or ("UNSPECIFIED",),
        entry_rule="|".join(entry_components) or str(payload.get("formula") or "NOT_RECORDED"),
        exit_rule="|".join(exit_components) or str(payload.get("execution_assumption") or "NOT_RECORDED"),
        stop_rule=str(payload.get("stop_rule") or "NOT_RECORDED"),
        target_rule=str(payload.get("target_rule") or "NOT_RECORDED"),
        time_exit=str(payload.get("time_exit") or payload.get("rebalance") or "NOT_RECORDED"),
        entry_timeframe=entry_timeframe,
        setup_timeframe=None if setup_timeframe is None else str(setup_timeframe),
        context_timeframes=context,
        feature_set=tuple(sorted(set([*entry_components, *confirmation, *regime]))),
        parameters=dict(payload.get("parameters") or {}),
        regime_filter="|".join(regime) or "NONE",
        position_management=str(
            payload.get("portfolio_model")
            or payload.get("selection_rule")
            or "NOT_RECORDED"
        ),
        cost_model_version="BOUND_PER_TRIAL" if is_standard else "NATIVE_BULK_COST_BPS",
        fill_model_version=str(payload.get("execution_assumption") or "NEXT_BAR_EXECUTION"),
        source_registry=source,
        completeness_status=(
            "DNA_COMPLETE" if not missing else f"DNA_PARTIAL:{'|'.join(sorted(missing))}"
        ),
    )


def _native_trial(
    row: sqlite3.Row,
    namespace: str,
    dna_by_strategy: dict[str, StrategyDNA],
) -> UnifiedTrialRecord:
    strategy_payload = json.loads(row["strategy_payload_json"])
    metrics = json.loads(row["metrics_json"])
    provenance = json.loads(row["provenance_json"])
    strategy_id = str(row["strategy_id"])
    dna = dna_by_strategy.get(strategy_id)
    status = str(row["status"])
    rejection: list[str] = []
    if status not in {"COMPLETE", "GO", "PASS"}:
        rejection.append(status)
    return UnifiedTrialRecord(
        source_namespace=namespace,
        source_record_id=str(row["trial_id"]),
        research_family=str(strategy_payload.get("family") or row["family"] or "UNCLASSIFIED"),
        hypothesis_id=str(
            strategy_payload.get("hypothesis_id")
            or strategy_payload.get("formula")
            or strategy_payload.get("family")
            or strategy_id
        ),
        strategy_id=strategy_id,
        strategy_spec_hash=(
            dna.strategy_spec_hash if dna is not None else strategy_payload.get("strategy_hash")
        ),
        model_id=None,
        parameters=dict(strategy_payload.get("parameters") or {}),
        timeframes=_tuple(
            [
                strategy_payload.get("entry_timeframe") or strategy_payload.get("timeframe"),
                strategy_payload.get("confirmation_timeframe"),
                strategy_payload.get("regime_timeframe"),
            ]
        ),
        universe=_tuple(
            strategy_payload.get("asset_scope")
            or strategy_payload.get("symbols")
            or strategy_payload.get("asset_class")
        ),
        features=_tuple(
            [
                *list(_tuple(strategy_payload.get("entry_components"))),
                *list(_tuple(strategy_payload.get("confirmation_components"))),
                *list(_tuple(strategy_payload.get("regime_components"))),
                *list(_tuple(strategy_payload.get("indicator_components"))),
            ]
        ),
        regime_filter="|".join(_tuple(strategy_payload.get("regime_components"))) or "NONE",
        entry=str(strategy_payload.get("formula") or strategy_payload.get("entry_components") or "NOT_RECORDED"),
        exit=str(strategy_payload.get("exit_components") or strategy_payload.get("execution_assumption") or "NOT_RECORDED"),
        stop=str(strategy_payload.get("stop_rule") or "NOT_RECORDED"),
        target=str(strategy_payload.get("target_rule") or "NOT_RECORDED"),
        holding_rule=str(strategy_payload.get("rebalance") or "STRATEGY_NATIVE"),
        cost_version=str(_native_cost_version(row, provenance)),
        fill_model_version=str(
            provenance.get("fill_model_version")
            or ("NEXT_BAR_EXECUTION" if provenance.get("next_bar_execution") else "UNAVAILABLE")
        ),
        data_hash=str(provenance.get("data_hash") or provenance.get("input_hash") or "UNAVAILABLE"),
        cutoff=str(provenance.get("evaluation_end") or provenance.get("cutoff") or "UNAVAILABLE"),
        code_hash=str(provenance.get("code_hash") or "UNAVAILABLE"),
        seed=_optional_int(strategy_payload.get("seed")),
        created_at=str(row["created_at"]),
        status=status,
        rejection_reason=tuple(rejection),
        metrics=metrics,
        provenance=provenance,
    )


def _read_only_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _native_cost_version(
    row: sqlite3.Row, provenance: dict[str, Any]
) -> str:
    if provenance.get("cost_version"):
        return str(provenance["cost_version"])
    keys = set(row.keys())
    if "cost_profile" in keys and row["cost_profile"]:
        return str(row["cost_profile"])
    if "cost_bps" in keys and row["cost_bps"] is not None:
        return f"COST_BPS_{row['cost_bps']}"
    return "UNAVAILABLE"


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if item is not None and str(item).strip())
    return (str(value),)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
