from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.p3.io import atomic_write_json, file_hash, read_json
from stocks.rl.contracts import stable_hash


CATALOG_PATH = Path("data/p4/private/catalog.json")
PUBLIC_STATUS_PATH = Path("output/p4/data-catalog-status.json")
BUNDLE_STATUS_PATH = Path("output/p4/bundle-ingest-status.json")


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    required_columns: tuple[str, ...]
    timestamp_columns: tuple[str, ...]
    identity_columns: tuple[str, ...]
    interval_start: str | None = None
    interval_end: str | None = None


DATASET_SCHEMAS = {
    "security_master": DatasetSchema(
        "security_master",
        (
            "security_id",
            "symbol",
            "exchange",
            "currency",
            "effective_from",
            "effective_to",
            "available_at",
            "listing_status",
        ),
        ("effective_from", "effective_to", "available_at"),
        ("security_id", "effective_from"),
        "effective_from",
        "effective_to",
    ),
    "universe_membership": DatasetSchema(
        "universe_membership",
        (
            "universe_id",
            "security_id",
            "effective_from",
            "effective_to",
            "available_at",
            "is_member",
        ),
        ("effective_from", "effective_to", "available_at"),
        ("universe_id", "security_id", "effective_from"),
        "effective_from",
        "effective_to",
    ),
    "delistings": DatasetSchema(
        "delistings",
        (
            "security_id",
            "delisting_date",
            "available_at",
            "delisting_return",
            "reason",
        ),
        ("delisting_date", "available_at"),
        ("security_id", "delisting_date"),
    ),
    "corporate_actions": DatasetSchema(
        "corporate_actions",
        (
            "security_id",
            "action_date",
            "available_at",
            "action_type",
            "value",
        ),
        ("action_date", "available_at"),
        ("security_id", "action_date", "action_type"),
    ),
    "fundamentals": DatasetSchema(
        "fundamentals",
        (
            "security_id",
            "period_end",
            "available_at",
            "metric",
            "value",
            "revision_id",
        ),
        ("period_end", "available_at"),
        ("security_id", "period_end", "metric", "revision_id"),
    ),
    "shariah_classification": DatasetSchema(
        "shariah_classification",
        (
            "security_id",
            "effective_from",
            "effective_to",
            "available_at",
            "status",
            "methodology_version",
        ),
        ("effective_from", "effective_to", "available_at"),
        ("security_id", "effective_from", "methodology_version"),
        "effective_from",
        "effective_to",
    ),
    "daily_prices": DatasetSchema(
        "daily_prices",
        (
            "security_id",
            "session_date",
            "available_at",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjustment_version",
        ),
        ("session_date", "available_at"),
        ("security_id", "session_date", "adjustment_version"),
    ),
}


@dataclass(frozen=True)
class SourceAttestation:
    provider: str
    source_version: str
    license_id: str
    licensed_for_research: bool
    complete_history_attested: bool
    point_in_time_semantics_attested: bool
    obtained_at: str
    operator: str

    def validate(self) -> None:
        for name in (
            "licensed_for_research",
            "complete_history_attested",
            "point_in_time_semantics_attested",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"P4 source attestation {name} must be boolean")
        if not all(
            str(value).strip()
            for value in (
                self.provider,
                self.source_version,
                self.license_id,
                self.obtained_at,
                self.operator,
            )
        ):
            raise ValueError("P4 source attestation identity is incomplete")
        obtained = pd.to_datetime(self.obtained_at, utc=True, errors="coerce")
        if pd.isna(obtained):
            raise ValueError("P4 source attestation obtained_at is invalid")
        if obtained > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1):
            raise ValueError("P4 source attestation obtained_at is in the future")

    @property
    def production_eligible(self) -> bool:
        return all(
            (
                self.licensed_for_research,
                self.complete_history_attested,
                self.point_in_time_semantics_attested,
            )
        )


class PITDataCatalog:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.path = self.project_root / CATALOG_PATH
        if not self.path.is_file():
            atomic_write_json(
                self.path,
                {
                    "schema": "p4_pit_data_catalog_v1",
                    "datasets": {},
                    "immutable_snapshots": True,
                    "current_data_may_replace_history": False,
                    "execution_authority": "NONE",
                    "money_control": False,
                    "updated_at": _now(),
                },
            )

    def register_snapshot(
        self,
        dataset: str,
        source_path: Path,
        normalized_path: Path,
        frame: pd.DataFrame,
        attestation: SourceAttestation,
    ) -> dict[str, Any]:
        attestation.validate()
        catalog = read_json(self.path)
        source_hash = file_hash(source_path)
        normalized_hash = file_hash(normalized_path)
        snapshot_id = stable_hash(
            {
                "dataset": dataset,
                "source_hash": source_hash,
                "normalized_hash": normalized_hash,
                "attestation": asdict(attestation),
            }
        )
        existing = catalog.setdefault("datasets", {}).setdefault(dataset, [])
        for record in existing:
            if record.get("snapshot_id") == snapshot_id:
                return record
        timestamps = _timestamp_extent(frame, DATASET_SCHEMAS[dataset])
        record = {
            "snapshot_id": snapshot_id,
            "dataset": dataset,
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "source_sha256": source_hash,
            "normalized_sha256": normalized_hash,
            "normalized_path": normalized_path.relative_to(self.project_root).as_posix(),
            "time_extent": timestamps,
            "attestation": asdict(attestation),
            "production_eligible": attestation.production_eligible,
            "registered_at": _now(),
        }
        existing.append(record)
        catalog["updated_at"] = _now()
        catalog["catalog_hash"] = stable_hash(
            {key: value for key, value in catalog.items() if key != "catalog_hash"}
        )
        atomic_write_json(self.path, catalog)
        return record

    def audit(self) -> dict[str, Any]:
        catalog = read_json(self.path)
        datasets = catalog.get("datasets", {})
        latest: dict[str, Any] = {}
        latest_research: dict[str, Any] = {}
        for name in DATASET_SCHEMAS:
            records = datasets.get(name, [])
            eligible = [
                row
                for row in records
                if row.get("production_eligible")
            ]
            candidate = eligible[-1] if eligible else None
            latest[name] = (
                candidate
                if candidate is not None
                and _snapshot_record_integrity(self.project_root, candidate)
                else None
            )
            research_candidate = records[-1] if records else None
            latest_research[name] = (
                research_candidate
                if research_candidate is not None
                and _snapshot_record_integrity(
                    self.project_root, research_candidate
                )
                else None
            )
        missing = [name for name, record in latest.items() if record is None]
        coherence = _audit_bundle_coherence(
            self.project_root,
            latest,
            target_universe=_target_universe(self.project_root),
        )
        gates = {
            "PIT_DATA_GO": all(
                latest.get(name) is not None
                for name in (
                    "security_master",
                    "universe_membership",
                    "corporate_actions",
                    "fundamentals",
                    "daily_prices",
                )
            )
            and coherence["pit_data_coherence_go"],
            "SURVIVORSHIP_GO": all(
                latest.get(name) is not None
                for name in ("security_master", "universe_membership", "delistings")
            )
            and coherence["survivorship_coherence_go"],
            "SHARIAH_PIT_GO": (
                latest.get("shariah_classification") is not None
                and coherence["shariah_coherence_go"]
            ),
        }
        catalog_records = [
            record
            for records in datasets.values()
            for record in records
        ]
        payload: dict[str, Any] = {
            "schema": "p4_pit_data_catalog_status_v1",
            "status": "GO" if all(gates.values()) else "EXTERNAL_DATA_REQUIRED",
            "gates": gates,
            "required_datasets": list(DATASET_SCHEMAS),
            "missing_production_eligible_datasets": missing,
            "latest_production_eligible_snapshots": latest,
            "latest_integrity_valid_snapshots": latest_research,
            "bundle_coherence": coherence,
            "snapshot_integrity_verified": bool(catalog_records) and all(
                _snapshot_record_integrity(self.project_root, record)
                for record in catalog_records
            ),
            "catalog_path": CATALOG_PATH.as_posix(),
            "current_membership_substitution_allowed": False,
            "current_shariah_substitution_allowed": False,
            "zero_delisting_return_imputation_allowed": False,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "broker_writes": 0,
            "money_control": False,
            "generated_at": _now(),
        }
        payload["content_hash"] = stable_hash(payload)
        atomic_write_json(self.project_root / PUBLIC_STATUS_PATH, payload)
        return payload


def ingest_point_in_time_snapshot(
    project_root: Path,
    dataset: str,
    source_path: Path,
    attestation: SourceAttestation,
) -> dict[str, Any]:
    root = project_root.resolve()
    if dataset not in DATASET_SCHEMAS:
        raise ValueError(f"unsupported P4 dataset: {dataset}")
    source = source_path.resolve()
    if not source.is_file():
        raise ValueError("P4 source snapshot does not exist")
    frame = _read_snapshot_source(source)
    normalized = validate_point_in_time_frame(frame, DATASET_SCHEMAS[dataset])
    attestation.validate()
    identity = stable_hash(
        {
            "dataset": dataset,
            "source_hash": file_hash(source),
            "attestation": asdict(attestation),
        }
    )
    destination = root / "data/p4/private/snapshots" / dataset / f"{identity}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"{identity}-",
        suffix=".parquet.tmp",
        dir=destination.parent,
        delete=False,
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        normalized.to_parquet(temporary, index=False)
        if destination.exists():
            if file_hash(destination) != file_hash(temporary):
                raise ValueError("immutable P4 normalized snapshot collision")
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    catalog = PITDataCatalog(root)
    record = catalog.register_snapshot(dataset, source, destination, normalized, attestation)
    return {
        "schema": "p4_pit_snapshot_ingest_result_v1",
        "status": "GO" if record["production_eligible"] else "RESEARCH_ONLY",
        "snapshot": record,
        "catalog_status": catalog.audit(),
        "execution_authority": "NONE",
        "broker_writes": 0,
    }


def ingest_point_in_time_bundle(
    project_root: Path, manifest_path: Path
) -> dict[str, Any]:
    root = project_root.resolve()
    manifest_file = manifest_path.resolve()
    if not manifest_file.is_file():
        raise ValueError("P4 bundle manifest does not exist")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("P4 bundle manifest is not valid JSON") from exc
    if manifest.get("schema") != "p4_pit_bundle_manifest_v1":
        raise ValueError("P4 bundle manifest schema is unsupported")
    configured_universe = _target_universe(root)
    manifest_universe = str(manifest.get("target_universe") or "").strip()
    if not manifest_universe:
        raise ValueError("P4 bundle target_universe is required")
    if configured_universe and manifest_universe != configured_universe:
        raise ValueError("P4 bundle target_universe does not match frozen policy")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("P4 bundle datasets must be an object")
    missing = sorted(set(DATASET_SCHEMAS) - set(datasets))
    extra = sorted(set(datasets) - set(DATASET_SCHEMAS))
    if missing or extra:
        raise ValueError(
            f"P4 bundle dataset set mismatch; missing={missing}; extra={extra}"
        )

    sources: dict[str, Path] = {}
    frames: dict[str, pd.DataFrame] = {}
    attestations: dict[str, SourceAttestation] = {}
    for dataset in DATASET_SCHEMAS:
        entry = datasets[dataset]
        if not isinstance(entry, dict):
            raise ValueError(f"P4 bundle {dataset} entry must be an object")
        raw_source = Path(str(entry.get("source") or ""))
        source = (
            raw_source
            if raw_source.is_absolute()
            else manifest_file.parent / raw_source
        ).resolve()
        if not source.is_file():
            raise ValueError(f"P4 bundle {dataset} source does not exist")
        raw_attestation = entry.get("attestation")
        if not isinstance(raw_attestation, dict):
            raise ValueError(f"P4 bundle {dataset} attestation is required")
        try:
            attestation = SourceAttestation(**raw_attestation)
        except TypeError as exc:
            raise ValueError(
                f"P4 bundle {dataset} attestation fields are invalid"
            ) from exc
        attestation.validate()
        sources[dataset] = source
        attestations[dataset] = attestation
        frames[dataset] = validate_point_in_time_frame(
            _read_snapshot_source(source), DATASET_SCHEMAS[dataset]
        )

    coherence = _audit_frames_coherence(
        frames, target_universe=manifest_universe
    )
    if coherence["status"] != "GO":
        payload = _bundle_status_payload(
            manifest_file,
            manifest_universe,
            coherence,
            snapshots={},
            status="NO_GO",
        )
        atomic_write_json(root / BUNDLE_STATUS_PATH, payload)
        return payload

    snapshots: dict[str, Any] = {}
    for dataset in DATASET_SCHEMAS:
        result = ingest_point_in_time_snapshot(
            root, dataset, sources[dataset], attestations[dataset]
        )
        snapshots[dataset] = result["snapshot"]
    status = (
        "GO"
        if all(row.get("production_eligible") for row in snapshots.values())
        else "RESEARCH_ONLY"
    )
    payload = _bundle_status_payload(
        manifest_file,
        manifest_universe,
        coherence,
        snapshots=snapshots,
        status=status,
    )
    payload["catalog_status"] = PITDataCatalog(root).audit()
    payload["content_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )
    atomic_write_json(root / BUNDLE_STATUS_PATH, payload)
    return payload


def validate_point_in_time_frame(
    frame: pd.DataFrame, schema: DatasetSchema
) -> pd.DataFrame:
    missing = sorted(set(schema.required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{schema.name} columns missing: {missing}")
    result = frame.loc[:, list(schema.required_columns)].copy()
    for name in schema.timestamp_columns:
        result[name] = pd.to_datetime(result[name], utc=True, errors="coerce")
        if name != schema.interval_end and result[name].isna().any():
            raise ValueError(f"{schema.name}.{name} contains invalid timestamps")
    if result.duplicated(list(schema.identity_columns)).any():
        raise ValueError(f"{schema.name} contains duplicate immutable identities")
    available = pd.to_datetime(result["available_at"], utc=True)
    if (available > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).any():
        raise ValueError(f"{schema.name} contains future availability timestamps")
    availability_floor = {
        "fundamentals": "period_end",
        "daily_prices": "session_date",
    }.get(schema.name)
    if availability_floor:
        reference = pd.to_datetime(result[availability_floor], utc=True)
        if (available < reference).any():
            raise ValueError(
                f"{schema.name}.available_at precedes {availability_floor}"
            )
    if schema.interval_start and schema.interval_end:
        starts = pd.to_datetime(result[schema.interval_start], utc=True)
        ends = pd.to_datetime(result[schema.interval_end], utc=True)
        invalid = ends.notna() & (ends <= starts)
        if invalid.any():
            raise ValueError(f"{schema.name} contains invalid effective intervals")
        _validate_non_overlapping_intervals(result, schema)
    numeric_ohlcv = [name for name in ("open", "high", "low", "close", "volume") if name in result]
    if numeric_ohlcv:
        values = result[numeric_ohlcv].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any() or (values[[name for name in numeric_ohlcv if name != "volume"]] <= 0).any().any():
            raise ValueError("daily_prices contains invalid OHLCV values")
        result[numeric_ohlcv] = values
        if "volume" in values and (values["volume"] < 0).any():
            raise ValueError("daily_prices contains negative volume")
        if {"open", "high", "low", "close"}.issubset(values.columns):
            if (
                (values["high"] < values[["open", "close"]].max(axis=1)).any()
                or (values["low"] > values[["open", "close"]].min(axis=1)).any()
                or (values["high"] < values["low"]).any()
            ):
                raise ValueError("daily_prices contains inconsistent OHLC values")
    return result.sort_values(list(schema.identity_columns)).reset_index(drop=True)


def _read_snapshot_source(source: Path) -> pd.DataFrame:
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(
            source, sep="\t" if source.suffix.lower() == ".tsv" else ","
        )
    raise ValueError("P4 snapshot must be CSV, TSV or Parquet")


def _validate_non_overlapping_intervals(frame: pd.DataFrame, schema: DatasetSchema) -> None:
    group_columns = [name for name in schema.identity_columns if name != schema.interval_start]
    if not group_columns:
        return
    for _, group in frame.groupby(group_columns, dropna=False):
        ordered = group.sort_values(schema.interval_start)
        previous_end: pd.Timestamp | None = None
        for _, row in ordered.iterrows():
            start = pd.Timestamp(row[schema.interval_start])
            end = row[schema.interval_end]
            if previous_end is not None and start < previous_end:
                raise ValueError(f"{schema.name} contains overlapping effective intervals")
            previous_end = pd.Timestamp.max.tz_localize("UTC") if pd.isna(end) else pd.Timestamp(end)


def _timestamp_extent(frame: pd.DataFrame, schema: DatasetSchema) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in schema.timestamp_columns:
        values = pd.to_datetime(frame[name], utc=True).dropna()
        result[name] = {
            "minimum": values.min().isoformat() if len(values) else None,
            "maximum": values.max().isoformat() if len(values) else None,
        }
    return result


def _snapshot_record_integrity(
    project_root: Path, record: dict[str, Any]
) -> bool:
    relative = str(record.get("normalized_path") or "")
    expected = str(record.get("normalized_sha256") or "")
    if not relative or not expected:
        return False
    candidate = project_root / relative
    if candidate.is_symlink():
        return False
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(project_root.resolve())
    except (OSError, ValueError):
        return False
    return path.is_file() and file_hash(path) == expected


def _target_universe(project_root: Path) -> str | None:
    policy_path = project_root / "config/p4_data_policy_v1.json"
    if not policy_path.is_file():
        return None
    value = read_json(policy_path).get("target_universe")
    return str(value).strip() or None if value is not None else None


def _audit_bundle_coherence(
    project_root: Path,
    latest: dict[str, Any],
    *,
    target_universe: str | None,
) -> dict[str, Any]:
    frames: dict[str, pd.DataFrame] = {}
    blockers: list[str] = []
    for name, record in latest.items():
        if record is None:
            continue
        try:
            frames[name] = pd.read_parquet(
                project_root / str(record["normalized_path"])
            )
        except (OSError, KeyError, ValueError) as exc:
            blockers.append(f"{name}:UNREADABLE:{type(exc).__name__}")

    return _audit_frames_coherence(frames, target_universe=target_universe, blockers=blockers)


def _audit_frames_coherence(
    frames: dict[str, pd.DataFrame],
    *,
    target_universe: str | None,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    blockers = list(blockers or [])
    master = frames.get("security_master")
    membership = frames.get("universe_membership")
    master_ids = _security_ids(master)
    member_ids: set[str] = set()
    target_universe_present = target_universe is None
    historical_membership_changes_present = False
    delisted_master_ids: set[str] = set()

    if master is not None:
        statuses = master["listing_status"].fillna("").astype(str).str.upper()
        delisted_master_ids = set(
            master.loc[statuses.isin({"DELISTED", "INACTIVE"}), "security_id"]
            .astype(str)
            .str.strip()
        )
        delisted_master_ids.discard("")
    if membership is not None:
        membership_ids = _security_ids(membership)
        member_mask = membership["is_member"].map(_strict_bool)
        invalid_membership_values = member_mask.isna().any()
        if invalid_membership_values:
            blockers.append("universe_membership:INVALID_IS_MEMBER")
        member_ids = set(
            membership.loc[member_mask.fillna(False), "security_id"]
            .astype(str)
            .str.strip()
        )
        member_ids.discard("")
        if target_universe is not None:
            target_universe_present = bool(
                membership["universe_id"].astype(str).eq(target_universe).any()
            )
            if not target_universe_present:
                blockers.append("universe_membership:TARGET_UNIVERSE_MISSING")
        historical_membership_changes_present = bool(
            membership["effective_to"].notna().any()
            or member_mask.eq(False).any()
        )
        if not historical_membership_changes_present:
            blockers.append("universe_membership:NO_HISTORICAL_EXIT_EVIDENCE")
        _append_foreign_id_blocker(
            blockers, "universe_membership", membership_ids, master_ids
        )

    coverage: dict[str, Any] = {}
    for name in (
        "delistings",
        "corporate_actions",
        "fundamentals",
        "shariah_classification",
        "daily_prices",
    ):
        frame = frames.get(name)
        ids = _security_ids(frame)
        _append_foreign_id_blocker(blockers, name, ids, master_ids)
        missing_members = sorted(member_ids - ids)
        coverage[name] = {
            "security_count": len(ids),
            "member_security_count": len(member_ids & ids),
            "missing_member_security_count": len(missing_members),
            "missing_member_security_ids_sample": missing_members[:20],
        }

    delisting_ids = _security_ids(frames.get("delistings"))
    missing_delistings = sorted(delisted_master_ids - delisting_ids)
    if not delisted_master_ids and master is not None:
        blockers.append("security_master:NO_DELISTED_OR_INACTIVE_SECURITIES")
    if missing_delistings:
        blockers.append("delistings:DELISTED_SECURITY_COVERAGE_INCOMPLETE")

    member_coverage = {
        name: bool(member_ids)
        and frames.get(name) is not None
        and not (member_ids - _security_ids(frames.get(name)))
        for name in ("fundamentals", "shariah_classification", "daily_prices")
    }
    referential_integrity_go = not any("FOREIGN_SECURITY_IDS" in row for row in blockers)
    pit_data_coherence_go = all(
        (
            bool(master_ids),
            bool(member_ids),
            target_universe_present,
            referential_integrity_go,
            member_coverage["fundamentals"],
            member_coverage["daily_prices"],
        )
    )
    survivorship_coherence_go = all(
        (
            bool(master_ids),
            bool(member_ids),
            bool(delisted_master_ids),
            not missing_delistings,
            historical_membership_changes_present,
            referential_integrity_go,
        )
    )
    shariah_coherence_go = all(
        (
            bool(member_ids),
            member_coverage["shariah_classification"],
            referential_integrity_go,
        )
    )
    return {
        "status": (
            "GO"
            if all(
                (
                    pit_data_coherence_go,
                    survivorship_coherence_go,
                    shariah_coherence_go,
                )
            )
            else "NO_GO"
        ),
        "target_universe": target_universe,
        "target_universe_present": target_universe_present,
        "master_security_count": len(master_ids),
        "member_security_count": len(member_ids),
        "delisted_master_security_count": len(delisted_master_ids),
        "missing_delisting_security_count": len(missing_delistings),
        "missing_delisting_security_ids_sample": missing_delistings[:20],
        "historical_membership_changes_present": historical_membership_changes_present,
        "referential_integrity_go": referential_integrity_go,
        "member_coverage": member_coverage,
        "coverage": coverage,
        "pit_data_coherence_go": pit_data_coherence_go,
        "survivorship_coherence_go": survivorship_coherence_go,
        "shariah_coherence_go": shariah_coherence_go,
        "blockers": sorted(set(blockers)),
    }


def _bundle_status_payload(
    manifest_path: Path,
    target_universe: str,
    coherence: dict[str, Any],
    *,
    snapshots: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "p4_pit_bundle_ingest_status_v1",
        "status": status,
        "manifest_sha256": file_hash(manifest_path),
        "target_universe": target_universe,
        "bundle_coherence": coherence,
        "snapshots": snapshots,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "money_control": False,
        "generated_at": _now(),
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _security_ids(frame: pd.DataFrame | None) -> set[str]:
    if frame is None or "security_id" not in frame.columns:
        return set()
    result = set(frame["security_id"].dropna().astype(str).str.strip())
    result.discard("")
    return result


def _append_foreign_id_blocker(
    blockers: list[str], dataset: str, ids: set[str], master_ids: set[str]
) -> None:
    if ids and not master_ids:
        blockers.append(f"{dataset}:SECURITY_MASTER_UNAVAILABLE")
    elif ids - master_ids:
        blockers.append(f"{dataset}:FOREIGN_SECURITY_IDS")


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "BUNDLE_STATUS_PATH",
    "CATALOG_PATH",
    "DATASET_SCHEMAS",
    "DatasetSchema",
    "PITDataCatalog",
    "PUBLIC_STATUS_PATH",
    "SourceAttestation",
    "ingest_point_in_time_bundle",
    "ingest_point_in_time_snapshot",
    "validate_point_in_time_frame",
]
