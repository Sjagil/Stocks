from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from stocks.execution.idempotency import stable_hash


class ProviderId(StrEnum):
    EODHD = "EODHD"
    SEC_EDGAR = "SEC_EDGAR"
    OPENEXCHANGERATES = "OPENEXCHANGERATES"


class CapabilityStatus(StrEnum):
    GO = "PROVIDER_CAPABILITY_GO"
    PARTIAL = "PROVIDER_CAPABILITY_PARTIAL"
    BLOCKED = "PROVIDER_CAPABILITY_BLOCKED"
    ENDPOINT_NOT_AVAILABLE = "ENDPOINT_NOT_AVAILABLE"
    PLAN_NOT_ENTITLED = "PLAN_NOT_ENTITLED"
    PIT_HISTORY_UNAVAILABLE = "PIT_HISTORY_UNAVAILABLE"
    REVISION_HISTORY_UNAVAILABLE = "REVISION_HISTORY_UNAVAILABLE"
    PUBLICATION_TIMESTAMP_UNAVAILABLE = "PUBLICATION_TIMESTAMP_UNAVAILABLE"
    SURVIVORSHIP_RISK_BLOCKED = "SURVIVORSHIP_RISK_BLOCKED"
    LICENSE_REVIEW_REQUIRED = "LICENSE_REVIEW_REQUIRED"


class PitStatus(StrEnum):
    ELIGIBLE = "PIT_ELIGIBLE"
    DAILY_DELAY = "PIT_ELIGIBLE_WITH_DAILY_DELAY"
    TIMESTAMP_INCOMPLETE = "PIT_TIMESTAMP_INCOMPLETE"
    REVISION_UNTRACKED = "PIT_REVISION_UNTRACKED"
    PROVIDER_BLOCKED = "PIT_PROVIDER_BLOCKED"
    FUTURE_LEAKAGE_BLOCKED = "PIT_FUTURE_LEAKAGE_BLOCKED"


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    dataset_id: str
    endpoint_family: str
    asset_coverage: tuple[str, ...]
    exchange_coverage: tuple[str, ...]
    active_symbols: bool
    delisted_symbols: bool
    historical_start: str | None
    timestamp_fields: tuple[str, ...]
    timezone: str
    revision_history: str
    restatement_support: bool
    publication_time_available: bool
    point_in_time_usable: bool
    survivorship_risk: str
    rate_limit: str
    retention_allowed: str
    raw_payload_storage_allowed: str
    license_notes: str
    capability_status: str
    verified_at: str
    documentation_hash: str

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PitRecord:
    record_id: str
    entity_id: str
    symbol: str
    con_id: int | None
    provider: str
    dataset: str
    source_record_hash: str
    source_version_hash: str
    payload_hash: str
    event_time: str | None
    period_end: str | None
    published_at: str | None
    accepted_at: str | None
    provider_available_at: str | None
    first_seen_at: str
    ingested_at: str
    revised_at: str | None
    superseded_at: str | None
    decision_available_at: str | None
    timezone: str
    timestamp_precision: str
    revision_number: int
    is_latest_version: bool
    PIT_eligibility: str
    PIT_blockers: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def available_for(self, decision_time: str) -> bool:
        if self.PIT_eligibility not in {PitStatus.ELIGIBLE, PitStatus.DAILY_DELAY}:
            return False
        if not self.decision_available_at:
            return False
        return datetime.fromisoformat(decision_time) >= datetime.fromisoformat(self.decision_available_at)


def make_pit_record(
    *,
    entity_id: str,
    symbol: str,
    provider: str,
    dataset: str,
    payload: Any,
    first_seen_at: str,
    ingested_at: str,
    event_time: str | None = None,
    period_end: str | None = None,
    published_at: str | None = None,
    accepted_at: str | None = None,
    provider_available_at: str | None = None,
    timestamp_precision: str = "SECOND",
    revision_number: int = 1,
    revision_tracked: bool = True,
    provider_blocked: bool = False,
    con_id: int | None = None,
) -> PitRecord:
    payload_hash = stable_hash(payload)
    candidates = [value for value in (accepted_at, provider_available_at, published_at, first_seen_at) if value]
    available_at = max(candidates, key=datetime.fromisoformat) if candidates else None
    blockers: list[str] = []
    if provider_blocked:
        blockers.append("PROVIDER_CAPABILITY_BLOCKED")
    if available_at is None:
        blockers.append("AVAILABLE_AT_MISSING")
    if not revision_tracked:
        blockers.append("REVISION_HISTORY_UNAVAILABLE")
    if period_end and available_at == period_end:
        blockers.append("PERIOD_END_AS_AVAILABLE_AT_BLOCKED")

    if provider_blocked:
        pit_status = PitStatus.PROVIDER_BLOCKED
    elif available_at is None or "PERIOD_END_AS_AVAILABLE_AT_BLOCKED" in blockers:
        pit_status = PitStatus.TIMESTAMP_INCOMPLETE
    elif not revision_tracked:
        pit_status = PitStatus.REVISION_UNTRACKED
    elif timestamp_precision.upper() == "DATE":
        pit_status = PitStatus.DAILY_DELAY
    else:
        pit_status = PitStatus.ELIGIBLE
    version_hash = stable_hash({"payload_hash": payload_hash, "revision_number": revision_number})
    return PitRecord(
        record_id=stable_hash({"provider": provider, "dataset": dataset, "entity": entity_id, "version": version_hash}),
        entity_id=entity_id,
        symbol=symbol,
        con_id=con_id,
        provider=provider,
        dataset=dataset,
        source_record_hash=stable_hash({"provider": provider, "dataset": dataset, "entity": entity_id}),
        source_version_hash=version_hash,
        payload_hash=payload_hash,
        event_time=event_time,
        period_end=period_end,
        published_at=published_at,
        accepted_at=accepted_at,
        provider_available_at=provider_available_at,
        first_seen_at=first_seen_at,
        ingested_at=ingested_at,
        revised_at=None,
        superseded_at=None,
        decision_available_at=available_at,
        timezone="UTC",
        timestamp_precision=timestamp_precision,
        revision_number=revision_number,
        is_latest_version=True,
        PIT_eligibility=pit_status,
        PIT_blockers=tuple(blockers),
    )


def pit_access_status(record: PitRecord, decision_time: str) -> str:
    if record.PIT_eligibility not in {PitStatus.ELIGIBLE, PitStatus.DAILY_DELAY}:
        return str(record.PIT_eligibility)
    return str(record.PIT_eligibility) if record.available_for(decision_time) else str(PitStatus.FUTURE_LEAKAGE_BLOCKED)
