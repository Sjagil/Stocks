from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any


MACRO_AUTHORITY = {
    "macro_analysis_authority": "RESEARCH_ONLY",
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "paper_strategy_authority": "NONE",
    "live_strategy_authority": "NONE",
    "broker_calls": 0,
    "order_calls": 0,
}


class ScoreStatus(StrEnum):
    VALID = "VALID"
    PARTIAL = "PARTIAL"
    DATA_INCOMPLETE = "DATA_INCOMPLETE"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


class RegimeLabel(StrEnum):
    EXPANSION_DISINFLATION = "EXPANSION_DISINFLATION"
    EXPANSION_REFLATION = "EXPANSION_REFLATION"
    SLOWDOWN_DISINFLATION = "SLOWDOWN_DISINFLATION"
    SLOWDOWN_INFLATION = "SLOWDOWN_INFLATION"
    CONTRACTION = "CONTRACTION"
    STAGFLATION = "STAGFLATION"
    LIQUIDITY_EXPANSION = "LIQUIDITY_EXPANSION"
    LIQUIDITY_CONTRACTION = "LIQUIDITY_CONTRACTION"
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MacroSeriesSpec:
    canonical_id: str
    name: str
    category: str
    region: str
    frequency: str
    unit: str
    transformation: str
    release_lag_days: int
    revision_sensitive: bool
    direction: int
    minimum_history: int
    stale_days: int
    primary_source: str
    fallback_source: str | None
    provider_id: str | None
    vintage_capable: bool

    def validate(self) -> None:
        if self.frequency not in {"daily", "weekly", "monthly", "quarterly"}:
            raise ValueError(f"INVALID_MACRO_FREQUENCY:{self.canonical_id}")
        if self.direction not in {-1, 1}:
            raise ValueError(f"INVALID_MACRO_DIRECTION:{self.canonical_id}")
        if min(self.release_lag_days, self.minimum_history, self.stale_days) < 0:
            raise ValueError(f"INVALID_MACRO_SERIES_LIMIT:{self.canonical_id}")


@dataclass(frozen=True)
class MacroObservation:
    series_id: str
    observation_date: date
    publication_at: datetime
    available_at: datetime
    revision_status: str
    source: str
    provider: str
    original_value: float
    transformed_value: float | None
    frequency: str
    region: str
    vintage: str | None
    quality_status: str
    stale_status: str
    provider_payload_hash: str

    def __post_init__(self) -> None:
        if self.publication_at.tzinfo is None or self.available_at.tzinfo is None:
            raise ValueError("MACRO_TIMESTAMP_MUST_BE_TIMEZONE_AWARE")
        if self.available_at < self.publication_at:
            raise ValueError("AVAILABLE_AT_BEFORE_PUBLICATION")
        if not self.series_id:
            raise ValueError("series_id is required")

    @property
    def observation_id(self) -> str:
        identity = {
            "series_id": self.series_id,
            "observation_date": self.observation_date.isoformat(),
            "available_at": self.available_at.astimezone(UTC).isoformat(),
            "vintage": self.vintage,
            "revision_status": self.revision_status,
        }
        return f"MACRO-OBS-{stable_hash(identity)[:24]}"

    def payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "observation_date": self.observation_date.isoformat(),
            "publication_at": self.publication_at.astimezone(UTC).isoformat(),
            "available_at": self.available_at.astimezone(UTC).isoformat(),
            "observation_id": self.observation_id,
        }


@dataclass(frozen=True)
class MacroScore:
    name: str
    value: float | None
    confidence: float
    coverage: float
    status: str
    positive_contributions: tuple[dict[str, Any], ...]
    negative_contributions: tuple[dict[str, Any], ...]
    missing_inputs: tuple[str, ...]
    stale_inputs: tuple[str, ...]


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest().upper()
