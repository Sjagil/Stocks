from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NormalizedNewsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["normalized_news_event_v1"] = (
        "normalized_news_event_v1"
    )
    event_id: str
    story_cluster_id: str
    normalized_title_hash: str
    title: str
    source: str
    source_class: str
    published_at: datetime
    received_at: datetime
    link_hash: str | None = None
    symbols: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()
    industries: tuple[str, ...] = ()
    commodities: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()
    event_classes: tuple[str, ...]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_method: str
    relevance: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    raw_impact: float = Field(ge=-1.0, le=1.0)
    materiality: float = Field(ge=0.0, le=1.0)
    material: bool
    hard_risk_flags: tuple[str, ...] = ()
    classification_method: str
    entity_linking_method: str
    authority: Literal["RANKING_AND_RISK_CONTEXT_ONLY"] = (
        "RANKING_AND_RISK_CONTEXT_ONLY"
    )
    standalone_entry_allowed: Literal[False] = False
    strategy_authority: Literal["NONE"] = "NONE"
    execution_authority: Literal["NONE"] = "NONE"

    @field_validator("published_at", "received_at")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timezone-aware timestamp required")
        return value.astimezone(UTC)


class NewsStoryCluster(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["news_story_cluster_v1"] = (
        "news_story_cluster_v1"
    )
    story_cluster_id: str
    representative_event_id: str
    first_published_at: datetime
    last_published_at: datetime
    article_count: int = Field(ge=1)
    independent_source_count: int = Field(ge=1)
    sources: tuple[str, ...]
    title: str
    symbols: tuple[str, ...]
    sectors: tuple[str, ...]
    industries: tuple[str, ...]
    commodities: tuple[str, ...]
    event_classes: tuple[str, ...]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    raw_impact: float = Field(ge=-1.0, le=1.0)
    materiality: float = Field(ge=0.0, le=1.0)
    material: bool
    hard_risk_flags: tuple[str, ...]
    authority: Literal["RANKING_AND_RISK_CONTEXT_ONLY"] = (
        "RANKING_AND_RISK_CONTEXT_ONLY"
    )
    standalone_entry_allowed: Literal[False] = False
    execution_authority: Literal["NONE"] = "NONE"


__all__ = ["NewsStoryCluster", "NormalizedNewsEvent"]
