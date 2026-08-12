from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AIAuthority(StrEnum):
    DATA_ONLY = "DATA_ONLY"
    CONTEXT_ONLY = "CONTEXT_ONLY"
    FEATURE_ALLOWED = "FEATURE_ALLOWED"
    RANKING_ALLOWED = "RANKING_ALLOWED"
    PORTFOLIO_ADVISORY = "PORTFOLIO_ADVISORY"
    RISK_MODIFIER_ALLOWED = "RISK_MODIFIER_ALLOWED"
    SHADOW_ONLY = "SHADOW_ONLY"


class ModelLifecycle(StrEnum):
    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    CANDIDATE = "CANDIDATE"
    FORWARD_VALIDATION = "FORWARD_VALIDATION"
    VALIDATED = "VALIDATED"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"


class ExperimentStatus(StrEnum):
    PROPOSED = "PROPOSED"
    REJECTED_STAGE0 = "REJECTED_STAGE0"
    VALIDATING = "VALIDATING"
    FAILED = "FAILED"
    RESEARCH_VALIDATED = "RESEARCH_VALIDATED"
    FORWARD_REQUIRED = "FORWARD_REQUIRED"
    FORWARD_VALIDATED = "FORWARD_VALIDATED"


class DatasetSplit(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    FORWARD = "FORWARD"
    LIVE_SHADOW = "LIVE_SHADOW"


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class FeatureDefinition(StrictContract):
    schema_version: Literal["ai_feature_definition_v1"] = (
        "ai_feature_definition_v1"
    )
    feature_name: str
    source: str
    event_time_semantics: str
    available_at_semantics: str
    calculation_window: str
    revision_semantics: str
    missingness_semantics: str
    normalization_method: str
    universe_scope: str
    feature_version: str
    closed_bar_only: bool = True
    point_in_time_required: bool = True
    authority: AIAuthority = AIAuthority.FEATURE_ALLOWED


class ModelPrediction(StrictContract):
    schema_version: Literal["ai_model_prediction_v1"] = (
        "ai_model_prediction_v1"
    )
    prediction_id: str
    model_id: str
    model_version: str
    feature_set_id: str
    symbol: str
    timestamp: datetime
    available_at: datetime
    horizon: str
    prediction: float
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated_probability: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    uncertainty: float | None = Field(default=None, ge=0.0)
    regime: str | None = None
    data_hash: str
    training_manifest: str
    validation_manifest: str
    authority: AIAuthority = AIAuthority.SHADOW_ONLY
    money_control: Literal[False] = False
    execution_authority: Literal["NONE"] = "NONE"

    @field_validator("timestamp", "available_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timezone-aware timestamp required")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _causal_availability(self) -> ModelPrediction:
        if self.available_at < self.timestamp:
            raise ValueError("available_at cannot precede event timestamp")
        return self


class ModelEvidence(StrictContract):
    """Typed, non-authoritative model evidence for native decision consumers."""

    schema_version: Literal["ai_model_evidence_v1"] = "ai_model_evidence_v1"
    evidence_id: str
    model_id: str = "GLOBAL_DECISION_INTELLIGENCE"
    model_version: str
    symbol: str
    as_of: datetime
    feature_timestamp: datetime
    probability_positive_net: float = Field(ge=0.0, le=1.0)
    predicted_net_return: float
    expected_win: float
    expected_loss: float
    conservative_expected_value: float
    uncertainty: float = Field(ge=0.0, le=1.0)
    model_disagreement: float = Field(default=0.0, ge=0.0)
    return_interval_lower_90: float | None = None
    return_interval_upper_90: float | None = None
    cross_sectional_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    meta_take: bool
    abstained: bool
    out_of_distribution: bool
    validation_status: str
    tournament_hash: str
    feature_hash: str
    authority: Literal["SHADOW_ONLY"] = "SHADOW_ONLY"
    money_control: Literal[False] = False
    mutates_financial_fields: Literal[False] = False
    execution_authority: Literal["NONE"] = "NONE"

    @field_validator("as_of", "feature_timestamp")
    @classmethod
    def _evidence_timestamp_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timezone-aware timestamp required")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _feature_precedes_decision(self) -> ModelEvidence:
        if self.feature_timestamp > self.as_of:
            raise ValueError("feature timestamp cannot follow decision time")
        if (
            self.return_interval_lower_90 is not None
            and self.return_interval_upper_90 is not None
            and self.return_interval_lower_90 > self.return_interval_upper_90
        ):
            raise ValueError("return interval bounds are inverted")
        return self


class ResearchHypothesis(StrictContract):
    schema_version: Literal["ai_research_hypothesis_v1"] = (
        "ai_research_hypothesis_v1"
    )
    hypothesis_id: str
    source: str
    description: str
    economic_rationale: str
    feature_dependencies: tuple[str, ...]
    target: str
    horizon: str
    created_at: datetime
    status: ExperimentStatus
    duplicate_of: str | None = None
    execution_authority: Literal["NONE"] = "NONE"

    @field_validator("created_at")
    @classmethod
    def _created_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timezone-aware created_at required")
        return value.astimezone(UTC)


class NLPEvent(StrictContract):
    schema_version: Literal["ai_nlp_event_v1"] = "ai_nlp_event_v1"
    event_id: str
    source: str
    published_at: datetime
    available_at: datetime
    entities: tuple[str, ...]
    tickers: tuple[str, ...]
    event_type: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    expected_horizon: str
    source_quality: float = Field(ge=0.0, le=1.0)
    raw_hash: str
    authority: AIAuthority = AIAuthority.FEATURE_ALLOWED
    standalone_entry_allowed: Literal[False] = False
    execution_authority: Literal["NONE"] = "NONE"

    @field_validator("published_at", "available_at")
    @classmethod
    def _event_timestamp_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timezone-aware timestamp required")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _published_before_available(self) -> NLPEvent:
        if self.available_at < self.published_at:
            raise ValueError("NLP event cannot be available before publication")
        return self


class ModelRecord(StrictContract):
    schema_version: Literal["ai_model_record_v1"] = "ai_model_record_v1"
    model_id: str
    family: str
    version: str
    feature_set: tuple[str, ...]
    target: str
    training_interval: str
    validation_interval: str
    test_interval: str
    forward_interval: str
    universe: tuple[str, ...]
    horizon: str
    regime_scope: str
    data_hash: str
    code_hash: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    drift_limits: dict[str, float]
    authority: AIAuthority
    lifecycle: ModelLifecycle
    created_at: datetime
    expires_at: datetime
    incremental_evidence: str
    money_control: Literal[False] = False
    automatic_promotion: Literal[False] = False
    execution_authority: Literal["NONE"] = "NONE"

    @field_validator("created_at", "expires_at")
    @classmethod
    def _model_timestamp_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timezone-aware model timestamp required")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _expiry_after_creation(self) -> ModelRecord:
        if self.expires_at <= self.created_at:
            raise ValueError("model expires_at must follow created_at")
        return self


class ExperimentRecord(StrictContract):
    schema_version: Literal["ai_experiment_record_v1"] = (
        "ai_experiment_record_v1"
    )
    experiment_id: str
    hypothesis_id: str
    code_hash: str
    dataset_hash: str
    cutoff: datetime
    parameters: dict[str, Any]
    seed: int
    transaction_cost_model_version: str
    result_artifact: str
    decision: ExperimentStatus
    hypothesis_count_at_selection: int = Field(ge=1)
    multiple_testing: dict[str, Any]
    execution_authority: Literal["NONE"] = "NONE"
    automatic_live_promotion: Literal[False] = False

    @field_validator("cutoff")
    @classmethod
    def _cutoff_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timezone-aware cutoff required")
        return value.astimezone(UTC)


class AIPortfolioProposal(StrictContract):
    schema_version: Literal["ai_portfolio_proposal_v1"] = (
        "ai_portfolio_proposal_v1"
    )
    proposal_id: str
    model_id: str
    as_of: datetime
    target_weights: dict[str, float]
    expected_net_edge_lower_bound: float | None = None
    authority: Literal["PORTFOLIO_ADVISORY", "SHADOW_ONLY"] = "SHADOW_ONLY"
    native_translation_required: Literal[True] = True
    publishes_broker_quantity: Literal[False] = False
    money_control: Literal[False] = False
    execution_authority: Literal["NONE"] = "NONE"

    @field_validator("as_of")
    @classmethod
    def _proposal_timestamp_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timezone-aware as_of required")
        return value.astimezone(UTC)

    @field_validator("target_weights")
    @classmethod
    def _long_only_bounded_weights(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        if any(weight < 0 or weight > 1 for weight in value.values()):
            raise ValueError("AI proposal weights must be long-only and bounded")
        if sum(value.values()) > 1.0 + 1e-12:
            raise ValueError("AI proposal weights cannot exceed one")
        return value


FORBIDDEN_AI_POWERS = (
    "BROKER_WRITE",
    "ORDER_AUTHORITY",
    "CAPITAL_PROMOTION",
    "STRATEGY_PROMOTION",
    "RISK_LIMIT_EXPANSION",
    "DIRECT_QUANTITY_AUTHORITY",
)


__all__ = [
    "AIAuthority",
    "AIPortfolioProposal",
    "DatasetSplit",
    "ExperimentRecord",
    "ExperimentStatus",
    "FORBIDDEN_AI_POWERS",
    "FeatureDefinition",
    "ModelLifecycle",
    "ModelEvidence",
    "ModelPrediction",
    "ModelRecord",
    "NLPEvent",
    "ResearchHypothesis",
]
