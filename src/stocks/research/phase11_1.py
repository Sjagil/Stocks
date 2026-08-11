from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from stocks.alpha.data_contracts import (
    AlphaInputs,
    InstrumentType,
    PITDataStatus,
    PointInTimeFact,
    ShariahScreen,
    ShariahStatus,
    parse_timestamp,
)
from stocks.alpha.discovery import (
    MoverObservation,
    MoverType,
    NewsAttribution,
    classify_mover,
    closing_location_value,
)
from stocks.alpha.point_in_time import aggregate_pit_status
from stocks.alpha.portfolio.constraints import ALLOWED_INSTRUMENT_TYPES, BLOCKED_INSTRUMENT_TYPES, validate_shariah_screen
from stocks.alpha.portfolio.sizing import capital_preservation_rotation
from stocks.alpha.strategies import (
    earnings_revision_guidance_trend,
    fundamental_inflection_catalyst,
    quality_value_momentum_news_veto,
)
from stocks.data.phase5_common import sha256_file, utc_now_iso


PHASE11_1_MARKER = "PHASE11_1_ORTHOGONAL_PIT_DATA_AND_ALPHA_STRATEGY_FOUNDATION_GO"
PHASE11_1_FREEZE_MARKER = "PHASE11_1_ORTHOGONAL_PIT_DATA_AND_ALPHA_STRATEGY_FOUNDATION_FROZEN_GO"
FOUNDATION_STATUS = "FOUNDATION_ONLY_NO_PROVIDER_ALPHA_EVIDENCE"
FINANCIAL_STATUS = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "blocked",
    "PAPER_STRATEGY_AUTHORITY": "blocked",
    "LIVE_STRATEGY_AUTHORITY": "blocked",
    "financial_decision": "NO_NEW_FINANCIAL_CANDIDATE",
}
STRATEGY_IDS = (
    "EARNINGS_REVISION_GUIDANCE_TREND_V1",
    "FUNDAMENTAL_INFLECTION_NEWS_CATALYST_V1",
    "QUALITY_VALUE_MOMENTUM_NEWS_VETO_V1",
)
OVERLAY_IDS = (
    "MACRO_REGIME_EXPOSURE_OVERLAY_V1",
    "NEGATIVE_NEWS_RISK_OVERLAY_V1",
    "SHARIAH_ELIGIBILITY_GATE_V1",
    "CAPITAL_PRESERVATION_ROTATION_V1",
    "DAILY_MOVERS_DISCOVERY_ENGINE_V1",
)
ARTIFACTS = (
    "schema.json",
    "preregistration.json",
    "data-contracts.json",
    "pit-validation.json",
    "shariah-policy.json",
    "strategy-specs.json",
    "discovery-specs.json",
    "strategy-fixture-decisions.parquet",
    "mover-fixture-classifications.parquet",
    "hedge-rotation-audit.json",
    "authority-audit.json",
    "status.json",
    "manifest.json",
    "freeze-status.json",
)


@dataclass(frozen=True)
class Phase111Layout:
    project_root: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> Phase111Layout:
        return cls(project_root=project_root)

    @property
    def config_yaml(self) -> Path:
        return self.project_root / "config" / "research" / "phase11_1_alpha_foundation.yaml"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output" / "research" / "phase11_1"

    @property
    def status_md(self) -> Path:
        return self.project_root / "PHASE11_1_STATUS.md"

    @property
    def freeze_md(self) -> Path:
        return self.project_root / "PHASE11_1_FREEZE_REPORT.md"

    @property
    def docs_md(self) -> Path:
        return self.project_root / "docs" / "PHASE11_1_ORTHOGONAL_PIT_ALPHA_RESEARCH.md"

    def artifact(self, name: str) -> Path:
        return self.output_dir / name


def phase11_1_schema(project_root: Path | None = None) -> dict[str, Any]:
    payload = _artifact(
        "phase11_1_schema_v1",
        {
            "status": "OFFLINE_SCHEMA_ONLY",
            "technical_marker": PHASE11_1_MARKER,
            "freeze_marker": PHASE11_1_FREEZE_MARKER,
            "foundation_status": FOUNDATION_STATUS,
            "strategy_ids": list(STRATEGY_IDS),
            "overlay_ids": list(OVERLAY_IDS),
            "daily_movers_role": "RESEARCH_DISCOVERY_NOT_BUY_SIGNAL",
            "required_timestamp_contract": "available_at <= decision_timestamp",
            "asset_classes_allowed": sorted(item.value for item in ALLOWED_INSTRUMENT_TYPES),
            "asset_classes_blocked": sorted(item.value for item in BLOCKED_INSTRUMENT_TYPES),
            "cash_policy": {
                "minimum_cash_target": 0.025,
                "normal_cash_target": 0.05,
                "maximum_emergency_cash": 0.10,
            },
            "execution": _execution_contract(),
            "provider_calls": _provider_counters(),
            "financial_calls": _forbidden_counters(),
            **FINANCIAL_STATUS,
        },
    )
    if project_root is not None:
        layout = Phase111Layout.from_project_root(project_root)
        layout.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(layout.artifact("schema.json"), payload)
    return payload


def preregister_phase11_1(project_root: Path) -> dict[str, Any]:
    layout = Phase111Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    if not layout.config_yaml.exists():
        _write_text(layout.config_yaml, _default_config_yaml())
    config = _load_config(layout.config_yaml)
    errors = _validate_config(config)
    config_hash = _stable_hash(config)
    payload = _artifact(
        "phase11_1_preregistration_v1",
        {
            "status": "GO" if not errors else "NO_GO",
            "config_path": str(layout.config_yaml),
            "config_hash": config_hash,
            "validation_errors": errors,
            "immutable_research_rules": True,
            "strategy_ids": list(STRATEGY_IDS),
            "overlay_ids": list(OVERLAY_IDS),
            "config": config,
            **FINANCIAL_STATUS,
        },
        input_hashes={"config_yaml": sha256_file(layout.config_yaml)},
    )
    _write_json(layout.artifact("preregistration.json"), payload)
    return payload


def run_phase11_1_pipeline(project_root: Path) -> dict[str, Any]:
    layout = Phase111Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    phase11_1_schema(project_root)
    preregistration = preregister_phase11_1(project_root)
    config_hash = preregistration.get("config_hash")
    data_contracts = data_contracts_artifact(config_hash)
    pit = pit_validation_artifact(config_hash)
    shariah = shariah_policy_artifact(config_hash)
    specs = strategy_specs_artifact(config_hash)
    discovery = discovery_specs_artifact(config_hash)
    decisions = fixture_decision_rows()
    mover_rows = mover_fixture_rows()
    hedge = hedge_rotation_audit(config_hash)
    authority = authority_audit(config_hash)

    _write_json(layout.artifact("data-contracts.json"), data_contracts)
    _write_json(layout.artifact("pit-validation.json"), pit)
    _write_json(layout.artifact("shariah-policy.json"), shariah)
    _write_json(layout.artifact("strategy-specs.json"), specs)
    _write_json(layout.artifact("discovery-specs.json"), discovery)
    _write_parquet(layout.artifact("strategy-fixture-decisions.parquet"), decisions)
    _write_parquet(layout.artifact("mover-fixture-classifications.parquet"), mover_rows)
    _write_json(layout.artifact("hedge-rotation-audit.json"), hedge)
    _write_json(layout.artifact("authority-audit.json"), authority)
    _write_json(layout.artifact("manifest.json"), manifest_artifact(layout))
    status = phase11_1_status(project_root)
    _write_status_docs(layout, status)
    return status


def phase11_1_status(project_root: Path) -> dict[str, Any]:
    layout = Phase111Layout.from_project_root(project_root)
    artifact_status = {
        name: layout.artifact(name).exists()
        for name in ARTIFACTS
        if name not in {"status.json", "freeze-status.json"}
    }
    prereg = _read_json(layout.artifact("preregistration.json"))
    pit = _read_json(layout.artifact("pit-validation.json"))
    shariah = _read_json(layout.artifact("shariah-policy.json"))
    authority = _read_json(layout.artifact("authority-audit.json"))
    decisions_count = _parquet_count(layout.artifact("strategy-fixture-decisions.parquet"))
    mover_count = _parquet_count(layout.artifact("mover-fixture-classifications.parquet"))
    go = (
        all(artifact_status.values())
        and prereg.get("status") == "GO"
        and pit.get("status") == "GO"
        and shariah.get("status") == "GO"
        and authority.get("status") == "GO"
        and decisions_count >= 6
        and mover_count >= 4
    )
    payload = _artifact(
        "phase11_1_status_v1",
        {
            "status": PHASE11_1_MARKER if go else "NO_GO",
            "foundation_status": FOUNDATION_STATUS,
            "artifact_status": artifact_status,
            "fixture_decision_count": decisions_count,
            "mover_fixture_classification_count": mover_count,
            "pit_status": pit.get("status", "MISSING"),
            "shariah_policy_status": shariah.get("status", "MISSING"),
            "authority_status": authority.get("status", "MISSING"),
            "provider_data_status": "NOT_COLLECTED_OFFLINE_FOUNDATION",
            "alpha_evidence_status": "NOT_YET_TESTED",
            "next_gate": "PHASE11_2_PREREGISTERED_ALPHA_STRATEGIES_GO",
            "execution": _execution_contract(),
            "provider_calls": _provider_counters(),
            "financial_calls": _forbidden_counters(),
            **FINANCIAL_STATUS,
        },
        input_hashes=_artifact_hashes(layout),
    )
    _write_json(layout.artifact("status.json"), payload)
    return payload


def phase11_1_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase111Layout.from_project_root(project_root)
    status = phase11_1_status(project_root)
    freeze_status = PHASE11_1_FREEZE_MARKER if status["status"] == PHASE11_1_MARKER else "NO_GO"
    payload = _artifact(
        "phase11_1_freeze_status_v1",
        {
            "freeze_status": freeze_status,
            "phase11_1_status": status["status"],
            "frozen_files": _source_hashes(project_root),
            "artifact_hashes": _artifact_hashes(layout),
            "execution": _execution_contract(),
            "provider_calls": _provider_counters(),
            "financial_calls": _forbidden_counters(),
            **FINANCIAL_STATUS,
        },
        input_hashes=_artifact_hashes(layout),
    )
    _write_json(layout.artifact("freeze-status.json"), payload)
    _write_freeze_docs(layout, payload)
    return payload


def data_contracts_artifact(config_hash: str | None) -> dict[str, Any]:
    return _artifact(
        "phase11_1_data_contracts_v1",
        {
            "status": "GO",
            "point_in_time_fields": [
                "event_time",
                "published_at",
                "first_seen_at",
                "available_at",
                "ingested_at",
                "revised_at",
                "source",
                "source_hash",
                "reporting_period",
            ],
            "strategy_decision_fields": [
                "strategy_id",
                "feature_timestamp",
                "available_at",
                "universe_snapshot",
                "signal_calculation",
                "target_weights",
                "risk_constraints",
                "turnover",
                "cost_estimate",
                "rejection_reasons",
            ],
            "provider_connectors": {
                "sec_companyfacts": "NOT_CONFIGURED_OFFLINE_FOUNDATION",
                "sec_submissions": "NOT_CONFIGURED_OFFLINE_FOUNDATION",
                "eodhd_fundamentals": "NOT_CONFIGURED_OFFLINE_FOUNDATION",
                "eodhd_news": "NOT_CONFIGURED_OFFLINE_FOUNDATION",
                "eodhd_calendar": "NOT_CONFIGURED_OFFLINE_FOUNDATION",
                "macro_credit_volatility": "NOT_CONFIGURED_OFFLINE_FOUNDATION",
            },
            "config_hash": config_hash,
        },
    )


def pit_validation_artifact(config_hash: str | None) -> dict[str, Any]:
    decision_ts = parse_timestamp("2026-07-20T14:30:00+00:00")
    assert decision_ts is not None
    valid = PointInTimeFact.from_mapping(
        {
            "fact_id": "FACT_VALID_REVISION",
            "entity_id": "SIM_SHARIAH_STOCK",
            "field_name": "FY1_EPS_REVISION",
            "value": 0.12,
            "event_time": "2026-07-19T20:00:00+00:00",
            "published_at": "2026-07-19T20:05:00+00:00",
            "first_seen_at": "2026-07-19T20:06:00+00:00",
            "available_at": "2026-07-20T08:00:00+00:00",
            "ingested_at": "2026-07-20T08:01:00+00:00",
            "revised_at": None,
            "source": "OFFLINE_FIXTURE",
            "source_hash": "fixture-valid",
        }
    )
    future = PointInTimeFact.from_mapping(
        {
            "fact_id": "FACT_FUTURE_BLOCK",
            "entity_id": "SIM_SHARIAH_STOCK",
            "field_name": "GUIDANCE_EVENT",
            "value": 1,
            "event_time": "2026-07-20T21:00:00+00:00",
            "published_at": "2026-07-20T21:01:00+00:00",
            "first_seen_at": "2026-07-20T21:02:00+00:00",
            "available_at": "2026-07-20T21:02:00+00:00",
            "ingested_at": "2026-07-20T21:03:00+00:00",
            "revised_at": None,
            "source": "OFFLINE_FIXTURE",
            "source_hash": "fixture-future",
        }
    )
    rows = [
        {"fact_id": valid.fact_id, "status": aggregate_pit_status([valid], decision_ts).value},
        {"fact_id": future.fact_id, "status": aggregate_pit_status([future], decision_ts).value},
    ]
    return _artifact(
        "phase11_1_pit_validation_v1",
        {
            "status": "GO" if rows[0]["status"] == "PIT_VALID" and rows[1]["status"] == "PIT_FUTURE_DATED_BLOCKED" else "NO_GO",
            "decision_timestamp": decision_ts.isoformat(),
            "rows": rows,
            "hard_rule": "available_at <= decision_timestamp",
            "config_hash": config_hash,
        },
    )


def shariah_policy_artifact(config_hash: str | None) -> dict[str, Any]:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    good = ShariahScreen(
        instrument_id="SIM_SHARIAH_STOCK",
        instrument_type=InstrumentType.STOCK,
        compliance_status=ShariahStatus.ELIGIBLE,
        screening_methodology="SP_STYLE_CANONICAL_V1",
        methodology_version="2026.07",
        screened_at=now,
        financial_statement_available_at=now,
    )
    blocked = ShariahScreen(
        instrument_id="SIM_FUTURES_PRODUCT",
        instrument_type=InstrumentType.FUTURE,
        compliance_status=ShariahStatus.FUTURES_EXPOSURE_BLOCKED,
        screening_methodology="SP_STYLE_CANONICAL_V1",
        methodology_version="2026.07",
        screened_at=now,
        financial_statement_available_at=now,
        has_futures=True,
    )
    rows = [validate_shariah_screen(good), validate_shariah_screen(blocked)]
    return _artifact(
        "phase11_1_shariah_policy_v1",
        {
            "status": "GO" if rows[0]["status"] == "GO" and rows[1]["status"] == "NO_GO" else "NO_GO",
            "methodology": "SP_STYLE_CANONICAL_V1",
            "methodology_version": "2026.07",
            "rows": rows,
            "allowed": sorted(item.value for item in ALLOWED_INSTRUMENT_TYPES),
            "blocked": sorted(item.value for item in BLOCKED_INSTRUMENT_TYPES),
            "config_hash": config_hash,
        },
    )


def strategy_specs_artifact(config_hash: str | None) -> dict[str, Any]:
    return _artifact(
        "phase11_1_strategy_specs_v1",
        {
            "status": "GO",
            "strategies": [
                {
                    "strategy_id": "EARNINGS_REVISION_GUIDANCE_TREND_V1",
                    "entry": "NEXT_SESSION_OPEN",
                    "max_holding_days": 60,
                    "score_formula": "0.30R + 0.25Surprise + 0.20Guidance + 0.15Quality + 0.10Technical",
                },
                {
                    "strategy_id": "FUNDAMENTAL_INFLECTION_NEWS_CATALYST_V1",
                    "entry": "NEXT_SESSION_OPEN",
                    "required_catalyst": True,
                    "vetoes": ["BALANCE_SHEET_RISK", "NEGATIVE_NEWS_RISK"],
                },
                {
                    "strategy_id": "QUALITY_VALUE_MOMENTUM_NEWS_VETO_V1",
                    "entry": "NEXT_SESSION_OPEN",
                    "news_role": "VETO_AND_MINOR_SCALE",
                    "score_formula": "0.40Q + 0.30V + 0.30M",
                },
            ],
            "overlays": list(OVERLAY_IDS),
            "cash_policy": {"normal_cash_target": 0.05, "maximum_emergency_cash": 0.10},
            "config_hash": config_hash,
        },
    )


def discovery_specs_artifact(config_hash: str | None) -> dict[str, Any]:
    return _artifact(
        "phase11_1_daily_movers_discovery_specs_v1",
        {
            "status": "GO",
            "role": "DISCOVERY_ONLY_NOT_AUTOMATIC_BUY_SIGNAL",
            "scan_families": [
                "TOP_PERCENT_GAINERS",
                "TOP_PERCENT_LOSERS",
                "TOP_ABSOLUTE_VOLUME",
                "TOP_RELATIVE_VOLUME",
                "GAP_UP",
                "GAP_DOWN",
                "NEW_20D_HIGH",
                "NEW_60D_HIGH",
                "NEW_52W_HIGH",
                "NEW_20D_LOW",
                "NEW_52W_LOW",
                "PERSISTENT_5D_20D_60D_120D_LEADERS",
            ],
            "funnel_order": [
                "SHARIAH_ELIGIBILITY_GATE",
                "LIQUIDITY_AND_QUALITY_FILTER",
                "MOVER_SCAN",
                "NEWS_EVENT_CLASSIFICATION",
                "FUNDAMENTAL_ANALYSIS",
                "TECHNICAL_MARKET_ACCEPTANCE",
                "WATCHLIST_OR_REJECTION",
            ],
            "candidate_states": [
                "DISCOVERED",
                "NEWS_PENDING",
                "EVENT_VALIDATED",
                "SHARIAH_VALIDATED",
                "FUNDAMENTAL_REVIEW",
                "TECHNICAL_CONFIRMATION_PENDING",
                "WATCHLIST",
                "ENTRY_READY",
                "POSITION_ACTIVE",
                "ADD_ALLOWED",
                "REDUCE",
                "EXIT",
                "REJECTED_PUMP",
                "REJECTED_VALUE_TRAP",
                "REJECTED_SHARIAH",
                "REJECTED_PERMANENT_IMPAIRMENT",
            ],
            "portfolio_budget": {
                "core_fundamental_portfolio": "50-65%",
                "emerging_gems": "15-25%",
                "event_driven_movers": "10-15%",
                "gold_energy_material_hedges": "10-25%",
                "operational_cash": "2.5-5%",
            },
            "single_new_mover_starter_max": 0.03,
            "all_event_driven_movers_max": 0.15,
            "top_losers_combined_max": 0.10,
            "closing_location_value_formula": "((Close-Low)-(High-Close))/(High-Low)",
            "config_hash": config_hash,
        },
    )


def fixture_decision_rows() -> list[dict[str, Any]]:
    decision_ts = datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
    good = AlphaInputs(
        instrument_id="SIM_SHARIAH_STOCK",
        decision_timestamp=decision_ts,
        pit_status=PITDataStatus.VALID,
        shariah_status=ShariahStatus.ELIGIBLE,
        quality_score=0.75,
        value_score=0.62,
        revision_score=0.80,
        earnings_surprise_score=0.70,
        guidance_event_score=0.72,
        catalyst_score=0.65,
        technical_confirmation_score=0.75,
        macro_regime_multiplier=0.90,
        volatility=0.18,
        metadata={"momentum_score": 0.68},
    )
    future_data = AlphaInputs(
        instrument_id="SIM_FUTURE_DATED_DATA",
        decision_timestamp=decision_ts,
        pit_status=PITDataStatus.FUTURE_DATED_BLOCKED,
        shariah_status=ShariahStatus.ELIGIBLE,
    )
    blocked_shariah = AlphaInputs(
        instrument_id="SIM_BLOCKED_FUTURES_PRODUCT",
        decision_timestamp=decision_ts,
        pit_status=PITDataStatus.VALID,
        shariah_status=ShariahStatus.FUTURES_EXPOSURE_BLOCKED,
    )
    negative_news = AlphaInputs(
        instrument_id="SIM_NEGATIVE_NEWS",
        decision_timestamp=decision_ts,
        pit_status=PITDataStatus.VALID,
        shariah_status=ShariahStatus.ELIGIBLE,
        negative_news_score=0.90,
    )
    rows = []
    for inputs in (good, future_data, blocked_shariah, negative_news):
        for decide in (
            earnings_revision_guidance_trend,
            fundamental_inflection_catalyst,
            quality_value_momentum_news_veto,
        ):
            rows.append(decide(inputs).as_dict())
    return rows


def mover_fixture_rows() -> list[dict[str, Any]]:
    rows = [
        classify_mover(
            MoverObservation(
                instrument_id="SIM_STRUCTURAL_GAINER",
                mover_type=MoverType.TOP_GAINER,
                shariah_status=ShariahStatus.ELIGIBLE,
                news_attribution=NewsAttribution.CONFIRMED_COMPANY_EVENT,
                event_quality=0.90,
                earnings_revision_score=0.80,
                fundamental_inflection=0.75,
                volume_confirmation=0.80,
                gap_retention=0.75,
                sector_relative_strength=0.70,
                valuation_room=0.65,
            )
        ),
        classify_mover(
            MoverObservation(
                instrument_id="SIM_UNEXPLAINED_PUMP",
                mover_type=MoverType.TOP_GAINER,
                shariah_status=ShariahStatus.ELIGIBLE,
                news_attribution=NewsAttribution.NO_MATERIAL_NEWS_FOUND,
                event_quality=0.10,
                volume_confirmation=0.80,
                low_quality_pump_risk=True,
            )
        ),
        classify_mover(
            MoverObservation(
                instrument_id="SIM_FALLEN_ANGEL",
                mover_type=MoverType.TOP_LOSER,
                shariah_status=ShariahStatus.ELIGIBLE,
                news_attribution=NewsAttribution.CONFIRMED_COMPANY_EVENT,
                business_quality=0.75,
                balance_sheet_strength=0.80,
                temporary_shock_probability=0.75,
                valuation_reset=0.70,
                revision_stabilization=0.60,
                technical_stabilization=0.55,
                insider_confirmation=0.40,
            )
        ),
        classify_mover(
            MoverObservation(
                instrument_id="SIM_PERSISTENT_GEM",
                mover_type=MoverType.PERSISTENT_LEADER,
                shariah_status=ShariahStatus.ELIGIBLE,
                news_attribution=NewsAttribution.MULTIPLE_EVENTS,
                revenue_acceleration=0.80,
                margin_expansion=0.75,
                fcf_improvement=0.70,
                earnings_revision_score=0.80,
                relative_strength=0.85,
                positive_weeks=0.80,
                higher_lows=0.75,
                accumulation_volume=0.70,
                revision_persistence=0.75,
                repeated_positive_events=0.80,
                business_quality=0.70,
                valuation_room=0.60,
            )
        ),
        classify_mover(
            MoverObservation(
                instrument_id="SIM_REJECTED_SHARIAH",
                mover_type=MoverType.TOP_GAINER,
                shariah_status=ShariahStatus.INELIGIBLE,
                news_attribution=NewsAttribution.CONFIRMED_COMPANY_EVENT,
            )
        ),
    ]
    for row in rows:
        row["example_closing_location_value"] = round(closing_location_value(110.0, 100.0, 108.0), 6)
    return rows


def hedge_rotation_audit(config_hash: str | None) -> dict[str, Any]:
    inflationary = capital_preservation_rotation(
        {
            "PHYSICAL_GOLD": 0.72,
            "SHARIAH_ENERGY_EQUITIES": 0.78,
            "SHARIAH_DEFENSIVE_EQUITIES": 0.40,
        },
        hedge_budget=0.30,
    )
    no_fit = capital_preservation_rotation(
        {
            "PHYSICAL_GOLD": 0.30,
            "SHARIAH_ENERGY_EQUITIES": 0.20,
            "SHARIAH_DEFENSIVE_EQUITIES": 0.45,
        },
        hedge_budget=0.30,
    )
    return _artifact(
        "phase11_1_hedge_rotation_audit_v1",
        {
            "status": "GO" if "OPERATIONAL_CASH" not in inflationary and no_fit.get("OPERATIONAL_CASH", 0.0) <= 0.10 else "NO_GO",
            "inflationary_supply_shock_rotation": inflationary,
            "no_eligible_hedge_rotation": no_fit,
            "cash_is_limited_fallback": True,
            "blocked_hedges": ["BONDS", "FUTURES", "COMMODITY_FUTURES_ETFS", "SYNTHETIC_COMMODITY_ETFS"],
            "config_hash": config_hash,
        },
    )


def authority_audit(config_hash: str | None) -> dict[str, Any]:
    return _artifact(
        "phase11_1_authority_audit_v1",
        {
            "status": "GO",
            "strategy_authority": "NONE",
            "shadow_authority": "NONE",
            "execution_authority": "NONE",
            "broker_observation_authority": "READ_ONLY",
            "attempt_SHADOW": "AUTHORITY_NOT_GRANTED",
            "attempt_PAPER": "AUTHORITY_NOT_GRANTED",
            "attempt_LIVE": "AUTHORITY_NOT_GRANTED",
            "automatic_strategy_activation": 0,
            "broker_write_attempts": 0,
            "provider_calls": _provider_counters(),
            "financial_calls": _forbidden_counters(),
            "config_hash": config_hash,
        },
    )


def manifest_artifact(layout: Phase111Layout) -> dict[str, Any]:
    return _artifact(
        "phase11_1_manifest_v1",
        {
            "status": "GO",
            "artifact_paths": {name: str(layout.artifact(name)) for name in ARTIFACTS},
            "private_data_paths": [],
            "execution": _execution_contract(),
            "provider_calls": _provider_counters(),
            "financial_calls": _forbidden_counters(),
            **FINANCIAL_STATUS,
        },
        input_hashes=_artifact_hashes(layout),
    )


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> list[str]:
    errors = []
    if config.get("phase") != "PHASE11_1_ORTHOGONAL_PIT_DATA_AND_ALPHA_STRATEGY_FOUNDATION":
        errors.append("phase identifier mismatch")
    if tuple(config.get("strategies", [])) != STRATEGY_IDS:
        errors.append("strategy preregistration mismatch")
    if config.get("authority", {}).get("execution_authority") != "NONE":
        errors.append("execution authority must remain NONE")
    if config.get("authority", {}).get("shadow_authority") != "NONE":
        errors.append("shadow authority must remain NONE")
    blocked = set(config.get("blocked_asset_classes", []))
    if "FUTURE" not in blocked or "BOND" not in blocked:
        errors.append("blocked asset classes must include FUTURE and BOND")
    if config.get("pit_rule") != "available_at <= decision_timestamp":
        errors.append("PIT rule must be available_at <= decision_timestamp")
    return errors


def _default_config_yaml() -> str:
    return """phase: PHASE11_1_ORTHOGONAL_PIT_DATA_AND_ALPHA_STRATEGY_FOUNDATION
pit_rule: available_at <= decision_timestamp
strategies:
  - EARNINGS_REVISION_GUIDANCE_TREND_V1
  - FUNDAMENTAL_INFLECTION_NEWS_CATALYST_V1
  - QUALITY_VALUE_MOMENTUM_NEWS_VETO_V1
overlays:
  - MACRO_REGIME_EXPOSURE_OVERLAY_V1
  - NEGATIVE_NEWS_RISK_OVERLAY_V1
  - SHARIAH_ELIGIBILITY_GATE_V1
  - CAPITAL_PRESERVATION_ROTATION_V1
  - DAILY_MOVERS_DISCOVERY_ENGINE_V1
allowed_asset_classes:
  - STOCK
  - SHARIAH_EQUITY_ETF
  - APPROVED_PHYSICAL_COMMODITY_PRODUCT
blocked_asset_classes:
  - BOND
  - CONVENTIONAL_FIXED_INCOME
  - FUTURE
  - OPTION
  - SWAP
  - CFD
  - SHORT
  - MARGIN
  - LEVERAGED_ETF
  - INVERSE_ETF
  - SYNTHETIC_ETF
cash_policy:
  minimum_cash_target: 0.025
  normal_cash_target: 0.05
  maximum_emergency_cash: 0.10
authority:
  strategy_authority: NONE
  shadow_authority: NONE
  execution_authority: NONE
  broker_observation_authority: READ_ONLY
"""


def _source_hashes(project_root: Path) -> dict[str, str | None]:
    paths = [
        "main.py",
        "src/stocks/research/phase11_1.py",
        "src/stocks/alpha/data_contracts.py",
        "src/stocks/alpha/discovery.py",
        "src/stocks/alpha/point_in_time.py",
        "src/stocks/alpha/normalization.py",
        "src/stocks/alpha/portfolio/constraints.py",
        "src/stocks/alpha/portfolio/sizing.py",
        "src/stocks/alpha/strategies/common.py",
        "src/stocks/alpha/strategies/earnings_revision_guidance_trend.py",
        "src/stocks/alpha/strategies/fundamental_inflection_catalyst.py",
        "src/stocks/alpha/strategies/quality_value_momentum_news_veto.py",
        "tests/test_phase11_1_alpha_foundation.py",
    ]
    return {path: sha256_file(project_root / path) for path in paths}


def _artifact_hashes(layout: Phase111Layout) -> dict[str, str | None]:
    return {
        name: sha256_file(layout.artifact(name))
        for name in ARTIFACTS
        if name != "freeze-status.json" and layout.artifact(name).exists()
    }


def _parquet_count(path: Path) -> int:
    if not path.exists():
        return 0
    return pq.read_table(path).num_rows


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _artifact(schema: str, payload: dict[str, Any], input_hashes: dict[str, str | None] | None = None) -> dict[str, Any]:
    body = {
        "schema": schema,
        "created_at": utc_now_iso(),
        **payload,
        "input_hashes": input_hashes or {},
    }
    body["content_hash"] = _stable_hash({key: value for key, value in body.items() if key not in {"created_at", "content_hash"}})
    return body


def _execution_contract() -> dict[str, object]:
    return {
        "strategy_authority": "NONE",
        "shadow_authority": "NONE",
        "execution_authority": "NONE",
        "orders_enabled": False,
        "broker_writes_enabled": False,
        "strategy_activation_enabled": False,
    }


def _provider_counters() -> dict[str, int]:
    return {
        "sec_companyfacts_calls": 0,
        "sec_submissions_calls": 0,
        "eodhd_fundamentals_calls": 0,
        "eodhd_news_calls": 0,
        "eodhd_calendar_calls": 0,
        "macro_provider_calls": 0,
    }


def _forbidden_counters() -> dict[str, int]:
    return {
        "place_order_calls": 0,
        "cancel_order_calls": 0,
        "global_cancel_calls": 0,
        "request_order_id_calls": 0,
        "auto_bind_order_calls": 0,
        "exercise_option_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "account_calls": 0,
    }


def _write_status_docs(layout: Phase111Layout, status: dict[str, Any]) -> None:
    _write_text(
        layout.status_md,
        "\n".join(
            [
                "# Phase 11.1 Status",
                "",
                f"status: {status['status']}",
                f"foundation_status: {status['foundation_status']}",
                f"financial_decision: {status['financial_decision']}",
                f"fixture_decision_count: {status['fixture_decision_count']}",
                "execution_authority: NONE",
                "strategy_authority: NONE",
                "provider_data_status: NOT_COLLECTED_OFFLINE_FOUNDATION",
                "",
            ]
        ),
    )
    _write_text(
        layout.docs_md,
        "\n".join(
            [
                "# Phase 11.1 Orthogonal PIT Alpha Research",
                "",
                "This phase defines the offline point-in-time alpha foundation.",
                "It does not collect provider data, activate strategies, or send broker orders.",
                "",
                "Implemented contracts:",
                "- point-in-time availability gate",
                "- Shariah-first long-only universe gate",
                "- three preregistered alpha strategy specifications",
                "- daily movers discovery as research-only candidate funnel",
                "- macro and negative-news overlays",
                "- capital preservation rotation with limited cash fallback",
                "",
            ]
        ),
    )


def _write_freeze_docs(layout: Phase111Layout, freeze: dict[str, Any]) -> None:
    _write_text(
        layout.freeze_md,
        "\n".join(
            [
                "# Phase 11.1 Freeze Report",
                "",
                f"freeze_status: {freeze['freeze_status']}",
                f"phase11_1_status: {freeze['phase11_1_status']}",
                f"financial_decision: {freeze['financial_decision']}",
                "execution_authority: NONE",
                "provider_calls: 0",
                "broker_write_attempts: 0",
                "",
            ]
        ),
    )
