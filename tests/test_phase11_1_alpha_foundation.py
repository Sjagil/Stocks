from __future__ import annotations

import json
from datetime import UTC, datetime

import pyarrow.parquet as pq

import main
from stocks.alpha.data_contracts import (
    AlphaInputs,
    InstrumentType,
    PITDataStatus,
    PointInTimeFact,
    ShariahScreen,
    ShariahStatus,
)
from stocks.alpha.discovery import (
    MoverObservation,
    MoverType,
    NewsAttribution,
    classify_mover,
    closing_location_value,
)
from stocks.alpha.point_in_time import aggregate_pit_status
from stocks.alpha.portfolio.constraints import validate_shariah_screen
from stocks.alpha.portfolio.sizing import capital_preservation_rotation
from stocks.alpha.strategies.earnings_revision_guidance_trend import decide as decide_earnings_revision
from stocks.research.phase11_1 import (
    PHASE11_1_FREEZE_MARKER,
    PHASE11_1_MARKER,
    Phase111Layout,
    fixture_decision_rows,
    mover_fixture_rows,
    phase11_1_freeze,
    phase11_1_schema,
    run_phase11_1_pipeline,
)


def test_phase11_1_schema_keeps_financial_and_execution_authority_blocked() -> None:
    schema = phase11_1_schema()

    assert schema["technical_marker"] == PHASE11_1_MARKER
    assert schema["execution"]["execution_authority"] == "NONE"
    assert schema["FINANCIAL_FINALIST_GO"] is False
    assert schema["PAPER_STRATEGY_AUTHORITY"] == "blocked"
    assert schema["financial_calls"]["place_order_calls"] == 0
    assert "FUTURE" in schema["asset_classes_blocked"]


def test_phase11_1_cli_schema(capsys) -> None:
    exit_code = main.main(["research", "phase11-1", "schema"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "phase11_1_schema_v1"
    assert payload["provider_calls"]["eodhd_news_calls"] == 0


def test_pit_available_at_after_decision_is_blocked() -> None:
    decision_ts = datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
    future_fact = PointInTimeFact.from_mapping(
        {
            "fact_id": "FUTURE",
            "entity_id": "AAA",
            "field_name": "eps_revision",
            "value": 1,
            "event_time": "2026-07-20T20:00:00+00:00",
            "published_at": "2026-07-20T20:00:00+00:00",
            "first_seen_at": "2026-07-20T20:00:00+00:00",
            "available_at": "2026-07-20T20:00:00+00:00",
            "ingested_at": "2026-07-20T20:01:00+00:00",
            "revised_at": None,
            "source": "fixture",
            "source_hash": "hash",
        }
    )

    assert aggregate_pit_status([future_fact], decision_ts) == PITDataStatus.FUTURE_DATED_BLOCKED


def test_shariah_gate_blocks_futures_derivatives_and_interest() -> None:
    screen = ShariahScreen(
        instrument_id="BAD_FUT",
        instrument_type=InstrumentType.FUTURE,
        compliance_status=ShariahStatus.FUTURES_EXPOSURE_BLOCKED,
        screening_methodology="SP_STYLE_CANONICAL_V1",
        methodology_version="2026.07",
        screened_at=datetime(2026, 7, 20, tzinfo=UTC),
        financial_statement_available_at=datetime(2026, 7, 20, tzinfo=UTC),
        has_futures=True,
        has_derivatives=True,
        has_interest_bearing_cash=True,
    )

    result = validate_shariah_screen(screen)

    assert result["status"] == "NO_GO"
    assert "BLOCKED_ASSET_CLASS_FUTURE" in result["rejection_reasons"]
    assert "DERIVATIVE_EXPOSURE_BLOCKED" in result["rejection_reasons"]
    assert "INTEREST_EXPOSURE_BLOCKED" in result["rejection_reasons"]


def test_earnings_revision_strategy_blocks_data_and_enters_only_on_valid_fixture() -> None:
    decision_ts = datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
    valid = AlphaInputs(
        instrument_id="GOOD",
        decision_timestamp=decision_ts,
        pit_status=PITDataStatus.VALID,
        shariah_status=ShariahStatus.ELIGIBLE,
        quality_score=0.80,
        revision_score=0.85,
        earnings_surprise_score=0.75,
        guidance_event_score=0.75,
        catalyst_score=0.75,
        technical_confirmation_score=0.80,
    )
    blocked = AlphaInputs(
        instrument_id="FUTURE_DATA",
        decision_timestamp=decision_ts,
        pit_status=PITDataStatus.FUTURE_DATED_BLOCKED,
        shariah_status=ShariahStatus.ELIGIBLE,
    )

    assert decide_earnings_revision(valid).status.value == "ENTRY_READY"
    assert decide_earnings_revision(valid).target_weight > 0
    assert decide_earnings_revision(blocked).status.value == "BLOCKED_DATA"


def test_negative_news_overlay_forces_risk_exit() -> None:
    decision_ts = datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
    inputs = AlphaInputs(
        instrument_id="BAD_NEWS",
        decision_timestamp=decision_ts,
        pit_status=PITDataStatus.VALID,
        shariah_status=ShariahStatus.ELIGIBLE,
        negative_news_score=0.90,
    )

    result = decide_earnings_revision(inputs)

    assert result.status.value == "EXIT_RISK_EVENT"
    assert result.target_weight == 0


def test_capital_preservation_rotation_uses_limited_cash_only_when_no_hedge_fits() -> None:
    selected = capital_preservation_rotation({"PHYSICAL_GOLD": 0.70, "SHARIAH_ENERGY_EQUITIES": 0.80}, 0.30)
    fallback = capital_preservation_rotation({"PHYSICAL_GOLD": 0.20, "SHARIAH_ENERGY_EQUITIES": 0.10}, 0.30)

    assert selected == {"SHARIAH_ENERGY_EQUITIES": 0.18, "PHYSICAL_GOLD": 0.12}
    assert fallback == {"OPERATIONAL_CASH": 0.10}


def test_fixture_decisions_cover_entry_data_shariah_and_news_blocks() -> None:
    statuses = {row["status"] for row in fixture_decision_rows()}

    assert "ENTRY_READY" in statuses
    assert "BLOCKED_DATA" in statuses
    assert "BLOCKED_SHARIAH" in statuses
    assert "EXIT_RISK_EVENT" in statuses


def test_daily_movers_are_discovery_not_automatic_buy_signals() -> None:
    structural = classify_mover(
        MoverObservation(
            instrument_id="GOOD_GAINER",
            mover_type=MoverType.TOP_GAINER,
            shariah_status=ShariahStatus.ELIGIBLE,
            news_attribution=NewsAttribution.CONFIRMED_COMPANY_EVENT,
            event_quality=0.90,
            earnings_revision_score=0.85,
            fundamental_inflection=0.80,
            volume_confirmation=0.80,
            gap_retention=0.80,
            sector_relative_strength=0.70,
            valuation_room=0.70,
        )
    )
    pump = classify_mover(
        MoverObservation(
            instrument_id="PUMP",
            mover_type=MoverType.TOP_GAINER,
            shariah_status=ShariahStatus.ELIGIBLE,
            news_attribution=NewsAttribution.NO_MATERIAL_NEWS_FOUND,
            low_quality_pump_risk=True,
        )
    )
    blocked = classify_mover(
        MoverObservation(
            instrument_id="BAD_SHARIAH",
            mover_type=MoverType.PERSISTENT_LEADER,
            shariah_status=ShariahStatus.INELIGIBLE,
            news_attribution=NewsAttribution.MULTIPLE_EVENTS,
        )
    )

    assert structural["candidate_state"] == "ENTRY_READY"
    assert structural["automatic_buy_signal"] is False
    assert pump["candidate_state"] == "REJECTED_PUMP"
    assert blocked["candidate_state"] == "REJECTED_SHARIAH"
    assert closing_location_value(110, 100, 108) == 0.6


def test_mover_fixture_rows_cover_gainer_loser_gem_and_rejections() -> None:
    rows = mover_fixture_rows()
    states = {row["candidate_state"] for row in rows}
    classes = {row["classification"] for row in rows}

    assert "ENTRY_READY" in states
    assert "WATCHLIST" in states
    assert "REJECTED_PUMP" in states
    assert "REJECTED_SHARIAH" in states
    assert "L1_TRANSIENT_OVERREACTION" in classes


def test_phase11_1_run_and_freeze_publish_artifacts(tmp_path) -> None:
    status = run_phase11_1_pipeline(tmp_path)
    freeze = phase11_1_freeze(tmp_path)
    layout = Phase111Layout.from_project_root(tmp_path)

    assert status["status"] == PHASE11_1_MARKER
    assert freeze["freeze_status"] == PHASE11_1_FREEZE_MARKER
    assert layout.artifact("strategy-fixture-decisions.parquet").exists()
    assert layout.artifact("mover-fixture-classifications.parquet").exists()
    assert pq.read_table(layout.artifact("strategy-fixture-decisions.parquet")).num_rows == 12
    assert pq.read_table(layout.artifact("mover-fixture-classifications.parquet")).num_rows == 5
    assert json.loads(layout.artifact("authority-audit.json").read_text(encoding="utf-8"))["status"] == "GO"
