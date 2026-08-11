from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock

from stocks.macro.analyst import deterministic_analysis, render_markdown
from stocks.macro.config import MacroConfig
from stocks.macro.contracts import MacroObservation, MacroScore
from stocks.macro.engine import (
    apply_hysteresis,
    apply_macro_exposure,
    classify_regime,
    compare_macro_variant,
    compute_macro_snapshot,
    compute_score,
    exposure_multiplier,
)
from stocks.macro.service import (
    _parse_ecb_calendar_html,
    _publish_immutable,
    macro_conflicts,
    macro_update,
)
from stocks.macro.storage import MacroLayout, MacroStore
from stocks.macro.transforms import (
    build_feature_snapshot,
    fx_normalize,
    inflation_adjust,
    point_in_time_series,
    transform_series,
)
from stocks.research.autopilot.components import MACRO_COMPONENT_NAMES
from stocks.research.autopilot.engine import deterministic_fixture, run_backtest
from stocks.research.autopilot.generator import (
    generate_macro_variant,
    generate_strategies,
    validate_strategy,
)
from stocks.screener.models import (
    AssetMetadata,
    AssetSnapshot,
    ShariahSnapshot,
)
from stocks.screener.scoring import _macro_context_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def macro_config() -> MacroConfig:
    return MacroConfig.load(PROJECT_ROOT)


@pytest.fixture
def macro_observations(
    macro_config: MacroConfig,
) -> list[dict[str, object]]:
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    result: list[dict[str, object]] = []
    for number, spec in enumerate(macro_config.series.values()):
        count = max(spec.minimum_history + 15, 80)
        if spec.frequency == "daily":
            dates = pd.bdate_range(end="2026-06-29", periods=count)
        elif spec.frequency == "weekly":
            dates = pd.date_range(end="2026-06-20", periods=count, freq="W")
        elif spec.frequency == "quarterly":
            dates = pd.date_range(end="2026-03-31", periods=count, freq="QE")
        else:
            dates = pd.date_range(end="2026-05-31", periods=count, freq="ME")
        for index, timestamp in enumerate(dates):
            if spec.transformation == "level_50":
                base = 50.0
            elif spec.transformation == "level_20":
                base = 20.0
            else:
                base = 100.0 + number
            value = base + spec.direction * index * 0.15
            available_at = as_of - timedelta(days=count - index)
            observation = MacroObservation(
                series_id=spec.canonical_id,
                observation_date=timestamp.date(),
                publication_at=available_at,
                available_at=available_at,
                revision_status="HISTORICAL_VINTAGE",
                source="FIXTURE",
                provider="OFFLINE_TEST",
                original_value=value,
                transformed_value=None,
                frequency=spec.frequency,
                region=spec.region,
                vintage=available_at.date().isoformat(),
                quality_status="FIXTURE_ONLY",
                stale_status="CURRENT",
                provider_payload_hash=f"FIX-{number}-{index}",
            )
            result.append(observation.payload())
    return result


def test_registry_is_versioned_and_covers_core_v1(
    macro_config: MacroConfig,
) -> None:
    assert macro_config.version == "MACRO_ENGINE_CONFIG_V1"
    assert len(macro_config.series) >= 40
    assert {
        "US_CPI",
        "US_CORE_PCE",
        "US_UNEMPLOYMENT",
        "US_M2",
        "US_HIGH_YIELD_SPREAD",
        "EU_CPI",
        "EURUSD",
        "VIX",
        "GOLD",
        "OIL",
        "COPPER",
        "EQUITY_BREADTH_GLOBAL",
    } <= set(macro_config.series)
    assert len(macro_config.score_weights) == 15


def test_point_in_time_revision_excludes_future_release() -> None:
    first = _observation(value=100.0, available="2025-02-01T12:00:00Z")
    revision = _observation(
        value=110.0,
        available="2025-03-01T12:00:00Z",
        revision="REVISED",
    )
    frame = point_in_time_series(
        [first.payload(), revision.payload()],
        series_id="TEST",
        as_of=datetime(2025, 2, 15, tzinfo=UTC),
    )
    assert float(frame.iloc[-1]["original_value"]) == 100.0
    later = point_in_time_series(
        [first.payload(), revision.payload()],
        series_id="TEST",
        as_of=datetime(2025, 3, 2, tzinfo=UTC),
    )
    assert float(later.iloc[-1]["original_value"]) == 110.0


def test_raw_market_close_supersedes_legacy_adjusted_close() -> None:
    legacy = _observation(value=100.0)
    raw = replace(
        legacy,
        original_value=101.0,
        revision_status="MARKET_CLOSE_RAW_V1",
        quality_status="MARKET_CLOSE_RAW_UNADJUSTED_V1",
        provider_payload_hash="RAW",
    )
    legacy = replace(
        legacy,
        revision_status="MARKET_CLOSE_FINAL",
        quality_status="MARKET_CLOSE",
        provider_payload_hash="ADJUSTED",
    )
    frame = point_in_time_series(
        [raw.payload(), legacy.payload()],
        series_id="TEST",
        as_of=datetime(2025, 2, 2, tzinfo=UTC),
    )
    assert float(frame.iloc[-1]["original_value"]) == 101.0
    assert frame.iloc[-1]["quality_status"] == (
        "MARKET_CLOSE_RAW_UNADJUSTED_V1"
    )


def test_available_at_before_publication_is_blocked() -> None:
    with pytest.raises(ValueError, match="AVAILABLE_AT_BEFORE_PUBLICATION"):
        _observation(
            value=1.0,
            available="2025-01-01T00:00:00Z",
            publication="2025-01-02T00:00:00Z",
        )


def test_legacy_period_start_release_estimate_is_quarantined(
    macro_config: MacroConfig,
) -> None:
    legacy = MacroObservation(
        series_id="US_CPI",
        observation_date=date(2025, 1, 1),
        publication_at=datetime(2025, 1, 15, tzinfo=UTC),
        available_at=datetime(2025, 1, 15, tzinfo=UTC),
        revision_status="LATEST_RELEASE",
        source="FRED",
        provider="FRED",
        original_value=100.0,
        transformed_value=None,
        frequency="monthly",
        region="US",
        vintage="2025-01-15",
        quality_status="ESTIMATED_RELEASE_LAG",
        stale_status="NOT_EVALUATED_AT_INGEST",
        provider_payload_hash="LEGACY",
    )
    conservative = replace(
        legacy,
        publication_at=datetime(2025, 2, 14, tzinfo=UTC),
        available_at=datetime(2025, 2, 14, tzinfo=UTC),
        vintage="2025-02-14",
        quality_status="CONSERVATIVE_PERIOD_END_RELEASE_LAG_V1",
        provider_payload_hash="CONSERVATIVE",
    )
    legacy_feature = build_feature_snapshot(
        [legacy.payload()],
        macro_config,
        as_of=datetime(2025, 3, 1, tzinfo=UTC),
    )["US_CPI"]
    corrected_feature = build_feature_snapshot(
        [legacy.payload(), conservative.payload()],
        macro_config,
        as_of=datetime(2025, 3, 1, tzinfo=UTC),
    )["US_CPI"]
    assert legacy_feature["status"] == "UNAVAILABLE"
    assert corrected_feature["quality_status"] == (
        "CONSERVATIVE_PERIOD_END_RELEASE_LAG_V1"
    )


def test_transformations_use_appropriate_lags() -> None:
    values = pd.Series(range(1, 25), dtype=float)
    yoy = transform_series(values, transformation="yoy", frequency="monthly")
    assert yoy.iloc[:12].isna().all()
    assert yoy.iloc[12] == pytest.approx(1200.0)
    assert transform_series(
        pd.Series([49.0, 51.0]),
        transformation="level_50",
        frequency="monthly",
    ).tolist() == [-1.0, 1.0]


def test_stale_detection_and_missing_confidence(
    macro_config: MacroConfig,
) -> None:
    row = _observation(
        series_id="US_CPI",
        value=100.0,
        available="2020-01-01T00:00:00Z",
    )
    features = build_feature_snapshot(
        [row.payload()],
        macro_config,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert features["US_CPI"]["stale"] is True
    score = compute_score(
        "inflation",
        macro_config.score_weights["inflation"],
        features,
        macro_config,
    )
    assert score.value is None
    assert score.status in {"UNAVAILABLE", "DATA_INCOMPLETE"}
    assert score.confidence < 0.5


def test_complete_fixture_produces_scores_and_regime(
    macro_config: MacroConfig,
    macro_observations: list[dict[str, object]],
) -> None:
    snapshot = compute_macro_snapshot(
        macro_observations,
        macro_config,
        as_of=datetime(2026, 7, 1, tzinfo=UTC),
    )
    assert snapshot["data_quality"]["status"] == "GO"
    assert all(
        snapshot["scores"][name]["value"] is not None
        for name in (
            "growth",
            "inflation",
            "liquidity",
            "credit",
            "breadth",
            "currency",
            "commodity",
        )
    )
    assert snapshot["regime"]["overall_macro_regime"] != "UNKNOWN"
    assert snapshot["financial_evidence"] is False


def test_score_contributions_are_explainable(
    macro_config: MacroConfig,
    macro_observations: list[dict[str, object]],
) -> None:
    snapshot = compute_macro_snapshot(
        macro_observations,
        macro_config,
        as_of=datetime(2026, 7, 1, tzinfo=UTC),
    )
    growth = snapshot["scores"]["growth"]
    assert growth["coverage"] >= macro_config.minimum_score_coverage
    assert growth["positive_contributions"]
    assert "series_id" in growth["positive_contributions"][0]


def test_unknown_regime_when_critical_scores_missing(
    macro_config: MacroConfig,
) -> None:
    missing = {
        name: _score(name, None)
        for name in macro_config.score_weights
    }
    regime = classify_regime(
        missing,
        macro_config,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        history=[],
    )
    assert regime["overall_macro_regime"] == "UNKNOWN"


def test_hysteresis_requires_confirmations_and_minimum_duration() -> None:
    history = [
        {
            "as_of": "2026-01-01T00:00:00+00:00",
            "regime": {
                "overall_macro_regime": "EXPANSION_DISINFLATION",
                "candidate_regime": "EXPANSION_DISINFLATION",
            },
        }
    ]
    accepted, status, count = apply_hysteresis(
        "SLOWDOWN_INFLATION",
        as_of=datetime(2026, 1, 3, tzinfo=UTC),
        history=history,
        minimum_confirmations=2,
        minimum_regime_days=5,
    )
    assert accepted == "TRANSITION"
    assert status == "PENDING_CONFIRMATION"
    assert count == 1


def test_hysteresis_confirms_persistent_candidate() -> None:
    history = [
        {
            "as_of": "2026-01-01T00:00:00+00:00",
            "regime": {
                "overall_macro_regime": "EXPANSION_DISINFLATION",
                "candidate_regime": "EXPANSION_DISINFLATION",
            },
        },
        {
            "as_of": "2026-01-07T00:00:00+00:00",
            "regime": {
                "overall_macro_regime": "TRANSITION",
                "candidate_regime": "SLOWDOWN_INFLATION",
            },
        },
    ]
    accepted, status, _ = apply_hysteresis(
        "SLOWDOWN_INFLATION",
        as_of=datetime(2026, 1, 8, tzinfo=UTC),
        history=history,
        minimum_confirmations=2,
        minimum_regime_days=5,
    )
    assert accepted == "SLOWDOWN_INFLATION"
    assert status == "REGIME_CHANGE_CONFIRMED"


def test_cycle_clock_and_sector_mappings_are_published(
    macro_config: MacroConfig,
    macro_observations: list[dict[str, object]],
) -> None:
    snapshot = compute_macro_snapshot(
        macro_observations,
        macro_config,
        as_of=datetime(2026, 7, 1, tzinfo=UTC),
    )
    assert snapshot["cycle_clock"]["quadrant"] != "UNKNOWN"
    technology = snapshot["implications"]["sectors_and_asset_classes"][
        "technology"
    ]
    assert technology["final_status"] == "WATCHLIST"
    assert technology["technical_confirmation"] == "REQUIRED"


def test_deterministic_analyst_text(
    macro_config: MacroConfig,
    macro_observations: list[dict[str, object]],
) -> None:
    snapshot = compute_macro_snapshot(
        macro_observations,
        macro_config,
        as_of=datetime(2026, 7, 1, tzinfo=UTC),
    )
    first = deterministic_analysis(snapshot, period="daily")
    second = deterministic_analysis(snapshot, period="daily")
    assert first == second
    assert "gegarandeerd" not in " ".join(first["paragraphs_nl"]).lower()
    assert "geen order" in render_markdown(first).lower().splitlines()[-1]
    assert first["order_intent"] is False


def test_fx_and_inflation_normalization() -> None:
    assert fx_normalize(100.0, 0.9) == 90.0
    assert inflation_adjust(0.10, 0.02) == pytest.approx(
        1.10 / 1.02 - 1.0
    )
    with pytest.raises(ValueError):
        fx_normalize(1.0, 0.0)


def test_macro_exposure_is_bounded_long_only_and_unlevered() -> None:
    index = pd.date_range("2026-01-01", periods=2, tz="UTC")
    weights = pd.DataFrame(
        {"A": [0.6, 0.4], "B": [0.4, 0.6]},
        index=index,
    )
    reduced = apply_macro_exposure(weights, 0.5)
    expanded = apply_macro_exposure(weights, 1.1)
    assert (reduced >= 0).all().all()
    assert reduced.sum(axis=1).max() <= 0.5 + 1e-12
    assert expanded.sum(axis=1).max() <= 1.0 + 1e-12


def test_exposure_multiplier_respects_configuration(
    macro_config: MacroConfig,
) -> None:
    value = exposure_multiplier(
        {
            "overall_macro_regime": "TRANSITION",
            "market_regime": "RISK_OFF",
        },
        macro_config,
    )
    assert 0.5 <= value <= 1.1


def test_macro_variant_requires_one_or_two_known_filters() -> None:
    base = generate_strategies(budget=1)[0]
    variant = generate_macro_variant(
        base,
        ("macro_credit_improving", "macro_risk_on"),
    )
    validate_strategy(variant)
    assert variant.parent_strategy_id == base.strategy_id
    assert len(set(variant.regime_components) & MACRO_COMPONENT_NAMES) == 2
    with pytest.raises(ValueError, match="ONE_OR_TWO"):
        generate_macro_variant(base, ())


def test_macro_filter_backtest_fails_closed_without_history() -> None:
    base = generate_strategies(budget=1)[0]
    variant = generate_macro_variant(base, ("macro_risk_on",))
    bars, eligible = deterministic_fixture(symbols=4, periods=400)
    result = run_backtest(
        variant,
        bars,
        eligible=eligible,
        fixture=True,
    )
    assert result.status.startswith("BLOCKED:MACRO_HISTORY_UNAVAILABLE")


def test_macro_variant_is_always_compared_with_baseline() -> None:
    index = pd.date_range("2025-01-01", periods=100, tz="UTC")
    baseline = pd.Series([0.001, -0.001] * 50, index=index)
    macro = pd.Series([0.001, -0.0005] * 50, index=index)
    comparison = compare_macro_variant(baseline, macro)
    assert comparison["baseline_comparison_required"] is True
    assert comparison["status"] == "VALUE_ADDED"


def test_store_is_append_only_and_detects_conflicts(tmp_path: Path) -> None:
    store = MacroStore(MacroLayout.from_project_root(tmp_path))
    try:
        first = _observation(value=100.0)
        assert store.append_observations([first])["inserted"] == 1
        assert store.append_observations([first])["existing"] == 1
        conflict = replace(
            first,
            original_value=101.0,
            provider_payload_hash="DIFFERENT",
        )
        with pytest.raises(
            ValueError,
            match="MACRO_OBSERVATION_IMMUTABILITY_CONFLICT",
        ):
            store.append_observations([conflict])
        quarantined = store.append_observations(
            [conflict],
            quarantine_conflicts=True,
        )
        assert quarantined["conflict_count"] == 1
        assert quarantined["quarantined_conflicts_by_series"] == {"TEST": 1}
    finally:
        store.close()


def test_conflict_resolution_requires_zero_idempotent_conflicts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output" / "macro"
    output.mkdir(parents=True)
    (output / "collection.json").write_text(
        """{"registration":{"conflict_count":0,
        "quarantined_conflicts_by_series":{}}}""",
        encoding="utf-8",
    )
    report = macro_conflicts(tmp_path)
    assert report["status"] == "GO"
    assert report["reported_initial_conflict_count"] == 2414
    assert report["current_conflict_count"] == 0
    assert report["legacy_rows_overwritten"] == 0


def test_official_ecb_future_calendar_is_parsed_with_utc_timestamp() -> None:
    source = """
    <dl><dt>31/07/2099 15:00 CEST<br/>Tentative</dt>
    <dd>Euro area seasonally adjusted HICP flash estimate
    (Dataset: HICP)<br/>Reference period: Jul-2099<br/></dd></dl>
    """
    rows = _parse_ecb_calendar_html(
        source,
        source_url="https://www.ecb.europa.eu/test",
    )
    assert len(rows) == 1
    assert rows[0]["event_id"] == "EU_CPI"
    assert rows[0]["scheduled_at"].endswith("+00:00")
    assert rows[0]["tentative"] is True
    assert rows[0]["automatic_exit"] is False


def test_macro_update_is_single_flight_and_fail_closed(
    tmp_path: Path,
) -> None:
    lock_path = (
        tmp_path / "data" / "macro" / "private" / "macro-update.lock"
    )
    lock_path.parent.mkdir(parents=True)
    lock = FileLock(str(lock_path))
    with lock:
        report = macro_update(tmp_path)
    assert report["status"] == "UPDATE_ALREADY_RUNNING_BLOCKED"
    assert report["execution_authority"] == "NONE"


def test_frozen_macro_artifact_is_immutable(tmp_path: Path) -> None:
    first = _publish_immutable(
        tmp_path,
        "frozen/test.json",
        {"status": "GO"},
    )
    second = _publish_immutable(
        tmp_path,
        "frozen/test.json",
        {"status": "GO"},
    )
    assert first == second
    with pytest.raises(
        ValueError,
        match="MACRO_FROZEN_ARTIFACT_IMMUTABILITY_CONFLICT",
    ):
        _publish_immutable(
            tmp_path,
            "frozen/test.json",
            {"status": "CHANGED"},
        )


def test_screener_macro_context_cannot_override_hard_gates(
    macro_config: MacroConfig,
    macro_observations: list[dict[str, object]],
) -> None:
    snapshot = compute_macro_snapshot(
        macro_observations,
        macro_config,
        as_of=datetime(2026, 7, 1, tzinfo=UTC),
    )
    asset = _asset_snapshot()
    score, context = _macro_context_score(asset, snapshot)
    assert score is not None
    assert 0 <= score <= 100
    assert context["order_signal"] is False
    missing_score, missing = _macro_context_score(asset, None)
    assert missing_score is None
    assert missing["status"] == "UNAVAILABLE"


def test_configuration_has_provider_fallback_provenance(
    macro_config: MacroConfig,
) -> None:
    assert macro_config.series["EU_CPI"].primary_source == "EUROSTAT"
    assert macro_config.series["EU_CPI"].fallback_source == "FRED"
    assert macro_config.series["ECB_POLICY_RATE"].primary_source == "ECB"
    assert (
        macro_config.series["US_BUSINESS_CONFIDENCE"].provider_id
        == "BSCICP03USM665S"
    )
    assert (
        macro_config.series["US_CPI"].vintage_capable is True
        and macro_config.series["EU_CPI"].vintage_capable is False
    )


def test_limited_pit_aggregate_reduces_score_confidence(
    macro_config: MacroConfig,
) -> None:
    feature = {
        "US_REPORTED_EARNINGS_GROWTH_BREADTH": {
            "normalized_score": 20.0,
            "stale": False,
            "quality_confidence_multiplier": 0.25,
        },
        "US_EARNINGS_REVISION_BREADTH": {
            "normalized_score": 20.0,
            "stale": False,
            "quality_confidence_multiplier": 1.0,
        },
    }
    score = compute_score(
        "earnings_cycle",
        macro_config.score_weights["earnings_cycle"],
        feature,
        macro_config,
    )
    assert score.value == pytest.approx(20.0)
    assert score.confidence < score.coverage


def test_no_ai_or_broker_write_imports_in_macro_source() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(
            (PROJECT_ROOT / "src" / "stocks" / "macro").glob("*.py")
        )
    )
    assert "import openai" not in text
    assert "import torch" not in text
    for token in (
        "place" + "Order",
        "cancel" + "Order",
        "req" + "Global" + "Cancel",
        "req" + "Ids",
    ):
        assert token not in text


def _score(name: str, value: float | None) -> MacroScore:
    return MacroScore(
        name=name,
        value=value,
        confidence=0.0 if value is None else 1.0,
        coverage=0.0 if value is None else 1.0,
        status="UNAVAILABLE" if value is None else "VALID",
        positive_contributions=(),
        negative_contributions=(),
        missing_inputs=(),
        stale_inputs=(),
    )


def _observation(
    *,
    series_id: str = "TEST",
    value: float,
    available: str = "2025-02-01T12:00:00Z",
    publication: str | None = None,
    revision: str = "ORIGINAL",
) -> MacroObservation:
    available_at = pd.Timestamp(available).to_pydatetime()
    publication_at = (
        available_at
        if publication is None
        else pd.Timestamp(publication).to_pydatetime()
    )
    return MacroObservation(
        series_id=series_id,
        observation_date=date(2025, 1, 1),
        publication_at=publication_at,
        available_at=available_at,
        revision_status=revision,
        source="FIXTURE",
        provider="TEST",
        original_value=value,
        transformed_value=None,
        frequency="monthly",
        region="US",
        vintage=available_at.date().isoformat(),
        quality_status="FIXTURE_ONLY",
        stale_status="CURRENT",
        provider_payload_hash=f"VALUE-{value}",
    )


def _asset_snapshot() -> AssetSnapshot:
    metadata = AssetMetadata(
        asset_key="A",
        symbol="A",
        name="Asset",
        con_id=1,
        asset_type="STOCK",
        exchange="SMART",
        currency="USD",
        sector="Technology",
        industry="Software - Application",
        category=None,
        inactive=False,
    )
    return AssetSnapshot(
        metadata=metadata,
        bars=pd.DataFrame(),
        price_source="FIXTURE",
        price_source_timestamp=None,
        fundamental=None,
        shariah=ShariahSnapshot(
            status="SHARIAH_COMPLIANT",
            screened_at=datetime(2026, 1, 1, tzinfo=UTC),
            expires_at=datetime(2027, 1, 1, tzinfo=UTC),
            methodology="TEST",
            source="TEST",
        ),
        benchmark_symbol="SPY",
        benchmark_bars=pd.DataFrame(),
    )
