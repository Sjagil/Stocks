from __future__ import annotations

from stocks.portfolio.real_assets import (
    active_swing_timeframe_context,
    opportunity_class,
    product_identity_context,
    real_asset_context,
)


def test_active_swing_weights_lower_timeframes_without_daily_veto() -> None:
    context = active_swing_timeframe_context(["1h", "4h"])

    assert context["score"] == 0.6
    assert context["classification"] == "ACTIVE_SWING_ALIGNED"
    assert context["higher_timeframe_risk_multiplier"] == 0.9
    assert context["higher_timeframe_policy"] == (
        "MULTIPLIER_NOT_AUTOMATIC_VETO"
    )
    assert context["standalone_entry_allowed"] is False


def test_daily_confirmation_increases_context_not_entry_authority() -> None:
    context = active_swing_timeframe_context(["1h", "4h", "1d"])

    assert context["score"] == 0.75
    assert context["higher_timeframe_risk_multiplier"] == 1.0
    assert context["timing_15m_status"] == "NOT_OBSERVED"


def test_fifteen_minute_is_tactical_context_not_standalone_authority() -> None:
    context = active_swing_timeframe_context(["15m", "1h", "4h", "1d"])
    assert context["classification"] == "TACTICAL_SWING_CONTEXT_AVAILABLE"
    assert context["timing_15m_status"] == "AVAILABLE"
    assert context["standalone_entry_allowed"] is False


def test_real_asset_classification_precedes_strategy_family() -> None:
    metadata = {
        "sleeve": "commodity_security",
        "commodity_exposure_type": "PHYSICAL_COMMODITY",
    }

    assert opportunity_class(metadata, ["rsi2_adx_pullback"]) == (
        "REAL_ASSET"
    )


def test_reversal_and_trend_classes_are_distinct() -> None:
    metadata = {
        "sleeve": "stock",
        "commodity_exposure_type": "NONE",
    }

    assert opportunity_class(metadata, ["rsi2_adx_pullback"]) == (
        "REVERSAL"
    )
    assert opportunity_class(metadata, ["donchian_breakout"]) == "TREND"


def test_real_asset_context_preserves_missing_specialized_inputs() -> None:
    context = real_asset_context(
        {
            "sleeve": "commodity_security",
            "commodity_exposure_type": "PHYSICAL_COMMODITY",
            "product_structure": "PHYSICAL_CLOSED_END_TRUST",
            "underlying_commodity": "COPPER",
        },
        {
            "signal_quality": 0.8,
            "timeframe_confirmation": 0.6,
            "relative_strength": 0.7,
            "setup_quality": 0.75,
            "regime_fit": 0.65,
            "liquidity": 0.8,
        },
    )

    assert context["status"] == "PARTIAL_AVAILABLE_COMPONENTS_ONLY"
    assert context["underlying_commodity"] == "COPPER"
    assert context["product_structure"] == "PHYSICAL_CLOSED_END_TRUST"
    assert context["unavailable_components"]["carry"].startswith(
        "UNAVAILABLE"
    )
    assert context["ranking_influence"] == (
        "OBSERVATION_ONLY_PENDING_ABLATION"
    )
    assert context["execution_authority"] == "NONE"
    assert context["product_identity"]["physical_structure_verified"] is False
    assert "PHYSICAL_PRODUCT_STRUCTURE_NOT_CURRENTLY_VERIFIED" in context[
        "product_identity"
    ]["blockers"]


def test_verified_physical_structure_does_not_imply_shariah_eligibility() -> None:
    identity = product_identity_context(
        {
            "commodity_exposure_type": "PHYSICAL_COMMODITY",
            "product_identity_status": "VERIFIED_PHYSICAL_STRUCTURE",
            "physical_structure_verified": True,
            "product_identity_screened_at": "2026-08-12T00:00:00Z",
            "product_identity_expires_at": "2026-09-11T23:59:59Z",
            "product_identity_source_count": 1,
            "shariah_product_status": "ATTESTATION_REQUIRED",
        }
    )

    assert identity["physical_structure_verified"] is True
    assert identity["shariah_is_separate_from_physical_structure"] is True
    assert identity["deployment_eligible"] is False
    assert identity["blockers"] == ["SHARIAH_PRODUCT_ATTESTATION_REQUIRED"]
