from stocks.analysis.confluence import evaluate_multilayer_confluence


def test_three_supportive_layers_confirm_without_authority() -> None:
    report = evaluate_multilayer_confluence(
        technical_score=0.82,
        fundamental_score=0.74,
        fundamental_required=True,
        macro_score=0.68,
        macro_confidence=0.80,
        macro_status="AVAILABLE",
    )

    assert report["status"] == "THREE_LAYER_CONFIRMED"
    assert report["confluence_score"] > 0.70
    assert report["ranking_multiplier"] == 1.05
    assert report["allocation_allowed"] is True
    assert report["standalone_entry_allowed"] is False
    assert report["execution_authority"] == "NONE"


def test_missing_required_fundamental_layer_blocks_allocation() -> None:
    report = evaluate_multilayer_confluence(
        technical_score=0.85,
        fundamental_score=None,
        fundamental_required=True,
        macro_score=0.70,
        macro_confidence=0.80,
        macro_status="AVAILABLE",
    )

    assert report["status"] == "BLOCKED_MISSING_REQUIRED_LAYER"
    assert report["allocation_allowed"] is False
    assert report["missing_required_layers"] == ["FUNDAMENTAL"]
    assert report["allocation_blockers"] == [
        "MULTILAYER_FUNDAMENTAL_DATA_REQUIRED"
    ]


def test_macro_headwind_reduces_ranking_but_cannot_create_entry() -> None:
    report = evaluate_multilayer_confluence(
        technical_score=0.80,
        fundamental_score=0.75,
        fundamental_required=True,
        macro_score=0.20,
        macro_confidence=0.90,
        macro_status="AVAILABLE",
    )

    assert report["status"] == "MACRO_HEADWIND_RISK_REDUCTION"
    assert report["ranking_multiplier"] == 0.85
    assert report["layers"]["macro"]["severe_headwind"] is True
    assert report["technical_signal_required"] is True
    assert report["standalone_entry_allowed"] is False


def test_missing_macro_is_not_silently_neutralized() -> None:
    report = evaluate_multilayer_confluence(
        technical_score=0.80,
        fundamental_score=0.75,
        fundamental_required=True,
        macro_score=None,
        macro_confidence=0.0,
        macro_status="UNAVAILABLE",
    )

    assert report["status"] == "BLOCKED_MISSING_REQUIRED_LAYER"
    assert "MACRO" in report["missing_required_layers"]
    assert "MULTILAYER_MACRO_CONTEXT_REQUIRED" in report[
        "allocation_blockers"
    ]
