from __future__ import annotations

import json
from pathlib import Path

from stocks.ibkr.data_capabilities import (
    build_capability_matrix,
    capability_schema,
    strategy_capability_gate,
)


def _observation(root: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "status": "GO_DEGRADED_NO_TAPE_OR_DEPTH",
        "generated_at": "2026-08-06T12:00:00+00:00",
        "requested_symbols_masked": ["ASSET-A"],
        "market_data_type": 1,
        "quote_row_count": 0,
        "trade_row_count": 0,
        "depth_row_count": 0,
        "observed_symbol_count": 0,
        "depth_observed_symbol_count": 0,
        "error_code_counts": {"10089": 1, "10189": 1},
    }
    payload.update(overrides)
    path = root / "output/market_context/realtime-equity-collection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_capability_schema_forbids_silent_fallback() -> None:
    schema = capability_schema()
    assert schema["fallback_policy"] == (
        "NO_SILENT_REALTIME_TO_DELAYED_FALLBACK"
    )
    assert schema["execution_authority"] == "NONE"


def test_entitlement_errors_block_tape_and_depth(tmp_path: Path) -> None:
    _observation(tmp_path)
    matrix = build_capability_matrix(tmp_path)
    assert matrix["summary"]["tick_by_tick_trades"] == (
        "UNAVAILABLE_ENTITLEMENT"
    )
    assert matrix["summary"]["market_depth_l2"] == (
        "UNAVAILABLE_ENTITLEMENT"
    )
    assert matrix["tape_features_allowed"] is False
    assert matrix["depth_features_allowed"] is False
    assert matrix["broker_write_calls"] == 0


def test_observed_rows_are_required_before_realtime_is_available(
    tmp_path: Path,
) -> None:
    _observation(
        tmp_path,
        error_code_counts={},
        quote_row_count=2,
        trade_row_count=3,
        depth_row_count=4,
        observed_symbol_count=1,
        depth_observed_symbol_count=1,
    )
    matrix = build_capability_matrix(tmp_path)
    assert matrix["summary"]["realtime_top_of_book"] == "AVAILABLE"
    assert matrix["summary"]["tick_by_tick_trades"] == "AVAILABLE"
    assert matrix["summary"]["market_depth_l2"] == "AVAILABLE"


def test_strategy_gate_fails_closed_when_tape_is_missing(
    tmp_path: Path,
) -> None:
    _observation(tmp_path)
    matrix = build_capability_matrix(tmp_path)
    gate = strategy_capability_gate(
        matrix,
        ["realtime_top_of_book", "tick_by_tick_trades"],
        asset_reference="ASSET-A",
    )
    assert gate["status"] == "NO_GO"
    assert "CAPABILITY_NOT_AVAILABLE:tick_by_tick_trades" in gate["blockers"]
    assert gate["execution_authority"] == "NONE"


def test_historical_bars_need_actual_cache_evidence(tmp_path: Path) -> None:
    _observation(tmp_path, error_code_counts={})
    path = (
        tmp_path
        / "data/bars/security_type=STK/con_id=MASKED/interval=1d/"
        "data_type=TRADES/bars.parquet"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PARQUET-EVIDENCE")
    matrix = build_capability_matrix(tmp_path)
    assert matrix["summary"]["historical_bars"] == "AVAILABLE"

