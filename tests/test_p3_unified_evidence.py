from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocks.p3.analytics import (
    multiple_testing_diagnostics,
    strategy_dependency_diagnostics,
)
from stocks.p3.contracts import StrategyDNA, UnifiedTrialRecord
from stocks.p3.io import atomic_write_json, atomic_write_jsonl, file_hash
from stocks.p3.ledger import UnifiedTrialLedger
from stocks.p3.publisher import _blocker_type, _unique


def _dna() -> StrategyDNA:
    return StrategyDNA(
        strategy_id="STRATEGY-1",
        native_strategy_hash="NATIVE-HASH",
        strategy_family="trend",
        economic_hypothesis="persistent trend continuation after confirmation",
        direction="LONG_ONLY",
        universe_scope=("MSFT", "AAPL"),
        entry_rule="close > sma_200",
        exit_rule="close < sma_50",
        stop_rule="2_atr",
        target_rule="trailing_stop",
        time_exit="20_bars",
        entry_timeframe="1d",
        setup_timeframe="1w",
        context_timeframes=("1mo",),
        feature_set=("sma_200", "sma_50", "atr_14"),
        parameters={"slow": 200, "fast": 50},
        regime_filter="risk_on",
        position_management="whole_share_risk_budget",
        cost_model_version="cost-v1",
        fill_model_version="next-open-v1",
        source_registry="TEST",
        completeness_status="DNA_COMPLETE",
    )


def _trial() -> UnifiedTrialRecord:
    dna = _dna()
    return UnifiedTrialRecord(
        source_namespace="TEST",
        source_record_id="TRIAL-1",
        research_family="trend",
        hypothesis_id="H-1",
        strategy_id=dna.strategy_id,
        strategy_spec_hash=dna.strategy_spec_hash,
        model_id=None,
        parameters={"fast": 50, "slow": 200},
        timeframes=("1d", "1w"),
        universe=("AAPL", "MSFT"),
        features=("sma_50", "sma_200"),
        regime_filter="risk_on",
        entry="close > sma_200",
        exit="close < sma_50",
        stop="2_atr",
        target="trailing_stop",
        holding_rule="20_bars",
        cost_version="cost-v1",
        fill_model_version="next-open-v1",
        data_hash="DATA-HASH",
        cutoff="2025-12-31",
        code_hash="CODE-HASH",
        seed=7,
        created_at="2026-08-11T00:00:00+00:00",
        status="INSUFFICIENT_EVIDENCE",
        rejection_reason=("FORWARD_REQUIRED",),
        metrics={"profit_factor": 1.1},
        provenance={"source": "unit-test"},
    )


def test_strategy_dna_hash_is_order_invariant() -> None:
    left = _dna()
    right = replace(
        left,
        universe_scope=tuple(reversed(left.universe_scope)),
        feature_set=tuple(reversed(left.feature_set)),
        parameters={"fast": 50, "slow": 200},
    )
    assert left.strategy_spec_hash == right.strategy_spec_hash
    assert left.as_record()["canonical_spec"]["universe_scope"] == ["AAPL", "MSFT"]


def test_research_trial_cannot_receive_money_authority() -> None:
    with pytest.raises(ValueError, match="MONEY_AUTHORITY_FORBIDDEN"):
        replace(_trial(), money_control=True)
    with pytest.raises(ValueError, match="MONEY_AUTHORITY_FORBIDDEN"):
        replace(_trial(), execution_authority="ACTIVE")


def test_unified_ledger_is_idempotent_and_immutable(tmp_path: Path) -> None:
    ledger = UnifiedTrialLedger(tmp_path)
    try:
        assert ledger.import_records([_trial()]) == {"inserted": 1, "existing": 0}
        assert ledger.import_records([_trial()]) == {"inserted": 0, "existing": 1}
        changed = replace(_trial(), metrics={"profit_factor": 1.2})
        with pytest.raises(ValueError, match="IMMUTABILITY_CONFLICT"):
            ledger.import_records([changed])
        assert ledger.counts() == {"TEST": 1, "TOTAL": 1}
        public = ledger.export_public_jsonl()
    finally:
        ledger.close()
    row = json.loads(public.read_text(encoding="utf-8"))
    assert row["execution_authority"] == "NONE"
    assert row["money_control"] is False


def test_atomic_artifact_writers_replace_complete_files(tmp_path: Path) -> None:
    json_path = atomic_write_json(tmp_path / "state.json", {"status": "GO"})
    jsonl_path = atomic_write_jsonl(tmp_path / "ledger.jsonl", [{"id": 1}, {"id": 2}])
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"status": "GO"}
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2
    assert file_hash(json_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_blockers_are_stable_and_classified() -> None:
    assert _unique(["A", "A", None, "", "B"]) == ["A", "B"]
    assert _blocker_type("LIVE_QUOTE_CAPTURE_BLOCKED") == "EXTERNAL_MARKET_DATA_OR_QUOTE"
    assert _blocker_type("COST_STRESS_FAILED") == "ECONOMIC_EXECUTION_COST"


def test_multiple_testing_uses_shared_oos_and_is_deterministic() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(0.0, 0.01, size=(320, 4))
    values[:, 0] += 0.0003
    frame = pd.DataFrame(values, columns=["A", "B", "C", "D"])
    first = multiple_testing_diagnostics(
        frame, global_trial_count=4_000, seed=17, bootstrap_samples=80
    )
    second = multiple_testing_diagnostics(
        frame, global_trial_count=4_000, seed=17, bootstrap_samples=80
    )
    assert first == second
    assert first["shared_observation_count"] == 320
    assert first["strategy_count"] == 4
    assert first["cscv_pbo"]["method"] == "CLASSICAL_CSCV_CONTIGUOUS_PARTITIONS"
    assert first["white_reality_check"]["p_value"] is not None
    assert first["hansen_spa"]["p_value"] is not None


def test_strategy_dependency_detects_near_duplicate_series() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.01, size=200)
    frame = pd.DataFrame(
        {
            "A": base,
            "A_COPY": base + rng.normal(0.0, 0.00001, size=200),
            "B": rng.normal(0.0, 0.01, size=200),
        }
    )
    result = strategy_dependency_diagnostics(frame)
    assert result["status"] == "GO"
    assert ["A", "A_COPY"] in result["near_duplicate_clusters"]
    assert result["trade_overlap_status"] == "NOT_AVAILABLE_FROM_DAILY_OOS_SERIES"
