from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocks.research.phase11_6 import (
    AUTHORITY,
    completion_audit,
    empirical_periods_per_year,
    nested_walk_forward_folds,
    phase11_6_schema,
    strategy_signal,
)


def test_daily_nested_walk_forward_has_five_year_train_and_disjoint_outer() -> None:
    folds = nested_walk_forward_folds("2000-01-01", "2026-06-30", "1d")
    assert len(folds) >= 5
    first = folds.iloc[0]
    assert first["train_end"] < first["validation_start"]
    assert first["validation_end"] < first["outer_test_start"]
    assert (first["train_end"] - first["train_start"]).days >= 365 * 4
    assert folds["outer_test_start"].is_monotonic_increasing


def test_intraday_nested_walk_forward_uses_three_month_outer_steps() -> None:
    folds = nested_walk_forward_folds("2020-01-01", "2024-01-01", "1h")
    assert len(folds) >= 5
    first = folds.iloc[0]
    assert (first["validation_end"] - first["validation_start"]).days in range(88, 93)
    assert (first["outer_test_end"] - first["outer_test_start"]).days in range(88, 93)


def test_walk_forward_outer_test_never_overlaps_selection_window() -> None:
    folds = nested_walk_forward_folds("2000-01-01", "2026-01-01", "1w")
    assert (folds["validation_end"] < folds["outer_test_start"]).all()
    assert (folds["train_end"] < folds["validation_start"]).all()


def test_empirical_annualization_is_not_252_times_24() -> None:
    hourly = pd.date_range("2025-01-02 14:30", periods=1_638, freq="h", tz="UTC")
    factor = empirical_periods_per_year(hourly)
    assert factor != 252 * 24
    assert factor > 0


def test_ma_signal_has_causal_warmup() -> None:
    frame = pd.DataFrame({"close": range(1, 251)})
    signal = strategy_signal(frame, "ma_crossover", {"fast": 50, "slow": 200})
    assert not signal.iloc[:199].any()
    assert signal.iloc[-1]


def test_phase11_6_contract_keeps_all_authority_disabled(tmp_path: Path) -> None:
    contract_root = tmp_path / "config" / "research_contracts"
    contract_root.mkdir(parents=True)
    source_root = Path(__file__).parents[1] / "config" / "research_contracts"
    for name in (
        "stocks_multitimeframe_data_contract_v1.json",
        "stocks_walk_forward_contract_v1.json",
        "stocks_combination_architecture_contract_v1.json",
        "stocks_strategy_timeframe_registry_v1.json",
    ):
        (contract_root / name).write_bytes((source_root / name).read_bytes())
    payload = phase11_6_schema(tmp_path)
    assert payload["EXECUTION_AUTHORITY"] == "NONE"
    assert payload["BROKER_CALLS"] == 0
    assert AUTHORITY["FINANCIAL_FINALIST_GO"] is False


def test_completion_audit_blocks_500_pilot_without_all_artifacts(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[1]
    (tmp_path / "src" / "stocks" / "research").mkdir(parents=True)
    (tmp_path / "src" / "stocks" / "data").mkdir(parents=True)
    for relative in (
        "src/stocks/research/phase11_6.py",
        "src/stocks/research/parameter_research_v2.py",
        "src/stocks/data/multitimeframe.py",
        "strategy_combo_research_lab.py",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    result = completion_audit(tmp_path)
    assert result["pilot_500_allowed"] is False
    assert result["pilot_500_status"] == "BLOCKED_BY_PREDECLARED_GATES"
    assert json.loads((tmp_path / "output" / "research" / "phase11_6" / "completion_audit.json").read_text())["authority_none"] if "authority_none" in result else result["fields"]["authority_none"]

