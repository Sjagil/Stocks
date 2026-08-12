from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import strategy_combo_research_lab as lab
from stocks.research import parameter_research_v2 as research


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "research_contracts" / "stocks_parameter_space_registry_v2.json"


def resolved_registry() -> dict[str, object]:
    return research.resolve_registry(REGISTRY, lab)


def strategy(name: str) -> dict[str, object]:
    return research.strategy_map(resolved_registry())[name]


def test_registry_resolves_every_runtime_strategy_and_rotational() -> None:
    resolved = resolved_registry()
    expected = {spec.name for spec in lab.strategy_registry()} | {"rotational_momentum"}
    assert set(research.strategy_map(resolved)) == expected
    assert len(expected) == 33


def test_registry_hash_is_stable_despite_resolution_timestamp() -> None:
    first = resolved_registry()
    second = resolved_registry()
    assert first["resolved_registry_sha256"] == second["resolved_registry_sha256"]
    assert first["source_registry_sha256"] == second["source_registry_sha256"]


def test_unregistered_parameter_value_is_blocked() -> None:
    item = strategy("rsi_adx")
    params = dict(item["baseline_configuration"])
    params["entry_threshold"] = 14.123
    with pytest.raises(ValueError, match="UNREGISTERED_PARAMETER_VALUE_BLOCKED"):
        research.validate_configuration(item, params, lab)


def test_structurally_invalid_ma_configuration_is_removed() -> None:
    item = strategy("ma_crossover")
    params = dict(item["baseline_configuration"])
    params["fast"] = 100
    params["slow"] = 100
    assert not research.validate_configuration(item, params, lab)


def test_small_space_uses_exhaustive_and_preserves_baseline() -> None:
    item = strategy("consecutive_lower_lows")
    planned, counts = research.plan_strategy(item, lab, "sobol", 100, 7)
    assert counts["search_method"] == "EXHAUSTIVE_GRID"
    assert planned[0] == item["baseline_configuration"]
    assert len(planned) == counts["raw_parameter_combinations"]


@pytest.mark.parametrize("method", ["sobol", "latin-hypercube", "deterministic-random"])
def test_stratified_sampling_is_deterministic(method: str) -> None:
    item = strategy("asymmetric_ma_crossover")
    first, _ = research.plan_strategy(item, lab, method, 12, 123)
    second, _ = research.plan_strategy(item, lab, method, 12, 123)
    assert first == second
    assert first[0] == item["baseline_configuration"]


def test_neighbors_change_exactly_one_registered_grid_step() -> None:
    item = strategy("ma_crossover")
    baseline = item["baseline_configuration"]
    neighbors = research.neighbor_configurations(item, baseline, lab)
    assert len(neighbors) >= 3
    for neighbor in neighbors:
        changed = [key for key in baseline if baseline[key] != neighbor[key]]
        assert len(changed) == 1


def test_decimal_refinement_is_registered_before_execution() -> None:
    item = strategy("rsi_adx")
    threshold = next(
        value for value in item["search_parameters"] if value["name"] == "entry_threshold"
    )
    assert 12.5 in threshold["refinement_allowed_values"]
    assert 12.5 in threshold["allowed_values"]
    refinements = research.refinement_configurations(item, item["baseline_configuration"], lab)
    assert any(value["entry_threshold"] in {12.5, 17.5} for value in refinements)


def test_trial_hash_is_timeframe_specific() -> None:
    params = {"fast": 50, "slow": 200, "max_hold": 2520}
    assert research.parameter_hash("ma_crossover", "1d", params) != research.parameter_hash(
        "ma_crossover", "1w", params
    )


def test_closed_week_resample_uses_week_ohlcv(tmp_path: Path) -> None:
    source = tmp_path / "daily.parquet"
    destination = tmp_path / "weekly.parquet"
    dates = pd.bdate_range("2020-01-06", periods=10)
    frame = pd.DataFrame(
        {
            "security_id": "SID-1",
            "ticker": "ABC",
            "date": dates.astype(str),
            "open": np.arange(10, dtype=float) + 10,
            "high": np.arange(10, dtype=float) + 11,
            "low": np.arange(10, dtype=float) + 9,
            "close": np.arange(10, dtype=float) + 10.5,
            "volume": 100.0,
            "sector": "Tech",
            "currency": "USD",
            "source": "TEST",
            "price_basis": "RAW",
        }
    )
    frame.to_parquet(source, index=False)
    research.materialize_weekly_data(source, destination, "2020-01-01", "2020-12-31", None, 7)
    weekly = pd.read_parquet(destination)
    assert len(weekly) == 2
    assert weekly.iloc[0]["open"] == 10.0
    assert weekly.iloc[0]["close"] == 14.5
    assert weekly.iloc[0]["volume"] == 500.0
    assert set(weekly["source"]) == {"PIT_CAUSAL_CLOSED_WEEK_RESAMPLE"}


def test_annualization_is_timeframe_aware() -> None:
    daily = pd.bdate_range("2020-01-01", periods=520)
    weekly = pd.date_range("2020-01-03", periods=104, freq="W-FRI")
    assert 250 < lab.annualization_factor(daily) < 265
    assert 51 < lab.annualization_factor(weekly) < 53


def test_forbidden_call_scan_is_ast_based(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    dirty = tmp_path / "dirty.py"
    clean.write_text("TOKENS = ['placeOrder']\n", encoding="utf-8")
    dirty.write_text("client.placeOrder(1, None, None)\n", encoding="utf-8")
    assert research.forbidden_call_scan([clean])["status"] == "GO"
    audit = research.forbidden_call_scan([dirty])
    assert audit["status"] == "NO_GO"
    assert audit["findings"][0]["token"] == "placeOrder"


def test_registry_source_declares_dedicated_15m_builder_without_generic_campaign() -> None:
    source = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert source["timeframes"]["1h"]["status"] == "DATA_UNAVAILABLE_BLOCKED"
    assert source["timeframes"]["15m"]["status"] == (
        "DEDICATED_ACTIVE_SWING_NATIVE_DATA_AND_SIGNAL_BUILDER_AVAILABLE"
    )
    assert source["timeframes"]["15m"]["generic_v2_parameter_campaign_enabled"] is False
    assert "15m" not in source["forbidden_timeframes"]
