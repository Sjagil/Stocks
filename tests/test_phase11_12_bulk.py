from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger
from stocks.research.phase11_12 import (
    ASSET_BUCKETS,
    CATALOG_FORMULAS,
    PAIR_ENSEMBLES,
    PROFILES,
    _bar_freshness,
    _forward_frames,
    _formula_signals,
    _select_shadow_candidates,
    _write_frame,
    generate_bulk_catalog,
    register_phase11_12_catalog,
    phase11_12_observe,
    run_phase11_12,
)
from stocks.research.phase11_9 import (
    BASE_STRATEGIES,
    ENSEMBLES,
    TIMEFRAMES,
)


def test_bulk_catalog_has_unique_complete_dna_combinations() -> None:
    catalog = generate_bulk_catalog()

    expected = (
        len(CATALOG_FORMULAS)
        * len(TIMEFRAMES)
        * len(PROFILES)
        * len(ASSET_BUCKETS)
    )
    assert expected >= 2_000
    assert len(catalog) == expected
    assert len({row["strategy_id"] for row in catalog}) == expected
    assert len({row["strategy_hash"] for row in catalog}) == expected
    assert {
        (
            row["formula"],
            row["timeframe"],
            row["profile"],
            row["asset_class"],
        )
        for row in catalog
    } == {
        (formula, timeframe, profile, asset_class)
        for formula in (*BASE_STRATEGIES, *ENSEMBLES)
        for timeframe in TIMEFRAMES
        for profile in PROFILES
        for asset_class in ASSET_BUCKETS
    } | {
        (formula, timeframe, profile, asset_class)
        for formula in PAIR_ENSEMBLES
        for timeframe in TIMEFRAMES
        for profile in PROFILES
        for asset_class in ASSET_BUCKETS
    }
    assert all(row["long_only"] for row in catalog)
    assert all(not row["leverage"] for row in catalog)
    assert all(row["research_authority"] == "NONE" for row in catalog)


def test_forward_frames_use_current_provider_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sentinel = {"1d": {"AAPL": pd.DataFrame({"close": [1.0]})}}
    monkeypatch.setattr(
        "stocks.research.phase11_12._load_current_frames",
        lambda project_root: sentinel,
    )

    assert _forward_frames(tmp_path) is sentinel


def test_bulk_strategy_registration_is_idempotent(tmp_path: Path) -> None:
    rows = generate_bulk_catalog()[:10]
    ledger = ResearchLedger(AutopilotLayout(tmp_path))
    try:
        first = ledger.register_bulk_strategies(rows)
        second = ledger.register_bulk_strategies(rows)
        counts = ledger.counts()
    finally:
        ledger.close()

    assert first == {"inserted": 10, "existing": 0}
    assert second == {"inserted": 0, "existing": 10}
    assert counts["bulk_strategy_dna"] == 10


def test_two_block_catalog_is_complete_and_unanimous(
    monkeypatch,
) -> None:
    catalog = generate_bulk_catalog(component_count=2)
    expected = (
        len(PAIR_ENSEMBLES)
        * len(TIMEFRAMES)
        * len(PROFILES)
        * len(ASSET_BUCKETS)
    )
    assert len(catalog) == expected
    assert all(
        len(row["indicator_components"]) == 2
        and row["combination_rule"] == "UNANIMOUS_CLOSED_BAR"
        for row in catalog
    )

    index = pd.date_range("2026-01-01", periods=3, tz="UTC")

    def fake_signals(_frames, formula, _timeframe, _profile):
        signal = (
            [True, True, False]
            if formula == "ma_crossover"
            else [False, True, True]
        )
        return {
            "A": pd.DataFrame(
                {"signal": signal, "score": [1.0, 2.0, 3.0]},
                index=index,
            )
        }

    monkeypatch.setattr(
        "stocks.research.phase11_12._signals",
        fake_signals,
    )
    formula = "pair::ma_crossover::asymmetric_ma"
    result = _formula_signals(
        {"A": pd.DataFrame(index=index)},
        formula,
        "1d",
        "balanced",
    )
    assert result["A"]["signal"].tolist() == [False, True, False]
    assert result["A"]["score"].iloc[1] == 2.0


def test_complexity_registration_is_idempotent(tmp_path: Path) -> None:
    first = register_phase11_12_catalog(
        tmp_path,
        complexity=2,
        resume=True,
    )
    second = register_phase11_12_catalog(
        tmp_path,
        complexity=2,
        resume=True,
    )

    assert first["status"] == "GO"
    assert first["strategy_dna_count"] > 50_000
    assert first["registration"]["inserted"] == first[
        "strategy_dna_count"
    ]
    assert second["registration"]["inserted"] == 0
    assert second["registration"]["existing"] == first[
        "strategy_dna_count"
    ]
    assert first["backtests_executed"] == 0
    assert first["EXECUTION_AUTHORITY"] == "NONE"


def test_pending_queue_advances_without_repeating_evaluated_dna(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "stocks.research.phase11_12._load_frames",
        lambda _project_root: {},
    )
    first = run_phase11_12(
        tmp_path,
        complexity=2,
        max_strategies=1,
        pending_only=True,
    )
    first_catalog = pd.read_parquet(
        tmp_path
        / "output/research/phase11_12/strategy-catalog.parquet"
    )
    second = run_phase11_12(
        tmp_path,
        complexity=2,
        max_strategies=1,
        pending_only=True,
    )
    second_catalog = pd.read_parquet(
        tmp_path
        / "output/research/phase11_12/strategy-catalog.parquet"
    )

    assert first["pending_before"] == len(PAIR_ENSEMBLES) * 54
    assert first["pending_after"] == first["pending_before"] - 1
    assert second["pending_before"] == first["pending_after"]
    assert second["pending_after"] == second["pending_before"] - 1
    assert first_catalog.loc[0, "strategy_id"] != second_catalog.loc[
        0,
        "strategy_id",
    ]
    assert second["append_only_result_count"] == 4
    assert second["cumulative_evaluated_strategy_count"] == 2
    assert len(
        pd.read_parquet(
            tmp_path
            / "output/research/phase11_12/strategy-summary.parquet"
        )
    ) == 2
    assert len(
        pd.read_parquet(
            tmp_path
            / (
                "output/research/phase11_12/"
                "strategy-summary-latest-batch.parquet"
            )
        )
    ) == 1
    assert second["EXECUTION_AUTHORITY"] == "NONE"


def test_repeated_parquet_writes_do_not_nest_json_encoding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frame.parquet"
    _write_frame(
        path,
        pd.DataFrame(
            [{"metrics": {"CAGR": 0.1}, "provenance": {"broker_calls": 0}}]
        ),
    )
    first = pd.read_parquet(path)
    _write_frame(path, first)
    second = pd.read_parquet(path)

    assert json.loads(second.loc[0, "metrics"]) == {"CAGR": 0.1}
    assert json.loads(second.loc[0, "provenance"]) == {
        "broker_calls": 0
    }


def _summary_row(
    strategy_id: str,
    *,
    profile: str,
    sharpe: float = 0.8,
    fills: int = 100,
    fingerprint: str | None = None,
) -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "strategy_hash": f"hash-{strategy_id}",
        "formula": "flow_consensus",
        "timeframe": "1h",
        "profile": profile,
        "asset_class": "COMMODITY_PROXY",
        "status": "COMPLETE",
        "CAGR": 0.08,
        "Sharpe": sharpe,
        "period_profit_factor": 1.2,
        "stress_50bps_profit_factor": 1.05,
        "maximum_drawdown": -0.1,
        "fill_count": fills,
        "economic_outcome_fingerprint": fingerprint or strategy_id,
    }


def test_shadow_selection_requires_profile_stability_and_sample() -> None:
    summary = pd.DataFrame(
        [
            _summary_row("one", profile="responsive", sharpe=2.0),
            _summary_row("two", profile="balanced", sharpe=1.0),
            _summary_row(
                "duplicate",
                profile="conservative",
                sharpe=0.9,
                fingerprint="two",
            ),
            {
                **_summary_row(
                    "unstable",
                    profile="responsive",
                    fills=25,
                ),
                "formula": "gap_recovery",
            },
        ]
    )
    selected = _select_shadow_candidates(summary, max_strategies=12)

    assert set(selected["strategy_id"]) == {"one"}
    assert selected.iloc[0]["stable_profile_count"] == 3


def test_shadow_observation_is_append_only_and_has_no_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary = pd.DataFrame(
        [
            _summary_row("one", profile="responsive"),
            _summary_row("two", profile="balanced", sharpe=0.7),
        ]
    )
    output = tmp_path / "output" / "research" / "phase11_12"
    output.mkdir(parents=True)
    summary.to_parquet(output / "strategy-summary.parquet", index=False)
    index = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": range(100, 200),
            "high": range(102, 202),
            "low": range(99, 199),
            "close": range(101, 201),
            "volume": [1000] * 100,
        },
        index=index,
    )
    monkeypatch.setattr(
        "stocks.research.phase11_12._load_current_frames",
        lambda _root: {"1h": {"DBC": frame}},
    )
    monkeypatch.setattr(
        "stocks.research.phase11_12._signals",
        lambda frames, *_args: {
            symbol: pd.DataFrame(
                {"signal": True, "score": 1.0},
                index=value.index,
            )
            for symbol, value in frames.items()
        },
    )
    monkeypatch.setattr(
        "stocks.research.phase11_12._bar_freshness",
        lambda *_args: "FRESH_CLOSED_BAR",
    )

    report = phase11_12_observe(tmp_path)

    assert report["status"] == "GO"
    assert report["active_signal_count"] == 1
    assert report["EXECUTION_AUTHORITY"] == "NONE"
    assert report["automatic_orders"] == 0
    assert report["active_signals"][0]["illustrative_stop"] < 200
    assert report["active_signals"][0]["order_intents"] == []
    observations = list((output / "forward-observations").glob("*.json"))
    assert len(observations) == 1


def test_shadow_freshness_blocks_old_intraday_bars() -> None:
    now = datetime(2026, 7, 28, 20, 0, tzinfo=UTC)

    assert (
        _bar_freshness("2026-07-28T17:00:00Z", now, "1h")
        == "FRESH_CLOSED_BAR"
    )
    assert (
        _bar_freshness("2026-07-24T20:00:00Z", now, "1h")
        == "STALE_BAR_BLOCKED"
    )
