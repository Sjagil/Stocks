from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stocks.research.phase11_14 import (
    _attested_target_weights,
    _build_forward_performance,
    _evaluate_forward_episode,
    _execution_envelope,
    _forward_asset_metadata,
    _forward_session_audit,
    _load_forward_observations,
    _select_exploratory_observers,
    _select_validation_profile,
    phase11_14_schema,
    phase11_14_status,
    select_survivor_cohort,
)


def _source_summary() -> pd.DataFrame:
    rows = []
    for timeframe in ("1h", "4h", "1d", "1w"):
        for asset_class in ("STOCK", "ETF", "COMMODITY_PROXY"):
            for profile_number, profile in enumerate(
                ("responsive", "balanced", "conservative"),
                start=1,
            ):
                identity = f"{timeframe}-{asset_class}-{profile}"
                rows.append(
                    {
                        "strategy_id": identity,
                        "strategy_hash": f"hash-{identity}",
                        "formula": f"formula-{timeframe}-{asset_class}",
                        "timeframe": timeframe,
                        "profile": profile,
                        "asset_class": asset_class,
                        "status": "COMPLETE",
                        "CAGR": 0.05 + profile_number / 1000,
                        "Sharpe": 0.8 + profile_number / 100,
                        "period_profit_factor": 1.2,
                        "stress_50bps_profit_factor": 1.05,
                        "maximum_drawdown": -0.10,
                        "fill_count": 100,
                        "economic_outcome_fingerprint": identity,
                    }
                )
    rows.append(
        {
            "strategy_id": "unstable",
            "strategy_hash": "hash-unstable",
            "formula": "unstable",
            "timeframe": "1h",
            "profile": "responsive",
            "asset_class": "STOCK",
            "status": "COMPLETE",
            "CAGR": 0.2,
            "Sharpe": 2.0,
            "period_profit_factor": 1.5,
            "stress_50bps_profit_factor": 1.2,
            "maximum_drawdown": -0.05,
            "fill_count": 200,
            "economic_outcome_fingerprint": "unstable",
        }
    )
    return pd.DataFrame(rows)


def test_schema_is_selection_conditioned_and_authority_none(
    tmp_path: Path,
) -> None:
    report = phase11_14_schema(tmp_path)

    assert report["status"] == "GO"
    assert report["primary_timeframes"] == ["1h", "4h", "1d", "1w"]
    assert "REUSED_HISTORY" in report["evidence_scope"]
    assert report["automatic_promotion"] is False
    assert report["EXECUTION_AUTHORITY"] == "NONE"
    assert report["BROKER_CALLS"] == 0
    assert report["ORDER_CALLS"] == 0


def test_survivor_selection_is_diversified_stable_and_deterministic() -> None:
    source = _source_summary()
    first = select_survivor_cohort(source, max_candidates=12)
    second = select_survivor_cohort(
        source.sample(frac=1.0, random_state=42),
        max_candidates=12,
    )

    assert len(first) == 12
    assert set(first["timeframe"]) == {"1h", "4h", "1d", "1w"}
    assert set(first["asset_class"]) == {
        "STOCK",
        "ETF",
        "COMMODITY_PROXY",
    }
    assert first["stable_profile_count"].eq(3).all()
    assert "unstable" not in set(first["source_strategy_id"])
    assert first["qualification_strategy_id"].is_unique
    assert set(first["qualification_strategy_id"]) == set(
        second["qualification_strategy_id"]
    )


def test_validation_profile_selection_uses_validation_metrics_only(
    monkeypatch,
) -> None:
    profiles = {
        "responsive": {"marker": "responsive"},
        "balanced": {"marker": "balanced"},
        "conservative": {"marker": "conservative"},
    }
    values = {
        "responsive": (1.18, 0.08),
        "balanced": (1.20, 0.07),
        "conservative": (0.95, 0.03),
    }

    def fake_run(_frames, signals, **_kwargs):
        profit_factor, cagr = values[signals["marker"]]
        return {
            "metrics": {
                "period_profit_factor": profit_factor,
                "CAGR": cagr,
            },
            "fills": pd.DataFrame([{"shares": 1}]),
        }

    monkeypatch.setattr(
        "stocks.research.phase11_14._run_portfolio",
        fake_run,
    )
    profile, plateau, rows = _select_validation_profile(
        {},
        profiles,
        {
            "validation_start": "2020-01-01",
            "validation_end": "2020-06-30",
        },
    )

    assert profile == "balanced"
    assert plateau is True
    assert sum(bool(row["selected"]) for row in rows) == 1


def test_status_is_not_run_without_artifacts(tmp_path: Path) -> None:
    report = phase11_14_status(tmp_path)
    assert report["status"] == "NOT_RUN"
    assert report["EXECUTION_AUTHORITY"] == "NONE"


def test_exploratory_observer_accepts_cost_resilient_one_hour_research_only(
) -> None:
    rows = pd.DataFrame(
        [
            {
                "strategy_id": "ONE-HOUR-OBSERVE",
                "timeframe": "1h",
                "research_pass": True,
                "robust_pass": False,
                "portfolio_invariants_go": True,
                "combined_period_profit_factor": 1.09,
                "cost_50bps_combined_return": 0.01,
                "normal_cost_fill_count": 175,
                "positive_fold_count": 3,
                "fold_count": 6,
                "maximum_drawdown": -0.11,
            },
            {
                "strategy_id": "NEGATIVE-COST-STRESS",
                "timeframe": "1h",
                "research_pass": True,
                "robust_pass": False,
                "portfolio_invariants_go": True,
                "combined_period_profit_factor": 1.20,
                "cost_50bps_combined_return": -0.01,
                "normal_cost_fill_count": 300,
                "positive_fold_count": 4,
                "fold_count": 6,
                "maximum_drawdown": -0.10,
            },
            {
                "strategy_id": "ROBUST-ALREADY-SEPARATE",
                "timeframe": "4h",
                "research_pass": True,
                "robust_pass": True,
                "portfolio_invariants_go": True,
                "combined_period_profit_factor": 1.20,
                "cost_50bps_combined_return": 0.20,
                "normal_cost_fill_count": 300,
                "positive_fold_count": 5,
                "fold_count": 6,
                "maximum_drawdown": -0.10,
            },
        ]
    )

    selected = _select_exploratory_observers(rows)

    assert selected["strategy_id"].tolist() == ["ONE-HOUR-OBSERVE"]
    assert not bool(selected.iloc[0]["robust_pass"])


def test_target_weights_require_current_attestation_and_stay_bounded() -> None:
    active = [
        {"symbol": "A", "score": 3.0},
        {"symbol": "B", "score": 2.0},
        {"symbol": "C", "score": 1.0},
    ]
    weights = _attested_target_weights(active, {"A", "C"})

    assert weights == {"A": 0.25, "C": 0.25}
    assert sum(weights.values()) <= 1.0


def test_forward_metadata_uses_existing_point_in_time_security_master(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "data/research/phase11_4/private/security-master.parquet"
    )
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "security_id": "SEC-1",
                "ticker": "TEST",
                "sector": "Technology",
                "industry": "Semiconductors",
                "is_delisted": False,
                "source_hash": "SOURCE-HASH",
            }
        ]
    ).to_parquet(path, index=False)

    metadata = _forward_asset_metadata(
        tmp_path,
        observed_at=datetime(2099, 1, 1, tzinfo=UTC),
    )

    assert metadata["TEST"]["sector"] == "Technology"
    assert metadata["TEST"]["industry"] == "Semiconductors"
    assert metadata["TEST"]["asset_metadata_status"] == (
        "AVAILABLE_AT_DECISION"
    )
    assert metadata["TEST"]["asset_metadata_source"] == (
        "SECURITY_MASTER_POINT_IN_TIME"
    )
    assert metadata["TEST"]["asset_metadata_source_hash"] == "SOURCE-HASH"


def test_forward_metadata_hashes_existing_broad_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config/universes/broad_multi_asset_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"groups":[{"group":"ETF","asset_type":"EQUITY_ETF",'
        '"sleeve":"core","instruments":{"SPY":'
        '{"sector":"BROAD_MARKET"}}}]}',
        encoding="utf-8",
    )

    metadata = _forward_asset_metadata(
        tmp_path,
        observed_at=datetime(2099, 1, 1, tzinfo=UTC),
    )

    assert metadata["SPY"]["asset_metadata_source"] == (
        "BROAD_MULTI_ASSET_MANIFEST"
    )
    assert metadata["SPY"]["asset_metadata_source_hash"]
    assert metadata["SPY"]["asset_metadata_source_accepted_at"]


def test_forward_metadata_blocks_ambiguous_active_classification(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "data/research/phase11_4/private/security-master.parquet"
    )
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "security_id": "SEC-1",
                "ticker": "TEST",
                "sector": "Technology",
                "industry": "Semiconductors",
                "is_delisted": False,
                "source_hash": "A",
            },
            {
                "security_id": "SEC-2",
                "ticker": "TEST",
                "sector": "Industrials",
                "industry": "Equipment",
                "is_delisted": False,
                "source_hash": "B",
            },
        ]
    ).to_parquet(path, index=False)

    metadata = _forward_asset_metadata(
        tmp_path,
        observed_at=datetime(2099, 1, 1, tzinfo=UTC),
    )

    assert metadata["TEST"]["sector"] == "UNAVAILABLE_AT_DECISION"
    assert metadata["TEST"]["asset_metadata_status"] == (
        "AMBIGUOUS_CLASSIFICATION_BLOCKED"
    )


def test_forward_audit_counts_only_post_boundary_observed_bars() -> None:
    boundary = {
        "qualification_hash": "HASH",
        "robust_strategy_ids": ["A", "B"],
        "data_end_by_strategy": {
            "A": "2026-07-24T00:00:00Z",
            "B": "2026-07-24T00:00:00Z",
        },
    }
    audit = _forward_session_audit(
        boundary,
        [
            {
                "observed_at": "2026-07-26T00:00:00Z",
                "observations": [
                    {
                        "strategy_id": "A",
                        "closed_bar_timestamp": "2026-07-25T00:00:00Z",
                    },
                    {
                        "strategy_id": "B",
                        "closed_bar_timestamp": "2026-07-24T00:00:00Z",
                    },
                ],
            },
            {
                "observed_at": "2026-07-26T00:00:00Z",
                "observations": [
                    {
                        "strategy_id": "B",
                        "closed_bar_timestamp": "2026-07-27T00:00:00Z",
                    }
                ],
            },
        ],
        observed_at=datetime(2026, 7, 26, tzinfo=UTC),
    )

    assert audit["status"] == "INDEPENDENT_FORWARD_SESSION_PARTIAL"
    assert audit["completed_strategy_count"] == 1
    assert audit["per_strategy"]["A"]["complete"] is True
    assert audit["per_strategy"]["B"]["complete"] is False
    assert audit["same_or_prior_bar_counted"] is False
    assert audit["EXECUTION_AUTHORITY"] == "NONE"


def test_execution_envelope_is_closed_bar_causal_and_family_aware() -> None:
    index = pd.date_range("2026-07-01", periods=30, freq="4h")
    frame = pd.DataFrame(
        {
            "open": range(100, 130),
            "high": [value + 2 for value in range(100, 130)],
            "low": [value - 2 for value in range(100, 130)],
            "close": [value + 1 for value in range(100, 130)],
        },
        index=index,
    )
    breakout = _execution_envelope(
        frame,
        common_close=index[-2],
        formula="choppiness_breakout",
        timeframe="4h",
    )
    pullback = _execution_envelope(
        frame,
        common_close=index[-2],
        formula="rsi14_trend_pullback",
        timeframe="4h",
    )

    assert breakout["execution_envelope_status"] == "GO"
    assert breakout["entry_reference"] == 129
    assert breakout["stop_loss"] < breakout["entry_reference"]
    assert breakout["take_profit_1"] > breakout["entry_reference"]
    assert breakout["reward_multiple"] == 3.0
    assert pullback["reward_multiple"] == 1.5
    assert breakout["maximum_holding_bars"] == 60


def _forward_signal(**overrides) -> dict:
    signal = {
        "symbol": "TEST",
        "currently_attested": True,
        "execution_envelope_status": "GO",
        "initial_risk_per_share": 5.0,
        "reward_multiple": 2.0,
        "maximum_holding_bars": 3,
    }
    signal.update(overrides)
    return signal


def _forward_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [99.0, 100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 111.0, 104.0],
            "low": [98.0, 99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 109.0, 103.0],
        },
        index=pd.date_range(
            "2026-07-30T16:00:00Z",
            periods=4,
            freq="h",
        ),
    )


def test_forward_loader_accepts_v1_and_v2_only(tmp_path: Path) -> None:
    root = tmp_path / "forward-observations"
    root.mkdir()
    for name, schema in (
        ("a", "phase11_14_forward_observation_v1"),
        ("b", "phase11_14_forward_observation_v2"),
        ("c", "phase11_14_forward_observation_v3"),
        ("d", "unrelated"),
    ):
        (root / f"{name}.json").write_text(
            '{"schema":"' + schema + '"}',
            encoding="utf-8",
        )

    loaded = _load_forward_observations(tmp_path)

    assert {row["schema"] for row in loaded} == {
        "phase11_14_forward_observation_v1",
        "phase11_14_forward_observation_v2",
        "phase11_14_forward_observation_v3",
    }


def test_forward_episode_enters_next_bar_and_closes_at_target() -> None:
    result = _evaluate_forward_episode(
        strategy_id="S1",
        formula="test",
        timeframe="1h",
        asset_class="STOCK",
        symbol="TEST",
        signal_time=pd.Timestamp("2026-07-30T16:00:00Z"),
        signal=_forward_signal(
            sector="Technology",
            industry="Semiconductors",
            asset_metadata_status="AVAILABLE_AT_DECISION",
            asset_metadata_source="SECURITY_MASTER_POINT_IN_TIME",
            asset_metadata_source_hash="SOURCE-HASH",
            asset_metadata_source_accepted_at="2026-07-29T00:00:00+00:00",
        ),
        frame=_forward_frame(),
        evidence_end=pd.Timestamp("2026-07-30T20:00:00Z"),
        qualification_hash="Q",
        cost_bps_per_side=10.0,
    )

    assert result["entry_timestamp"] == "2026-07-30T17:00:00+00:00"
    assert result["entry_price"] == 100.0
    assert result["would_fill"] is True
    assert result["fill_timestamp"] == result["entry_timestamp"]
    assert result["fill_price"] == result["entry_price"]
    assert result["fill_fraction"] == 1.0
    assert result["outcome_status"] == "CLOSED_TARGET"
    assert result["exit_timestamp"] == "2026-07-30T18:00:00+00:00"
    assert result["gross_return"] == 0.1
    assert result["net_return"] == 0.098
    assert result["gross_R"] == 2.0
    assert result["net_R"] == 1.96
    assert result["first_barrier_hit"] == "TARGET"
    assert result["holding_duration_seconds"] == 3600.0
    assert result["maximum_adverse_excursion_R"] == -0.2
    assert result["maximum_favorable_excursion_R"] == 2.2
    assert result["estimated_commission"] is None
    assert result["estimated_slippage"] is None
    assert result["cost_attribution_status"] == (
        "BUNDLED_BPS_NOT_SEPARATELY_OBSERVED"
    )
    assert result["time_to_mfe_seconds"] == 3600.0
    assert result["time_to_mae_seconds"] == 0.0
    assert result["time_to_stop_seconds"] is None
    assert result["spread_evidence_status"] == "UNAVAILABLE_INCLUDED_IN_COST_BUNDLE"
    assert result["market_regime"] == "UNAVAILABLE_AT_DECISION"
    assert result["sector"] == "Technology"
    assert result["industry"] == "Semiconductors"
    assert result["asset_metadata_status"] == "AVAILABLE_AT_DECISION"
    assert result["asset_metadata_source"] == (
        "SECURITY_MASTER_POINT_IN_TIME"
    )
    assert result["exit_quality"] == "BAR_BASED_CAUSAL"
    assert result["EXECUTION_AUTHORITY"] == "NONE"


def test_forward_episode_uses_pessimistic_stop_when_both_hit() -> None:
    frame = _forward_frame()
    frame.loc[pd.Timestamp("2026-07-30T17:00:00Z"), ["high", "low"]] = [
        111.0,
        94.0,
    ]
    result = _evaluate_forward_episode(
        strategy_id="S1",
        formula="test",
        timeframe="1h",
        asset_class="STOCK",
        symbol="TEST",
        signal_time=pd.Timestamp("2026-07-30T16:00:00Z"),
        signal=_forward_signal(),
        frame=frame,
        evidence_end=pd.Timestamp("2026-07-30T20:00:00Z"),
        qualification_hash="Q",
        cost_bps_per_side=10.0,
    )

    assert result["outcome_status"] == "CLOSED_STOP"
    assert result["exit_reason"] == "STOP_FIRST_SAME_BAR_AMBIGUITY"
    assert result["net_return"] == -0.052
    assert result["gross_R"] == -1.0
    assert result["net_R"] == -1.04
    assert result["first_barrier_hit"] == (
        "STOP_AND_TARGET_SAME_BAR_STOP_FIRST"
    )
    assert result["time_to_stop_seconds"] == 0.0
    assert result["exit_quality"] == "CONSERVATIVE_SAME_BAR_PATH"


def test_forward_episode_never_enters_before_append_only_decision() -> None:
    result = _evaluate_forward_episode(
        strategy_id="S1",
        formula="test",
        timeframe="1h",
        asset_class="STOCK",
        symbol="TEST",
        signal_time=pd.Timestamp("2026-07-30T16:00:00Z"),
        decision_time=pd.Timestamp("2026-07-30T18:30:00Z"),
        signal=_forward_signal(),
        frame=_forward_frame(),
        evidence_end=pd.Timestamp("2026-07-30T20:00:00Z"),
        qualification_hash="Q",
        cost_bps_per_side=10.0,
    )

    assert result["decision_timestamp"] == "2026-07-30T18:30:00+00:00"
    assert result["entry_timestamp"] == "2026-07-30T19:00:00+00:00"


def test_persistent_forward_signal_creates_one_episode() -> None:
    boundary = {
        "qualification_hash": "Q",
        "robust_strategy_ids": ["S1"],
        "data_end_by_strategy": {"S1": "2026-07-30T15:00:00Z"},
    }
    observations = []
    for hour in (16, 17):
        observations.append(
            {
                "schema": "phase11_14_forward_observation_v2",
                "observation_id": f"O{hour}",
                "observed_at": f"2026-07-30T{hour + 1}:30:00Z",
                "observations": [
                    {
                        "strategy_id": "S1",
                        "formula": "test",
                        "timeframe": "1h",
                        "asset_class": "STOCK",
                        "closed_bar_timestamp": (
                            f"2026-07-30T{hour}:00:00Z"
                        ),
                        "raw_active_signals": [_forward_signal()],
                    }
                ],
            }
        )
    report = _build_forward_performance(
        boundary,
        observations,
        {"1h": {"TEST": _forward_frame()}},
        observed_at=datetime(2026, 7, 30, 20, tzinfo=UTC),
    )

    assert report["counts"]["episode_count"] == 1
    assert report["counts"]["closed_episode_count"] == 1
    assert report["aggregate"]["sample_status"] == "INSUFFICIENT_SAMPLE"
    assert report["automatic_orders"] == 0
    assert report["EXECUTION_AUTHORITY"] == "NONE"
