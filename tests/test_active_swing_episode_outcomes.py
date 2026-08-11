from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import stocks.context.episode_outcomes as outcomes
from stocks.execution.idempotency import stable_hash


NOW = datetime(2026, 8, 4, 16, tzinfo=UTC)


def _episode(*, episode_id: str = "E1") -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "active_swing_forward_episode_v1",
        "episode_id": episode_id,
        "symbol": "AAPL",
        "strategy_id": "S1",
        "timeframe": "1h",
        "decision_timestamp": "2026-08-03T10:00:00+00:00",
        "signal_timestamp": "2026-08-03T10:00:00+00:00",
        "decision_contract": {"status": "GO"},
        "context_snapshot": {"regime": "RISK_ON"},
        "setup_snapshot": {
            "asset_class": "STOCK",
            "action": "BUY",
            "current_market_price": 100.0,
            "entry_zone_high": 101.0,
            "stop_loss": 95.0,
            "take_profit_1": 110.0,
            "take_profit_2": 115.0,
            "estimated_transaction_costs_eur": 0.5,
        },
        "entry_snapshot": {"tape": {"spread_bps": 4.0}},
        "proposed_order": {"limit_price": 100.0},
    }
    row["feature_snapshot_hash"] = stable_hash(
        {
            "decision_contract": row["decision_contract"],
            "context_snapshot": row["context_snapshot"],
            "setup_snapshot": row["setup_snapshot"],
            "entry_snapshot": row["entry_snapshot"],
        }
    )
    return row


def _write_episode(root: Path, row: dict[str, object]) -> None:
    path = root / "data/market_context/private/entry-episodes.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"open": open_, "high": high, "low": low, "close": close}
            for _, open_, high, low, close in rows
        ],
        index=pd.to_datetime([row[0] for row in rows], utc=True),
    )


def test_episode_closes_once_at_tp1_and_measures_after_fill(
    tmp_path: Path, monkeypatch
) -> None:
    _write_episode(tmp_path, _episode())
    frame = _frame(
        [
            ("2026-08-03T11:00:00Z", 100, 104, 99, 103),
            ("2026-08-03T12:00:00Z", 103, 108, 101, 107),
            ("2026-08-03T13:00:00Z", 107, 111, 106, 110),
        ]
    )
    monkeypatch.setattr(
        outcomes,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAPL": frame}},
    )

    first = outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)
    second = outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)

    stored = outcomes._read_jsonl(
        tmp_path
        / "data/market_context/private/entry-episode-outcomes.jsonl"
    )
    assert first["new_terminal_episode_count"] == 1
    assert first["completion_ratio"] == 1.0
    assert second["new_terminal_episode_count"] == 0
    assert len(stored) == 1
    assert stored[0]["terminal_status"] == "TP1_EXIT"
    assert stored[0]["maximum_favourable_excursion"] == 2.2
    assert stored[0]["maximum_adverse_excursion"] == 0.2
    assert stored[0]["time_to_mfe_seconds"] == 7200
    assert stored[0]["time_to_mae_seconds"] == 3600
    assert stored[0]["time_to_first_barrier_seconds"] == 7200
    assert stored[0]["time_to_stop_seconds"] is None
    assert stored[0]["gross_exit_capture_ratio"] == round(2.0 / 2.2, 8)
    assert stored[0]["market_regime"] == "RISK_ON"
    assert stored[0]["sector"] == "UNAVAILABLE_AT_DECISION"
    assert stored[0]["decision_context_hash"]
    assert stored[0]["hypothetical_spread_at_decision_bps"] == 4.0
    assert stored[0]["realized_spread_at_fill_bps"] is None
    assert (
        stored[0]["spread_observation_status"]
        == "DECISION_PROXY_ONLY_FILL_SPREAD_UNAVAILABLE"
    )
    assert stored[0]["net_R"] < stored[0]["gross_R"]


def test_long_action_uses_the_same_long_only_lifecycle(
    tmp_path: Path, monkeypatch
) -> None:
    row = _episode()
    row["setup_snapshot"]["action"] = "LONG"  # type: ignore[index]
    row["feature_snapshot_hash"] = stable_hash(
        {
            "decision_contract": row["decision_contract"],
            "context_snapshot": row["context_snapshot"],
            "setup_snapshot": row["setup_snapshot"],
            "entry_snapshot": row["entry_snapshot"],
        }
    )
    _write_episode(tmp_path, row)
    frame = _frame(
        [
            ("2026-08-03T11:00:00Z", 100, 104, 99, 103),
            ("2026-08-03T12:00:00Z", 103, 108, 101, 107),
            ("2026-08-03T13:00:00Z", 107, 111, 106, 110),
        ]
    )
    monkeypatch.setattr(
        outcomes,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAPL": frame}},
    )

    outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)
    stored = outcomes._read_jsonl(
        tmp_path
        / "data/market_context/private/entry-episode-outcomes.jsonl"
    )

    assert stored[0]["terminal_status"] == "TP1_EXIT"


def test_contract_only_execution_veto_remains_research_observable(
    tmp_path: Path, monkeypatch
) -> None:
    row = _episode()
    row["state"] = "WATCHLIST_HARD_VETO_BLOCKED"
    row["decision_contract"] = {
        "hard_veto_pass": False,
        "hard_vetoes": ["CONTRACT_IDENTITY_REQUIRED"],
        "research_observation_eligible": True,
        "research_observation_blockers": [],
        "brokerability_status": "BROKERABILITY_BLOCKED_CONTRACT_IDENTITY",
        "gates": {"contract_resolved": False},
    }
    row["feature_snapshot_hash"] = stable_hash(
        {
            "decision_contract": row["decision_contract"],
            "context_snapshot": row["context_snapshot"],
            "setup_snapshot": row["setup_snapshot"],
            "entry_snapshot": row["entry_snapshot"],
        }
    )
    _write_episode(tmp_path, row)
    frame = _frame(
        [
            ("2026-08-03T11:00:00Z", 100, 104, 99, 103),
            ("2026-08-03T12:00:00Z", 103, 108, 101, 107),
            ("2026-08-03T13:00:00Z", 107, 111, 106, 110),
        ]
    )
    monkeypatch.setattr(
        outcomes,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAPL": frame}},
    )

    report = outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)
    stored = outcomes._read_jsonl(
        tmp_path
        / "data/market_context/private/entry-episode-outcomes.jsonl"
    )

    assert report["research_observation_eligible_episode_count"] == 1
    assert report["brokerability_blocked_research_eligible_count"] == 1
    assert stored[0]["terminal_status"] == "TP1_EXIT"
    assert stored[0]["label_source"] == "COUNTERFACTUAL_BAR_PATH_OBSERVATION"
    assert stored[0]["canonical_close"] is False
    assert stored[0]["canonical_fill_evidence"] is False
    assert stored[0]["execution_authority"] == "NONE"


def test_immutable_hard_veto_outcome_gets_append_only_research_revision(
    tmp_path: Path, monkeypatch
) -> None:
    row = _episode()
    row["state"] = "WATCHLIST_HARD_VETO_BLOCKED"
    row["decision_contract"] = {
        "hard_veto_pass": False,
        "hard_vetoes": ["CONTRACT_IDENTITY_REQUIRED"],
        "research_observation_eligible": True,
        "brokerability_status": "BROKERABILITY_BLOCKED_CONTRACT_IDENTITY",
        "gates": {"contract_resolved": False},
    }
    row["feature_snapshot_hash"] = stable_hash(
        {
            "decision_contract": row["decision_contract"],
            "context_snapshot": row["context_snapshot"],
            "setup_snapshot": row["setup_snapshot"],
            "entry_snapshot": row["entry_snapshot"],
        }
    )
    _write_episode(tmp_path, row)
    private = tmp_path / "data/market_context/private"
    original = {
        "schema": "active_swing_forward_episode_outcome_v1",
        "episode_id": "E1",
        "terminal": True,
        "terminal_status": "INVALIDATED_BEFORE_ENTRY",
        "outcome_classification": "HARD_VETO_BEFORE_ENTRY",
        "outcome_hash": "ORIGINAL",
    }
    (private / "entry-episode-outcomes.jsonl").write_text(
        json.dumps(original) + "\n", encoding="utf-8"
    )
    frame = _frame(
        [
            ("2026-08-03T11:00:00Z", 100, 104, 99, 103),
            ("2026-08-03T12:00:00Z", 103, 108, 101, 107),
            ("2026-08-03T13:00:00Z", 107, 111, 106, 110),
        ]
    )
    monkeypatch.setattr(
        outcomes,
        "_load_episode_frames",
        lambda *_args, **_kwargs: {"1h": {"AAPL": frame}},
    )

    first = outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)
    second = outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)
    unchanged = outcomes._read_jsonl(
        private / "entry-episode-outcomes.jsonl"
    )
    revisions = outcomes._read_jsonl(
        private / "entry-episode-outcome-revisions.jsonl"
    )

    assert unchanged == [original]
    assert first["new_research_revision_count"] == 1
    assert second["new_research_revision_count"] == 0
    assert len(revisions) == 1
    assert revisions[0]["terminal_status"] == "TP1_EXIT"
    assert revisions[0]["supersedes_outcome_hash"] == "ORIGINAL"
    assert revisions[0]["canonical_evidence_replaced"] is False
    assert revisions[0]["execution_authority"] == "NONE"


def test_same_bar_stop_and_target_is_not_invented(
    tmp_path: Path, monkeypatch
) -> None:
    _write_episode(tmp_path, _episode())
    frame = _frame(
        [("2026-08-03T11:00:00Z", 100, 111, 94, 102)]
    )
    monkeypatch.setattr(
        outcomes,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAPL": frame}},
    )

    report = outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)
    stored = outcomes._read_jsonl(
        tmp_path
        / "data/market_context/private/entry-episode-outcomes.jsonl"
    )

    assert report["intrabar_path_ambiguous_count"] == 1
    assert stored[0]["terminal_status"] == "DATA_FAILURE"
    assert stored[0]["outcome_classification"] == "INTRABAR_PATH_AMBIGUOUS"
    assert stored[0]["net_R"] is None


def test_unfilled_expired_episode_has_no_pnl(
    tmp_path: Path, monkeypatch
) -> None:
    _write_episode(tmp_path, _episode())
    frame = _frame(
        [
            (f"2026-08-03T{hour:02d}:00:00Z", 105, 107, 103, 106)
            for hour in (11, 12, 13, 14)
        ]
    )
    monkeypatch.setattr(
        outcomes,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAPL": frame}},
    )

    outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)
    stored = outcomes._read_jsonl(
        tmp_path
        / "data/market_context/private/entry-episode-outcomes.jsonl"
    )

    assert stored[0]["terminal_status"] == "NO_FILL_EXPIRED"
    assert stored[0]["would_fill"] is False
    assert stored[0]["gross_R"] is None
    assert stored[0]["net_R"] is None


def test_feature_snapshot_mutation_is_terminal_data_failure(
    tmp_path: Path, monkeypatch
) -> None:
    row = _episode()
    row["setup_snapshot"]["stop_loss"] = 96.0  # type: ignore[index]
    _write_episode(tmp_path, row)
    monkeypatch.setattr(
        outcomes,
        "_load_current_frames",
        lambda *_args, **_kwargs: {},
    )

    outcomes.settle_entry_episodes(tmp_path, observed_at=NOW)
    stored = outcomes._read_jsonl(
        tmp_path
        / "data/market_context/private/entry-episode-outcomes.jsonl"
    )

    assert stored[0]["terminal_status"] == "DATA_FAILURE"
    assert stored[0]["outcome_classification"] == "FEATURE_SNAPSHOT_MUTATED"


def test_legacy_episodes_and_their_outcomes_are_quarantined(
    tmp_path: Path, monkeypatch
) -> None:
    legacy = {"episode_id": "OLD", "symbol": "AAPL"}
    current = _episode(episode_id="NEW")
    path = tmp_path / "data/market_context/private/entry-episodes.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(legacy) + "\n" + json.dumps(current) + "\n",
        encoding="utf-8",
    )
    outcome_path = path.parent / "entry-episode-outcomes.jsonl"
    outcome_path.write_text(
        json.dumps({"episode_id": "OLD", "terminal_status": "DATA_FAILURE"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        outcomes,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {}},
    )

    report = outcomes.settle_entry_episodes(
        tmp_path,
        observed_at=datetime(2026, 8, 3, 11, tzinfo=UTC),
    )

    assert report["legacy_episode_count"] == 1
    assert report["episode_count"] == 1
    assert not outcome_path.exists()
    assert list((path.parent / "quarantine").glob("*.jsonl"))
