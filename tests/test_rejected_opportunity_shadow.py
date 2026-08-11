from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import stocks.research.rejected_shadow as rejected
from stocks.execution.idempotency import stable_hash


NOW = datetime(2026, 8, 7, 16, tzinfo=UTC)


def _episode(episode_id: str = "R1") -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "active_swing_forward_episode_v1",
        "episode_id": episode_id,
        "symbol": "AAA",
        "strategy_id": "S1",
        "timeframe": "1h",
        "state": "WATCHLIST_HARD_VETO_BLOCKED",
        "decision_timestamp": "2026-08-07T10:00:00+00:00",
        "signal_timestamp": "2026-08-07T10:00:00+00:00",
        "decision_contract": {
            "hard_vetoes": ["SPREAD_TOO_WIDE"],
            "soft_vetoes": [],
        },
        "context_snapshot": {"regime": "BULL"},
        "setup_snapshot": {
            "asset_class": "STOCK",
            "action": "BUY",
            "current_market_price": 100.0,
            "entry_zone_high": 101.0,
            "stop_loss": 95.0,
            "take_profit_1": 110.0,
            "take_profit_2": 115.0,
        },
        "entry_snapshot": {"tape": {"spread_bps": 20.0}},
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


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"open": 100, "high": 104, "low": 99, "close": 103},
            {"open": 103, "high": 108, "low": 101, "close": 107},
            {"open": 107, "high": 111, "low": 106, "close": 110},
        ],
        index=pd.to_datetime(
            [
                "2026-08-07T11:00:00Z",
                "2026-08-07T12:00:00Z",
                "2026-08-07T13:00:00Z",
            ],
            utc=True,
        ),
    )


def test_rejected_setup_is_evaluated_without_canonical_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_episode(tmp_path, _episode())
    canonical = (
        tmp_path
        / "data/market_context/private/entry-episode-outcomes.jsonl"
    )
    canonical.write_text('{"canonical":true}\n', encoding="utf-8")
    before = hashlib.sha256(canonical.read_bytes()).hexdigest()
    monkeypatch.setattr(
        rejected,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAA": _bars()}},
    )

    report = rejected.settle_rejected_opportunities(
        tmp_path,
        observed_at=NOW,
    )
    stored = rejected._read_jsonl(
        tmp_path
        / "data/market_context/private/rejected-opportunity-outcomes-v3.jsonl"
    )

    assert report["canonical_outcome_store_unchanged"] is True
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == before
    assert report["new_terminal_counterfactual_count"] == 1
    assert stored[0]["terminal_status"] == "TP1_EXIT"
    assert stored[0]["rejection_codes"] == ["SPREAD_TOO_WIDE"]
    assert stored[0]["canonical_training_eligible"] is False
    assert stored[0]["execution_authority"] == "NONE"


def test_rejected_shadow_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    _write_episode(tmp_path, _episode())
    monkeypatch.setattr(
        rejected,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAA": _bars()}},
    )

    first = rejected.settle_rejected_opportunities(
        tmp_path,
        observed_at=NOW,
    )
    second = rejected.settle_rejected_opportunities(
        tmp_path,
        observed_at=NOW,
    )

    assert first["new_terminal_counterfactual_count"] == 1
    assert second["new_terminal_counterfactual_count"] == 0
    assert second["terminal_counterfactual_count"] == 1


def test_long_signal_is_a_valid_counterfactual_entry(
    tmp_path: Path,
    monkeypatch,
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
    monkeypatch.setattr(
        rejected,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAA": _bars()}},
    )

    rejected.settle_rejected_opportunities(tmp_path, observed_at=NOW)
    stored = rejected._read_jsonl(
        tmp_path
        / "data/market_context/private/rejected-opportunity-outcomes-v3.jsonl"
    )

    assert stored[0]["terminal_status"] == "TP1_EXIT"
    assert stored[0]["net_R"] is not None


def test_avoid_action_is_only_bypassed_by_counterfactual_evaluator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    row = _episode()
    row["setup_snapshot"]["action"] = "AVOID"  # type: ignore[index]
    row["feature_snapshot_hash"] = stable_hash(
        {
            "decision_contract": row["decision_contract"],
            "context_snapshot": row["context_snapshot"],
            "setup_snapshot": row["setup_snapshot"],
            "entry_snapshot": row["entry_snapshot"],
        }
    )
    _write_episode(tmp_path, row)
    monkeypatch.setattr(
        rejected,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAA": _bars()}},
    )

    rejected.settle_rejected_opportunities(tmp_path, observed_at=NOW)
    stored = rejected._read_jsonl(
        tmp_path
        / "data/market_context/private/rejected-opportunity-outcomes-v3.jsonl"
    )

    assert stored[0]["terminal_status"] == "TP1_EXIT"
    assert stored[0]["counterfactual_evaluator_version"] == (
        "v3-explicit-long-envelope"
    )


def test_gate_attribution_requires_sample_before_policy_change() -> None:
    rows = [
        {
            "terminal": True,
            "rejection_codes": ["HTF_ADVERSE"],
            "would_fill": True,
            "net_R": 0.4,
        }
        for _ in range(10)
    ]

    report = rejected.build_gate_value_attribution(rows, generated_at=NOW)
    gate = report["gates"][0]

    assert gate["sample_status"] == "INSUFFICIENT_SAMPLE"
    assert gate["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert gate["posterior_success_mean"] == 0.91666667
    assert gate["posterior_success_10pct"] < gate["posterior_success_90pct"]
    assert gate["observational_assessment"] == (
        "INSUFFICIENT_SAMPLE_NO_POLICY_CHANGE"
    )
    assert report["automatic_gate_relaxation"] is False
    assert report["causal_gate_value_claimed"] is False


def test_gate_attribution_identifies_observationally_protective_gate() -> None:
    rows = [
        {
            "terminal": True,
            "rejection_codes": ["SPREAD_TOO_WIDE"],
            "would_fill": True,
            "net_R": -0.2,
        }
        for _ in range(30)
    ]

    report = rejected.build_gate_value_attribution(rows, generated_at=NOW)
    gate = report["gates"][0]

    assert gate["sample_status"] == "EVALUABLE"
    assert gate["estimated_protective_value_R"] == 0.2
    assert gate["evidence_status"] == "LIKELY_VALUE_ADD"
    assert gate["avoided_losses_R"] == 6.0
    assert gate["net_gate_contribution_R"] == 6.0
    assert gate["observational_assessment"] == (
        "OBSERVATIONALLY_PROTECTIVE_REQUIRES_ABLATION"
    )


def test_public_gate_artifact_has_no_symbol_or_order_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_episode(tmp_path, _episode())
    monkeypatch.setattr(
        rejected,
        "_load_current_frames",
        lambda *_args, **_kwargs: {"1h": {"AAA": _bars()}},
    )

    rejected.settle_rejected_opportunities(tmp_path, observed_at=NOW)
    public = (
        tmp_path
        / "output/research/active_swing/rejected_shadow/gate-attribution.json"
    ).read_text(encoding="utf-8")

    assert "AAA" not in public
    assert '"execution_authority": "NONE"' in public
    assert '"automatic_orders": 0' in public
