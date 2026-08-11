from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

from stocks.research.autopilot.runtime import run_once, runtime_command


def _patch_cycle_dependencies(monkeypatch, *, queue_report=None) -> list[int]:
    batches: list[int] = []
    monkeypatch.setattr(
        "stocks.research.autopilot.runtime.research_status",
        lambda: {"ledger_counts": {}},
    )
    monkeypatch.setattr(
        "stocks.research.autopilot.runtime.autopilot_generate",
        lambda **_kwargs: {"status": "GO", "generated_count": 1},
    )
    monkeypatch.setattr(
        "stocks.research.autopilot.runtime.autopilot_campaign",
        lambda *_args, **_kwargs: {
            "status": "DATA_BLOCKED",
            "trial_count": 2,
            "complete_trial_count": 0,
            "eligibility": {"status": "PIT_ELIGIBILITY_UNAVAILABLE"},
        },
    )
    monkeypatch.setattr(
        "stocks.research.autopilot.runtime.recover_survivors",
        lambda _root: {
            "status": "GO",
            "survivor_count": 0,
            "classification_counts": {},
        },
    )

    def fake_queue(_root, **kwargs):
        batch = int(kwargs["max_strategies"])
        batches.append(batch)
        if queue_report is not None:
            return queue_report
        return {
            "status": "GO",
            "catalog_count": batch,
            "trial_count": batch * 2,
            "expected_trial_count": batch * 2,
            "pending_before": 100,
            "pending_after": 100 - batch,
            "EXECUTION_AUTHORITY": "NONE",
            "BROKER_CALLS": 0,
            "ORDER_CALLS": 0,
        }

    monkeypatch.setattr(
        "stocks.research.autopilot.runtime.run_phase11_12",
        fake_queue,
    )
    return batches


def test_runtime_start_pause_resume_stop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "stocks.research.autopilot.runtime.research_status",
        lambda: {"ledger_counts": {}},
    )
    assert runtime_command(tmp_path, "start")["enabled"] is True
    assert runtime_command(tmp_path, "pause")["paused"] is True
    assert runtime_command(tmp_path, "resume")["paused"] is False
    assert runtime_command(tmp_path, "stop")["enabled"] is False


def test_runtime_single_instance_lock(tmp_path: Path) -> None:
    lock_path = (
        tmp_path / "data" / "research" / "autopilot" / "private" / "autopilot.lock"
    )
    lock_path.parent.mkdir(parents=True)
    with FileLock(lock_path, timeout=0):
        result = run_once(tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "AUTOPILOT_SINGLE_INSTANCE_LOCKED"
    assert result["orders_generated"] == 0


def test_runtime_budget_exhaustion_is_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "data" / "research" / "autopilot" / "private"
    private.mkdir(parents=True)
    private.joinpath("runtime-state.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "paused": False,
                "budget_date": datetime.now(UTC).date().isoformat(),
                "new_strategies_today": 50,
                "backtests_today": 1000,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "stocks.research.autopilot.runtime.research_status",
        lambda: {"ledger_counts": {}},
    )
    result = run_once(tmp_path)
    assert result["status"] == "GO"
    assert result["scheduler_status"] == "DAILY_BUDGET_EXHAUSTED"
    assert result["orders_generated"] == 0


def test_runtime_never_auto_promotes_live(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "stocks.research.autopilot.runtime.research_status",
        lambda: {"ledger_counts": {}},
    )
    status = runtime_command(tmp_path, "status")
    assert status["AUTOPILOT_AUTO_LIVE_PROMOTION"] is False
    assert status["execution_authority"] == "NONE"


def test_runtime_advances_phase11_queue_with_shared_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batches = _patch_cycle_dependencies(monkeypatch)
    monkeypatch.setenv("AUTOPILOT_REAL_BACKTESTS_PER_CYCLE", "1")
    monkeypatch.setenv("AUTOPILOT_PHASE11_12_DNA_PER_CYCLE", "3")

    result = run_once(tmp_path)

    assert result["status"] == "GO"
    assert batches == [3]
    assert result["campaign"]["trial_count"] == 2
    assert result["phase11_12_queue"]["trial_count"] == 6
    assert result["checkpoint"]["backtests_today"] == 8
    assert result["checkpoint"]["phase11_12_dna_today"] == 3
    assert result["checkpoint"]["phase11_12_trials_today"] == 6
    assert result["execution_authority"] == "NONE"
    assert result["broker_calls"] == 0
    assert result["orders_generated"] == 0


def test_phase11_queue_continues_after_generation_budget_exhaustion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batches = _patch_cycle_dependencies(monkeypatch)
    monkeypatch.setenv("AUTOPILOT_MAX_NEW_STRATEGIES_PER_DAY", "1")
    monkeypatch.setenv("AUTOPILOT_PHASE11_12_DNA_PER_CYCLE", "4")
    private = tmp_path / "data/research/autopilot/private"
    private.mkdir(parents=True)
    private.joinpath("runtime-state.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "paused": False,
                "budget_date": datetime.now(UTC).date().isoformat(),
                "new_strategies_today": 1,
                "backtests_today": 0,
            }
        ),
        encoding="utf-8",
    )

    result = run_once(tmp_path)

    assert result["status"] == "GO"
    assert result["generated"]["status"] == "SKIPPED"
    assert result["campaign"]["status"] == "SKIPPED"
    assert batches == [4]
    assert result["checkpoint"]["backtests_today"] == 8


def test_phase11_queue_is_capped_by_remaining_daily_trial_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batches = _patch_cycle_dependencies(monkeypatch)
    monkeypatch.setenv("AUTOPILOT_MAX_NEW_STRATEGIES_PER_DAY", "1")
    monkeypatch.setenv("AUTOPILOT_MAX_TOTAL_BACKTESTS_PER_DAY", "10")
    monkeypatch.setenv("AUTOPILOT_PHASE11_12_DNA_PER_CYCLE", "25")
    private = tmp_path / "data/research/autopilot/private"
    private.mkdir(parents=True)
    private.joinpath("runtime-state.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "paused": False,
                "budget_date": datetime.now(UTC).date().isoformat(),
                "new_strategies_today": 1,
                "backtests_today": 7,
            }
        ),
        encoding="utf-8",
    )

    result = run_once(tmp_path)

    assert batches == [1]
    assert result["phase11_12_queue"]["trial_count"] == 2
    assert result["checkpoint"]["backtests_today"] == 9


def test_phase11_queue_authority_violation_is_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batches = _patch_cycle_dependencies(
        monkeypatch,
        queue_report={
            "status": "GO",
            "catalog_count": 1,
            "trial_count": 2,
            "expected_trial_count": 2,
            "pending_before": 10,
            "pending_after": 9,
            "EXECUTION_AUTHORITY": "PAPER",
            "BROKER_CALLS": 0,
            "ORDER_CALLS": 0,
        },
    )
    monkeypatch.setenv("AUTOPILOT_MAX_NEW_STRATEGIES_PER_DAY", "1")
    monkeypatch.setenv("AUTOPILOT_PHASE11_12_DNA_PER_CYCLE", "1")
    private = tmp_path / "data/research/autopilot/private"
    private.mkdir(parents=True)
    private.joinpath("runtime-state.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "paused": False,
                "budget_date": datetime.now(UTC).date().isoformat(),
                "new_strategies_today": 1,
                "backtests_today": 0,
            }
        ),
        encoding="utf-8",
    )

    result = run_once(tmp_path)

    assert batches == [1]
    assert result["status"] == "DEGRADED"
    assert result["failure_count"] == 1
    assert result["failures"][0]["stage"] == "PHASE11_12_QUEUE"
    assert "EXECUTION_AUTHORITY_VIOLATION" in result["failures"][0]["error"]
    assert result["checkpoint"]["backtests_today"] == 0
    assert result["execution_authority"] == "NONE"
