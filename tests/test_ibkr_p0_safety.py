from __future__ import annotations

import json
from pathlib import Path

from stocks.ibkr.p0_safety import (
    REQUIRED_SCENARIOS,
    inspect_p0_safety_gate,
    write_p0_safety_report,
)
from stocks.ibkr.p0_readiness import (
    inspect_p0_execution_readiness_gate,
    write_p0_execution_readiness,
)
from stocks.ibkr.paper_execution.restart_recovery import (
    restart_recovery_audit,
)


def test_p0_matrix_executes_every_required_offline_scenario(
    tmp_path: Path,
) -> None:
    report = write_p0_safety_report(tmp_path)
    gate = inspect_p0_safety_gate(tmp_path)

    assert report["status"] == "GO"
    assert report["broker_write_calls"] == 0
    assert report["network_calls"] == 0
    assert report["execution_authority"] == "NONE"
    assert set(report["scenario_statuses"]) == set(REQUIRED_SCENARIOS)
    assert set(report["scenario_statuses"].values()) == {"GO"}
    assert len(report["scenario_statuses"]) >= 80
    assert report["parallel_financial_ledger_created"] is False
    assert gate["status"] == "GO"


def test_regression_pass_does_not_imply_current_execution_readiness(
    tmp_path: Path,
) -> None:
    write_p0_safety_report(tmp_path)
    readiness = write_p0_execution_readiness(tmp_path)
    gate = inspect_p0_execution_readiness_gate(tmp_path)

    assert readiness["sub_gates"]["REGRESSION_MATRIX_PASS"] is True
    assert readiness["sub_gates"]["IBKR_CONNECTION_READY"] is False
    assert readiness["status"] == "NO_GO"
    assert readiness["marker"] == "NO_GO"
    assert gate["status"] == "NO_GO"
    assert gate["execution_authority"] == "NONE"


def test_p0_gate_rejects_tampered_attestation(tmp_path: Path) -> None:
    write_p0_safety_report(tmp_path)
    path = tmp_path / "output/ibkr/phase9/p0-safety-matrix.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenario_statuses"]["RESTART_REPLAY_IDEMPOTENT"] = "NO_GO"
    path.write_text(json.dumps(payload), encoding="utf-8")

    gate = inspect_p0_safety_gate(tmp_path)

    assert gate["status"] == "NO_GO"
    assert "P0_SAFETY_MATRIX_CONTENT_HASH_MISMATCH" in gate["blockers"]
    assert "P0_REQUIRED_SCENARIO_NOT_GO" in gate["blockers"]


def test_restart_audit_never_manufactures_go_without_evidence() -> None:
    report = restart_recovery_audit()

    assert report["status"] == "NO_GO"
    assert report["new_submission_blocked_until_reconciliation"] is False
