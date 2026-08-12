from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from stocks.application.phase_gates import phase1_freeze_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_powershell(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        raise RuntimeError("PowerShell executable not found")

    command = [executable, "-NoProfile"]

    if os.name == "nt":
        command.extend(["-ExecutionPolicy", "Bypass"])

    command.extend(["-File", script, *args])

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_disconnect_artifact_verifier_rejects_no_go_fixture() -> None:
    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_no_go.json"),
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "status must be GO" in payload["errors"]


def test_disconnect_artifact_verifier_accepts_go_fixture() -> None:
    project_copy = PROJECT_ROOT
    result = run_powershell(
        str(project_copy / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(project_copy / "tests" / "fixtures" / "disconnect_drill_go.json"),
        "-NoWriteFreezeReport",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "GO"
    assert payload["freeze_report"] is None
    assert "main.py" in payload["mutable_application_entrypoint_hash"]
    assert "src/stocks/application/config.py" in payload["immutable_phase1_service_hashes"]
    assert "src/stocks/ibkr/connection.py" in payload["immutable_phase1_service_hashes"]
    assert "src/stocks/ibkr/client.py" in payload["immutable_phase1_service_hashes"]
    assert "src/stocks/ibkr/callbacks.py" in payload["immutable_phase1_service_hashes"]
    assert "src/stocks/ibkr/errors.py" in payload["immutable_phase1_service_hashes"]
    assert "src/stocks/ibkr/health.py" in payload["immutable_phase1_service_hashes"]
    assert "src/stocks/application/context.py" not in payload["immutable_phase1_service_hashes"]
    assert len(payload["mutable_application_entrypoint_hash"]["main.py"]) == 64


def test_disconnect_artifact_verifier_rejects_short_go_artifact(tmp_path: Path) -> None:
    fixture = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text())
    fixture["seconds"] = 60.0
    artifact = tmp_path / "short_disconnect_drill_go.json"
    artifact.write_text(json.dumps(fixture), encoding="utf-8")

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-NoWriteFreezeReport",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "seconds must be at least 180 for a Phase 1 freeze" in payload["errors"]


def test_disconnect_artifact_verifier_rejects_gateway_paper_port(tmp_path: Path) -> None:
    fixture = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text())
    fixture["port"] = 4002
    artifact = tmp_path / "gateway_paper_disconnect_drill_go.json"
    artifact.write_text(json.dumps(fixture), encoding="utf-8")

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-NoWriteFreezeReport",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "port must be 7497 for the TWS paper Phase 1 freeze drill" in payload["errors"]


def test_disconnect_artifact_verifier_rejects_boolean_client_id(tmp_path: Path) -> None:
    fixture = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text())
    fixture["client_id"] = True
    artifact = tmp_path / "boolean_client_id_disconnect_drill_go.json"
    artifact.write_text(json.dumps(fixture), encoding="utf-8")

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-NoWriteFreezeReport",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "client_id must be a positive integer" in payload["errors"]


def test_disconnect_artifact_verifier_rejects_boolean_financial_counter(tmp_path: Path) -> None:
    fixture = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text())
    fixture["financial_calls"]["place_order"] = False
    artifact = tmp_path / "boolean_financial_counter_disconnect_drill_go.json"
    artifact.write_text(json.dumps(fixture), encoding="utf-8")

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-NoWriteFreezeReport",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "financial_calls.place_order must be 0" in payload["errors"]


def test_disconnect_artifact_verifier_refuses_freeze_from_fixture_path() -> None:
    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json"),
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "freeze-writing verification requires an artifact under output\\ibkr" in payload["errors"]


def test_disconnect_artifact_verifier_refuses_freeze_from_wrong_artifact_name(tmp_path: Path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_dir = tmp_path / "output" / "ibkr"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "disconnect_drill_go.json"
    artifact.write_text(
        (PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-ProjectRoot",
        str(tmp_path),
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "freeze-writing verification requires artifact name phase1-disconnect-drill-YYYYMMDD-HHMMSS.json" in payload["errors"]


def test_disconnect_artifact_verifier_refuses_freeze_from_unparseable_artifact_timestamp(tmp_path: Path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_dir = tmp_path / "output" / "ibkr"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "phase1-disconnect-drill-99999999-999999.json"
    artifact.write_text(
        (PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-ProjectRoot",
        str(tmp_path),
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "freeze-writing verification requires parseable artifact timestamp YYYYMMDD-HHMMSS" in payload["errors"]


def test_disconnect_artifact_verifier_rejects_reconnect_before_disconnect(tmp_path: Path) -> None:
    fixture = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text())
    fixture["observed_statuses"] = [
        {"phase": "start", "status": "HEALTHY"},
        {"phase": "reconnect", "status": "HEALTHY"},
        {"phase": "heartbeat", "status": "DISCONNECTED"},
    ]
    artifact = tmp_path / "wrong_order_disconnect_drill_go.json"
    artifact.write_text(json.dumps(fixture), encoding="utf-8")

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-NoWriteFreezeReport",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "NO_GO"
    assert "observed_statuses must include a reconnect phase with HEALTHY or DEGRADED after disconnect" in payload["errors"]


def test_disconnect_artifact_verifier_written_report_opens_phase_gate(tmp_path: Path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_dir = tmp_path / "output" / "ibkr"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "phase1-disconnect-drill-20260720-000000.json"
    artifact.write_text(
        (PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-ProjectRoot",
        str(tmp_path),
    )
    status = phase1_freeze_status(tmp_path)

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "GO"
    assert payload["freeze_report"] == str(tmp_path / "PHASE1_FREEZE_REPORT.md")
    assert status.status == "PHASE1_FROZEN"
    assert status.frozen is True


def test_phase_gate_rejects_freeze_report_missing_immutable_config_hash(tmp_path: Path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_dir = tmp_path / "output" / "ibkr"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "phase1-disconnect-drill-20260720-000000.json"
    artifact.write_text(
        (PROJECT_ROOT / "tests" / "fixtures" / "disconnect_drill_go.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = run_powershell(
        str(PROJECT_ROOT / "scripts" / "verify_phase1_disconnect_drill_artifact.ps1"),
        "-ArtifactPath",
        str(artifact),
        "-ProjectRoot",
        str(tmp_path),
    )
    report_path = tmp_path / "PHASE1_FREEZE_REPORT.md"
    report_text = report_path.read_text(encoding="utf-8")
    report_path.write_text(
        "\n".join(
            line
            for line in report_text.splitlines()
            if "src/stocks/application/config.py" not in line
        ),
        encoding="utf-8",
    )

    status = phase1_freeze_status(tmp_path)

    assert result.returncode == 0
    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "frozen hash for src/stocks/application/config.py is missing from PHASE1_FREEZE_REPORT.md"


def test_disconnect_drill_helper_runs_post_freeze_static_audit() -> None:
    helper_text = (PROJECT_ROOT / "scripts" / "run_phase1_disconnect_drill.ps1").read_text(encoding="utf-8")

    assert "scripts\\run_phase1_static_audit.ps1" in helper_text
    assert "POST-FREEZE STATIC AUDIT GO" in helper_text


def test_disconnect_drill_helper_can_require_operator_ready_confirmation() -> None:
    helper_text = (PROJECT_ROOT / "scripts" / "run_phase1_disconnect_drill.ps1").read_text(encoding="utf-8")

    assert "[switch]$RequireOperatorReady" in helper_text
    assert "Operator confirmation required before the countdown starts." in helper_text
    assert 'if ($ready -ne "READY")' in helper_text


def test_static_audit_validates_public_env_examples() -> None:
    audit_text = (PROJECT_ROOT / "scripts" / "run_phase1_static_audit.ps1").read_text(encoding="utf-8")

    assert '"env_example_config"' in audit_text
    assert ".env.ibkr.example" in audit_text
    assert "env.ibkr.example" in audit_text


def test_static_audit_account_scan_excludes_generated_artifacts() -> None:
    audit_text = (PROJECT_ROOT / "scripts" / "run_phase1_static_audit.ps1").read_text(encoding="utf-8")
    account_scan_line = next(
        line for line in audit_text.splitlines() if line.startswith("$accountScanOutput")
    )

    assert r".\output\ibkr" not in account_scan_line


def _write_required_phase1_files(project_root: Path) -> None:
    for file_name in (
        "ibkr_tws_probe.py",
        "requirements.lock.txt",
        "main.py",
        "src/stocks/application/config.py",
        "src/stocks/application/context.py",
        "src/stocks/application/phase_gates.py",
        "src/stocks/application/lifecycle.py",
        "src/stocks/ibkr/connection.py",
        "src/stocks/ibkr/client.py",
        "src/stocks/ibkr/callbacks.py",
        "src/stocks/ibkr/errors.py",
        "src/stocks/ibkr/health.py",
    ):
        path = project_root / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{file_name}\n", encoding="utf-8")
