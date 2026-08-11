from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PHASE1_FREEZE_MARKER = "IBKR_PHASE_1_READ_ONLY_CONNECTION_SERVICE_GO"
PHASE1_FROZEN_MARKER = "PHASE1_CONNECTION_SERVICE_FROZEN_GO"
FROZEN_PHASE0_HASH_FILES = (
    "ibkr_tws_probe.py",
    "requirements.lock.txt",
)
IMMUTABLE_PHASE1_SERVICE_HASH_FILES = (
    "src/stocks/application/config.py",
    "src/stocks/ibkr/connection.py",
    "src/stocks/ibkr/client.py",
    "src/stocks/ibkr/callbacks.py",
    "src/stocks/ibkr/errors.py",
    "src/stocks/ibkr/health.py",
)
MUTABLE_APPLICATION_ENTRYPOINT_HASH_FILES = ("main.py",)


@dataclass(frozen=True)
class PhaseGateStatus:
    name: str
    status: str
    frozen: bool
    report_path: Path
    reason: str | None = None

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "name": self.name,
            "status": self.status,
            "frozen": self.frozen,
            "report_path": str(self.report_path),
            "reason": self.reason,
        }


def phase1_freeze_status(project_root: Path) -> PhaseGateStatus:
    report_path = project_root / "PHASE1_FREEZE_REPORT.md"
    if not report_path.exists():
        return PhaseGateStatus(
            name="phase1",
            status="PHASE1_NOT_FROZEN",
            frozen=False,
            report_path=report_path,
            reason="PHASE1_FREEZE_REPORT.md is missing",
        )

    report_text = report_path.read_text(encoding="utf-8", errors="replace")
    missing_markers = [
        marker
        for marker in (PHASE1_FREEZE_MARKER, PHASE1_FROZEN_MARKER)
        if marker not in report_text
    ]
    if missing_markers:
        return PhaseGateStatus(
            name="phase1",
            status="PHASE1_NOT_FROZEN",
            frozen=False,
            report_path=report_path,
            reason=f"freeze marker is missing from PHASE1_FREEZE_REPORT.md: {', '.join(missing_markers)}",
        )

    evidence_error = _validate_phase1_freeze_report_evidence(project_root, report_text)
    if evidence_error:
        return PhaseGateStatus(
            name="phase1",
            status="PHASE1_NOT_FROZEN",
            frozen=False,
            report_path=report_path,
            reason=evidence_error,
        )

    return PhaseGateStatus(
        name="phase1",
        status="PHASE1_FROZEN",
        frozen=True,
        report_path=report_path,
    )


def _validate_phase1_freeze_report_evidence(project_root: Path, report_text: str) -> str | None:
    artifact_text = _fenced_value_after_heading(report_text, "Verified artifact")
    if artifact_text is None:
        return "verified artifact path is missing from PHASE1_FREEZE_REPORT.md"

    artifact_path = Path(artifact_text)
    if not artifact_path.is_absolute():
        artifact_path = project_root / artifact_path
    try:
        artifact_path = artifact_path.resolve(strict=True)
    except OSError:
        return "verified artifact does not exist"

    try:
        output_root = (project_root / "output" / "ibkr").resolve(strict=True)
    except OSError:
        return "output/ibkr evidence directory does not exist"
    if output_root not in (artifact_path, *artifact_path.parents):
        return "verified artifact must be under output/ibkr"
    if not _is_phase1_disconnect_artifact_name(artifact_path.name):
        return "verified artifact name must match phase1-disconnect-drill-YYYYMMDD-HHMMSS.json"

    expected_artifact_hash = _fenced_value_after_heading(report_text, "Artifact SHA256")
    if expected_artifact_hash is None:
        return "artifact SHA256 is missing from PHASE1_FREEZE_REPORT.md"
    if not re.fullmatch(r"[A-F0-9]{64}", expected_artifact_hash):
        return "artifact SHA256 must be a 64-character uppercase hex digest"

    actual_artifact_hash = _sha256_hex(artifact_path)
    if actual_artifact_hash != expected_artifact_hash:
        return "artifact SHA256 does not match PHASE1_FREEZE_REPORT.md"

    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "verified artifact is not readable JSON"

    artifact_error = _phase1_drill_artifact_error(artifact)
    if artifact_error:
        return artifact_error

    return _frozen_hash_error(project_root, report_text)


def _phase1_drill_artifact_error(artifact: dict[str, object]) -> str | None:
    if artifact.get("schema") != "ibkr_forced_disconnect_drill_v1":
        return "verified artifact schema must be ibkr_forced_disconnect_drill_v1"
    if artifact.get("status") != "GO":
        return "verified artifact status must be GO"
    if artifact.get("host") != "127.0.0.1":
        return "verified artifact host must be 127.0.0.1 for the TWS paper Phase 1 freeze drill"
    port = artifact.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or port != 7497:
        return "verified artifact port must be 7497 for the TWS paper Phase 1 freeze drill"
    client_id = artifact.get("client_id")
    if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id <= 0:
        return "verified artifact client_id must be a positive integer"
    if artifact.get("disconnect_observed") is not True:
        return "verified artifact disconnect_observed must be true"
    if artifact.get("reconnect_successful") is not True:
        return "verified artifact reconnect_successful must be true"
    seconds_raw = artifact.get("seconds")
    poll_seconds_raw = artifact.get("poll_seconds")
    if (
        isinstance(seconds_raw, bool)
        or isinstance(poll_seconds_raw, bool)
        or not isinstance(seconds_raw, int | float)
        or not isinstance(poll_seconds_raw, int | float)
    ):
        return "verified artifact seconds and poll_seconds must be numeric"
    seconds = float(seconds_raw)
    poll_seconds = float(poll_seconds_raw)
    if seconds < 180.0:
        return "verified artifact seconds must be at least 180"
    if poll_seconds <= 0.0 or poll_seconds > seconds:
        return "verified artifact poll_seconds must be positive and no greater than seconds"
    failure_reason = artifact.get("failure_reason")
    if failure_reason is not None and str(failure_reason).strip():
        return "verified artifact failure_reason must be null or blank"

    financial_calls = artifact.get("financial_calls")
    if not isinstance(financial_calls, dict):
        return "verified artifact financial_calls must be an object"
    for field_name in ("place_order", "cancel_order", "global_cancel"):
        value = financial_calls.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            return f"verified artifact financial_calls.{field_name} must be 0"

    observed_statuses = artifact.get("observed_statuses")
    if not isinstance(observed_statuses, list):
        return "verified artifact observed_statuses must be an array"
    healthy_start_index = None
    disconnect_index = None
    reconnect_index = None
    for index, item in enumerate(observed_statuses):
        if not isinstance(item, dict):
            continue
        phase = item.get("phase")
        status = item.get("status")
        if healthy_start_index is None and phase == "start" and status == "HEALTHY":
            healthy_start_index = index
        if (
            disconnect_index is None
            and healthy_start_index is not None
            and index > healthy_start_index
            and status in {"DISCONNECTED", "STALE"}
        ):
            disconnect_index = index
        if (
            reconnect_index is None
            and disconnect_index is not None
            and index > disconnect_index
            and phase == "reconnect"
            and status in {"HEALTHY", "DEGRADED"}
        ):
            reconnect_index = index
    if healthy_start_index is None:
        return "verified artifact observed_statuses must include start HEALTHY"
    if disconnect_index is None:
        return "verified artifact observed_statuses must include DISCONNECTED or STALE after start"
    if reconnect_index is None:
        return "verified artifact observed_statuses must include reconnect HEALTHY or DEGRADED after disconnect"
    return None


def _fenced_value_after_heading(text: str, heading: str) -> str | None:
    pattern = rf"(?ms)^{re.escape(heading)}:\s*```text\s*(.*?)\s*```"
    match = re.search(pattern, text)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _frozen_hash_error(project_root: Path, report_text: str) -> str | None:
    phase0_hashes = _parse_hash_block(report_text, "Frozen Phase 0 hashes")
    immutable_hashes = _parse_hash_block(report_text, "Immutable Phase 1 service hashes")
    mutable_entrypoint_hashes = _parse_hash_block(report_text, "Mutable application entrypoint hash")
    if phase0_hashes is None:
        return "Frozen Phase 0 hashes are missing from PHASE1_FREEZE_REPORT.md"
    if immutable_hashes is None:
        immutable_hashes = _parse_hash_block(report_text, "Frozen Phase 1 hashes")
    if immutable_hashes is None:
        return "Immutable Phase 1 service hashes are missing from PHASE1_FREEZE_REPORT.md"

    for file_name in FROZEN_PHASE0_HASH_FILES:
        error = _frozen_file_hash_error(project_root, phase0_hashes, file_name)
        if error:
            return error
    for file_name in IMMUTABLE_PHASE1_SERVICE_HASH_FILES:
        error = _frozen_file_hash_error(project_root, immutable_hashes, file_name)
        if error:
            return error
    if mutable_entrypoint_hashes is not None:
        for file_name in MUTABLE_APPLICATION_ENTRYPOINT_HASH_FILES:
            error = _reported_file_hash_format_error(mutable_entrypoint_hashes, file_name)
            if error:
                return error
    return None


def _parse_hash_block(text: str, heading: str) -> dict[str, str] | None:
    block = _fenced_value_after_heading(text, heading)
    if block is None:
        return None
    hashes: dict[str, str] = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        file_name, digest = parts
        hashes[file_name] = digest
    return hashes


def _frozen_file_hash_error(project_root: Path, hashes: dict[str, str], file_name: str) -> str | None:
    expected_hash = hashes.get(file_name)
    if expected_hash is None:
        return f"frozen hash for {file_name} is missing from PHASE1_FREEZE_REPORT.md"
    if not re.fullmatch(r"[A-F0-9]{64}", expected_hash):
        return f"frozen hash for {file_name} must be a 64-character uppercase hex digest"
    file_path = project_root / file_name
    if not file_path.exists():
        return f"frozen file is missing: {file_name}"
    if _sha256_hex(file_path) != expected_hash:
        return f"frozen hash mismatch for {file_name}"
    return None


def _reported_file_hash_format_error(hashes: dict[str, str], file_name: str) -> str | None:
    expected_hash = hashes.get(file_name)
    if expected_hash is None:
        return f"mutable application entrypoint hash for {file_name} is missing from PHASE1_FREEZE_REPORT.md"
    if not re.fullmatch(r"[A-F0-9]{64}", expected_hash):
        return f"mutable application entrypoint hash for {file_name} must be a 64-character uppercase hex digest"
    return None


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _is_phase1_disconnect_artifact_name(file_name: str) -> bool:
    match = re.fullmatch(r"phase1-disconnect-drill-(\d{8})-(\d{6})\.json", file_name)
    if match is None:
        return False
    try:
        datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return False
    return True
