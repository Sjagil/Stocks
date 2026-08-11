from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PHASE4_SOURCE_FILES = (
    "src/stocks/data/ibkr_historical.py",
    "src/stocks/data/bars.py",
    "src/stocks/market/sessions.py",
)

PHASE4_CONTRACT_CACHE_FILES = (
    "output/ibkr/contracts/stocks.parquet",
    "output/ibkr/contracts/contract_manifest.json",
    "output/ibkr/contracts/contract_requests.jsonl",
    "output/ibkr/contracts/contract_errors.jsonl",
)

PHASE4_SESSION_CACHE_FILES = (
    "data/sessions/sessions.parquet",
    "data/sessions/session_manifest.json",
    "data/sessions/session_conflicts.jsonl",
    "data/sessions/session_errors.jsonl",
)

PHASE4_BAR_SCHEMA_FILES = (
    "data/bars/bar_manifest.json",
    "output/ibkr/bars/cache-validation.json",
)

PHASE4_ERROR_CLASSIFICATION_FILES = (
    "output/ibkr/bars/error-classification.json",
)

PHASE4_BAR_DATA_GLOB = "data/bars/security_type=STK/con_id=*/interval=1d/data_type=TRADES/bars.parquet"


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    status_path = project_root / "output" / "ibkr" / "phase4-historical-bars-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "IBKR_PHASE4_HISTORICAL_BARS_READ_ONLY_GO":
        raise SystemExit("Phase 4 status artifact is not GO")

    source_hashes = _hash_group(project_root, PHASE4_SOURCE_FILES)
    contract_cache_hashes = _hash_group(project_root, PHASE4_CONTRACT_CACHE_FILES)
    session_cache_hashes = _hash_group(project_root, PHASE4_SESSION_CACHE_FILES)
    bar_schema_hashes = _hash_group(project_root, PHASE4_BAR_SCHEMA_FILES)
    error_classification_hashes = _hash_group(project_root, PHASE4_ERROR_CLASSIFICATION_FILES)
    bar_data_hashes = _hash_paths(project_root, sorted(project_root.glob(PHASE4_BAR_DATA_GLOB)))

    freeze = {
        "schema": "ibkr_phase4_historical_bars_freeze_v1",
        "contract_id": "IBKR_PHASE4_HISTORICAL_BARS_V1",
        "status": "IBKR_PHASE4_HISTORICAL_BARS_V1_FROZEN_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "phase4_status_artifact": str(status_path),
        "phase4_status_hash": _sha256(status_path),
        "source_hashes": source_hashes,
        "contract_cache_hashes": contract_cache_hashes,
        "session_cache_hashes": session_cache_hashes,
        "bar_schema_hashes": bar_schema_hashes,
        "error_classification_hashes": error_classification_hashes,
        "bar_data_hashes": bar_data_hashes,
        "go_evidence": {
            "instrument_count": status["cache_validation"]["instrument_count"],
            "file_count": status["cache_validation"]["file_count"],
            "row_count": status["cache_validation"]["row_count"],
            "duplicate_rows": status["cache_validation"]["duplicate_rows"],
            "invalid_ohlc_rows": status["cache_validation"]["invalid_ohlc_rows"],
            "timezone_errors": status["cache_validation"]["timezone_errors"],
            "contract_mismatches": status["cache_validation"]["contract_mismatches"],
            "financial_calls": status["financial_calls"],
            "provider_calls": status["provider_calls"],
            "tests": status["tests"],
        },
    }
    output_path = project_root / "output" / "ibkr" / "phase4-freeze-status.json"
    output_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_markdown(project_root / "PHASE4_FREEZE_REPORT.md", freeze)
    print(json.dumps(freeze, indent=2, ensure_ascii=True))
    return 0


def _hash_group(project_root: Path, relative_paths: tuple[str, ...]) -> dict[str, str | None]:
    return {path: _sha256(project_root / path) if (project_root / path).exists() else None for path in relative_paths}


def _hash_paths(project_root: Path, paths: list[Path]) -> dict[str, str]:
    return {path.relative_to(project_root).as_posix(): _sha256(path) for path in paths}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_markdown(path: Path, freeze: dict[str, Any]) -> None:
    lines = [
        "# Phase 4 Freeze Report",
        "",
        "Status:",
        "",
        "```text",
        str(freeze["status"]),
        "```",
        "",
        "GO evidence:",
        "",
        "```text",
        f"instrument_count       {freeze['go_evidence']['instrument_count']}",
        f"file_count             {freeze['go_evidence']['file_count']}",
        f"row_count              {freeze['go_evidence']['row_count']}",
        f"duplicate_rows         {freeze['go_evidence']['duplicate_rows']}",
        f"invalid_ohlc_rows      {freeze['go_evidence']['invalid_ohlc_rows']}",
        f"timezone_errors        {freeze['go_evidence']['timezone_errors']}",
        f"contract_mismatches    {freeze['go_evidence']['contract_mismatches']}",
        "financial_calls        0",
        "```",
        "",
        "Source hashes:",
        "",
        "```text",
        *_hash_lines(freeze["source_hashes"]),
        "```",
        "",
        "Contract cache hashes:",
        "",
        "```text",
        *_hash_lines(freeze["contract_cache_hashes"]),
        "```",
        "",
        "Session cache hashes:",
        "",
        "```text",
        *_hash_lines(freeze["session_cache_hashes"]),
        "```",
        "",
        "Bar schema hashes:",
        "",
        "```text",
        *_hash_lines(freeze["bar_schema_hashes"]),
        "```",
        "",
        "Error classification hashes:",
        "",
        "```text",
        *_hash_lines(freeze["error_classification_hashes"]),
        "```",
        "",
        "Bar data hashes:",
        "",
        "```text",
        *_hash_lines(freeze["bar_data_hashes"]),
        "```",
        "",
        "Phase 4 remains read-only. This report does not grant order authority.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _hash_lines(mapping: dict[str, str | None]) -> list[str]:
    return [f"{key:<78} {value}" for key, value in mapping.items()]


if __name__ == "__main__":
    raise SystemExit(main())
