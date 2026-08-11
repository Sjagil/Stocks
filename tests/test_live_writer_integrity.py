from __future__ import annotations

import json
from pathlib import Path

from stocks.live.integrity import (
    build_manifest,
    freeze_manifest,
    inspect_manifest,
    normalized_file_hash,
    verify_manifest,
)
from stocks.live import service as live_service


SOURCES = (
    "config/writer.json",
    "src/stocks/live/adapter.py",
    "src/stocks/live/integrity.py",
)


def _tree(root: Path) -> None:
    for relative, content in {
        "config/writer.json": '{"authority":"NONE"}\n',
        "src/stocks/live/adapter.py": "def writer():\n    return 'blocked'\n",
        "src/stocks/live/integrity.py": "VERSION = 2\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_line_ending_normalization_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "writer.py"
    path.write_bytes(b"first\r\nsecond\r\n")
    windows_hash = normalized_file_hash(path)
    path.write_bytes(b"first\nsecond\n")
    assert normalized_file_hash(path) == windows_hash


def test_manifest_detects_changed_and_missing_critical_file(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    frozen = build_manifest(
        tmp_path,
        SOURCES,
        operator="tester",
        reason="fixture",
    )
    assert frozen["status"] == "GO"

    (tmp_path / "src/stocks/live/adapter.py").write_text(
        "def writer():\n    return 'changed'\n", encoding="utf-8"
    )
    (tmp_path / "config/writer.json").unlink()
    verified = verify_manifest(tmp_path, SOURCES, frozen)
    assert verified["status"] == "NO_GO"
    assert "src/stocks/live/adapter.py" in verified["changed_files"]
    assert "config/writer.json" in verified["missing_files"]
    assert verified["execution_authority"] == "NONE"


def test_manifest_detects_extra_configured_critical_file(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    frozen = build_manifest(
        tmp_path,
        SOURCES,
        operator="tester",
        reason="fixture",
    )
    extra = "src/stocks/live/new_writer.py"
    (tmp_path / extra).write_text("VERSION = 1\n", encoding="utf-8")

    verified = verify_manifest(tmp_path, (*SOURCES, extra), frozen)

    assert verified["status"] == "NO_GO"
    assert verified["extra_critical_files"] == [extra]
    assert "UNAUTHORIZED_CRITICAL_FILE:" + extra in verified["blockers"]


def test_inspect_is_fail_closed_when_current_hash_differs(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    frozen = build_manifest(
        tmp_path,
        SOURCES,
        operator="tester",
        reason="fixture",
    )
    freeze_path = tmp_path / "output/ibkr/live/freeze-status.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    changed = "src/stocks/live/adapter.py"
    (tmp_path / changed).write_text("VERSION = 3\n", encoding="utf-8")

    inspected = inspect_manifest(tmp_path, SOURCES)

    assert inspected["status"] == "NO_GO"
    assert inspected["inspection_status"] == "MISMATCH"
    assert inspected["writer_hash_integrity"] is False
    assert inspected["changed_files"] == [changed]
    assert "CRITICAL_FILE_CHANGED:" + changed in inspected["blockers"]


def test_inspect_reports_match_only_for_verified_freeze(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    frozen = build_manifest(
        tmp_path,
        SOURCES,
        operator="tester",
        reason="fixture",
    )
    freeze_path = tmp_path / "output/ibkr/live/freeze-status.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")

    inspected = inspect_manifest(tmp_path, SOURCES)

    assert inspected["status"] == "GO"
    assert inspected["inspection_status"] == "MATCH"
    assert inspected["writer_hash_integrity"] is True
    assert inspected["blockers"] == []


def test_diff_preserves_invalid_freeze_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    invalid_verification = {
        "schema": "live_writer_integrity_verification_v2",
        "status": "NO_GO",
        "writer_hash_integrity": False,
        "missing_files": [],
        "extra_critical_files": [],
        "changed_files": [],
        "unauthorized_live_modules": [],
        "blockers": ["LEGACY_OR_INVALID_FREEZE_SCHEMA"],
        "execution_authority": "NONE",
        "live_trading_allowed": False,
    }
    monkeypatch.setattr(
        live_service,
        "verify_manifest",
        lambda *_args, **_kwargs: invalid_verification,
    )

    result = live_service.live_writer_integrity_command(tmp_path, "diff")

    assert result["status"] == "NO_GO"
    assert result["path_differences_detected"] is False
    assert result["difference_free"] is False
    assert result["blockers"] == ["LEGACY_OR_INVALID_FREEZE_SCHEMA"]


def test_public_live_artifact_hash_binds_masking_metadata(
    tmp_path: Path,
) -> None:
    live_service._write_live_artifact(
        tmp_path,
        "reconciliation.json",
        {
            "schema": "fixture_v1",
            "status": "GO",
            "execution_authority": "NONE",
        },
    )
    payload = json.loads(
        (
            tmp_path
            / "output"
            / "ibkr"
            / "live"
            / "reconciliation.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["account_masked"] is True
    assert payload["credentials_logged"] is False
    assert live_service._artifact_content_hash_valid(payload) is True


def test_unexpected_live_module_is_fail_closed(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "src/stocks/live/rogue.py").write_text(
        "def placeOrder(): pass\n", encoding="utf-8"
    )
    manifest = build_manifest(
        tmp_path,
        SOURCES,
        operator="tester",
        reason="fixture",
    )
    assert manifest["status"] == "NO_GO"
    assert manifest["unauthorized_live_modules"] == [
        "src/stocks/live/rogue.py"
    ]


def test_refreeze_requires_explicit_confirmation_and_keeps_history(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    first = freeze_manifest(
        tmp_path,
        SOURCES,
        operator="operator-a",
        reason="initial audit",
        re_freeze=False,
        confirmed=False,
    )
    assert first["status"] == "GO"
    (tmp_path / "src/stocks/live/adapter.py").write_text(
        "def writer():\n    return 'reviewed change'\n", encoding="utf-8"
    )
    blocked = freeze_manifest(
        tmp_path,
        SOURCES,
        operator="operator-b",
        reason="reviewed change",
        re_freeze=True,
        confirmed=False,
    )
    assert blocked["status"] == "NO_GO"

    accepted = freeze_manifest(
        tmp_path,
        SOURCES,
        operator="operator-b",
        reason="reviewed change",
        re_freeze=True,
        confirmed=True,
    )
    assert accepted["status"] == "GO"
    history = [
        json.loads(line)
        for line in (
            tmp_path
            / "output/ibkr/live/writer-integrity-history.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in history] == [
        "INITIAL_FREEZE",
        "REFREEZE",
    ]
    assert history[-1]["old_manifest_hash"] == first["manifest_hash"]
