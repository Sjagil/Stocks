from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from stocks import readiness


def test_readiness_reports_frozen_live_writer(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        readiness,
        "recover_survivors",
        lambda _root: {
            "files_scanned": 0,
            "rows_scanned": 0,
            "survivor_count": 1,
            "classification_counts": {"FROZEN_SHADOW": 1},
        },
    )
    monkeypatch.setattr(
        readiness,
        "signal_status",
        lambda _root: {
            "manual_actionable_strategies": 0,
            "execution_authority": "NONE",
        },
    )
    monkeypatch.setattr(
        readiness,
        "runtime_command",
        lambda _root, _command: {"status": "GO"},
    )
    monkeypatch.setattr(
        readiness,
        "live_preflight",
        lambda _root, **_kwargs: {
            "status": "NO_GO",
            "blockers": ["EXACT_OPERATOR_APPROVAL_REQUIRED"],
            "checks": {"live_writer_frozen": True},
        },
    )
    monkeypatch.setattr(
        readiness,
        "live_kill_switch",
        lambda _root, **_kwargs: {"status": "GO"},
    )
    monkeypatch.setattr(
        readiness,
        "telegram_command",
        lambda _root, _command: {"status": "ENABLED"},
    )

    report = readiness.system_readiness(tmp_path)

    assert (
        report["architecture"]["live_adapter"]
        == "LEVEL1_LIVE_CANARY_WRITER_OFFLINE_FROZEN"
    )
    assert report["execution_authority"] == "NONE"
    audit = tmp_path / "output" / "reports" / "system_audit"
    assert (audit / "repository_inventory.json").exists()
    assert (audit / "live_blocker_report.json").exists()
    assert (audit / "architecture_gap_report.md").exists()


def test_readiness_artifact_writer_is_concurrency_safe(tmp_path) -> None:
    path = tmp_path / "readiness.json"

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda value: readiness._atomic_text(path, str(value)),
                range(32),
            )
        )

    assert int(path.read_text(encoding="utf-8")) in range(32)
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []
