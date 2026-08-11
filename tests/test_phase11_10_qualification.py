from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import stocks.research.phase11_10 as phase11_10


def _research_artifacts(root: Path) -> Path:
    output = root / "output/research/phase11_10"
    output.mkdir(parents=True)
    pd.DataFrame([{"architecture": "A", "value": 1}]).to_csv(
        output / "architecture-summary.csv", index=False
    )
    pd.DataFrame([{"architecture": "A", "selected_profile": "balanced"}]).to_csv(
        output / "parameter-selections.csv", index=False
    )
    pd.DataFrame([{"architecture": "A", "result": 1}]).to_parquet(
        output / "nested-results.parquet", index=False
    )
    pd.DataFrame([{"architecture": "A", "start": "2020-01-01"}]).to_csv(
        output / "coverage.csv", index=False
    )
    (output / "shortlist.json").write_text(
        json.dumps({"candidates": [{"strategy_id": "MTF-ONE"}]}),
        encoding="utf-8",
    )
    status = {
        "status": "GO",
        "research_source_hash": "CODE",
        "historical_data_cutoff": "2026-07-24T23:59:59+00:00",
        "historical_data_hash": "DATA",
        "coverage": [{"architecture": "A", "end": "2026-07-24"}],
        "shortlist": {"candidates": [{"strategy_id": "MTF-ONE"}]},
    }
    (output / "status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    (output / "run-checkpoint.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8"
    )
    return output


def test_qualification_freeze_is_immutable_and_runtime_bars_are_excluded(
    tmp_path: Path, monkeypatch
) -> None:
    output = _research_artifacts(tmp_path)
    monkeypatch.setattr(
        phase11_10, "_research_source_hash", lambda _root: "CODE"
    )
    monkeypatch.setattr(
        phase11_10,
        "_research_source_components",
        lambda _root: {"phase11_10.py": "CODE"},
    )

    freeze = phase11_10.phase11_10_qualification_freeze(tmp_path)
    pointer = json.loads(
        (output / "qualification/current.json").read_text(encoding="utf-8")
    )
    immutable = output / pointer["immutable_manifest_path"]
    frozen_bytes = immutable.read_bytes()
    runtime = tmp_path / "data/research/multitimeframe/private/new.parquet"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"new runtime bar")
    audit = phase11_10.phase11_10_qualification_audit(tmp_path)

    assert freeze["status"] == "GO"
    assert audit["status"] == "GO"
    assert audit["runtime_bars_part_of_research_hash"] is False
    assert immutable.read_bytes() == frozen_bytes


def test_qualification_freeze_requires_replay_after_source_change(
    tmp_path: Path, monkeypatch
) -> None:
    _research_artifacts(tmp_path)
    monkeypatch.setattr(
        phase11_10, "_research_source_hash", lambda _root: "NEW-CODE"
    )

    report = phase11_10.phase11_10_qualification_freeze(tmp_path)

    assert report["status"] == "BLOCKED"
    assert "RESEARCH_SOURCE_REPLAY_REQUIRED" in report["blockers"]


def test_research_result_hash_excludes_operational_status_metadata(
    tmp_path: Path,
) -> None:
    output = _research_artifacts(tmp_path)
    original = phase11_10._research_result_hash(output)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["research_source_hash"] = "NEW-SOURCE"
    status["parallel_worker_limit"] = 12
    status_path.write_text(json.dumps(status), encoding="utf-8")

    assert phase11_10._research_result_hash(output) == original


def test_research_result_hash_detects_financial_artifact_change(
    tmp_path: Path,
) -> None:
    output = _research_artifacts(tmp_path)
    original = phase11_10._research_result_hash(output)
    pd.DataFrame([{"architecture": "A", "value": 2}]).to_csv(
        output / "architecture-summary.csv", index=False
    )

    assert phase11_10._research_result_hash(output) != original


def test_historical_cutoff_excludes_new_bars_without_mutating_attributes() -> None:
    frame = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.to_datetime(["2026-07-24", "2026-07-25"], utc=True),
    )
    frame.attrs["provider"] = "TEST"

    truncated = phase11_10._truncate_frames(
        {"1d": {"AAPL": frame}},
        cutoff=phase11_10._historical_cutoff("2026-07-24"),
    )

    result = truncated["1d"]["AAPL"]
    assert len(result) == 1
    assert result.attrs["provider"] == "TEST"
    assert len(frame) == 2


def test_worker_and_fingerprint_use_only_cutoff_rows(monkeypatch) -> None:
    base = pd.DataFrame(
        {"close": [1.0]},
        index=pd.to_datetime(["2026-07-24"], utc=True),
    )
    extended = pd.DataFrame(
        {"close": [1.0, 999.0]},
        index=pd.to_datetime(["2026-07-24", "2026-08-04"], utc=True),
    )
    monkeypatch.setattr(
        phase11_10,
        "_load_frames",
        lambda _root: {"1d": {"AAPL": extended}},
    )

    phase11_10._initialize_mtf_worker(".", "2026-07-24")
    truncated_extended = phase11_10._truncate_frames(
        {"1d": {"AAPL": extended}},
        cutoff=phase11_10._historical_cutoff("2026-07-24"),
    )

    assert len(phase11_10._MTF_WORKER_FRAMES["1d"]["AAPL"]) == 1
    assert phase11_10._multitimeframe_frames_fingerprint(
        {"1d": {"AAPL": base}}
    ) == phase11_10._multitimeframe_frames_fingerprint(truncated_extended)


def test_cutoff_breach_is_detected() -> None:
    breaches = phase11_10._cutoff_coverage_breaches(
        [{"architecture": "A", "end": "2026-07-25"}],
        cutoff=phase11_10._historical_cutoff("2026-07-24"),
    )

    assert breaches[0]["reason"] == "HISTORICAL_CUTOFF_BREACH"
