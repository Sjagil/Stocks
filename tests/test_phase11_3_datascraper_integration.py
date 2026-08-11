from __future__ import annotations

import json
from pathlib import Path

from stocks.research.phase11_3.datascraper_adapter import DatascraperAdapter
from stocks.research.phase11_3.export_manifest import stable_hash, validate_export_manifest
from stocks.research.phase11_3.import_audit import Phase113Store
from stocks.research.phase11_3.shariah_history import _price_before, _safe_ratio


def _write_export(root: Path, *, batch: str = "B1", value: int = 1, pit: str = "PIT_ELIGIBLE_NEXT_SESSION", license_status: str = "PUBLIC_DATA") -> Path:
    directory = root / "output" / "stocks_research_exports" / "phase11_3" / "prices" / f"batch-{batch}"
    directory.mkdir(parents=True)
    data = directory / "records.jsonl"
    data.write_text(json.dumps({"symbol": "A.US", "timestamp": "2020-01-01", "close": value}) + "\n", encoding="utf-8")
    import hashlib
    content_hash = hashlib.sha256(data.read_bytes()).hexdigest().upper()
    manifest = {"schema_version": "stocks_phase11_3_export_v1", "batch_id": batch, "run_id": "R", "provider": "TEST", "dataset": "prices", "requested_interval": {"start": "2000-01-01", "end": "2020-01-01"}, "returned_interval": {"start": "2020-01-01", "end": "2020-01-01"}, "symbol_count": 1, "record_count": 1, "generated_at": "2026-01-01T00:00:00Z", "PIT_classification": pit, "license_classification": license_status, "data_file": "records.jsonl", "content_hash": content_hash, "source_hash": "S", "contains_credentials": False, "execution_authority": "NONE", "order_authority": "NONE"}
    manifest["manifest_hash"] = stable_hash(manifest)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_acceptance_and_tamper_detection(tmp_path: Path) -> None:
    path = _write_export(tmp_path)
    assert validate_export_manifest(path).status == "DATASCRAPER_EXPORT_ACCEPTED"
    (path.parent / "records.jsonl").write_text("tampered", encoding="utf-8")
    assert validate_export_manifest(path).status == "DATASCRAPER_EXPORT_HASH_MISMATCH"


def test_manifest_pit_and_license_fail_closed(tmp_path: Path) -> None:
    assert validate_export_manifest(_write_export(tmp_path / "pit", pit="UNKNOWN")).status == "DATASCRAPER_EXPORT_PIT_BLOCKED"
    assert validate_export_manifest(_write_export(tmp_path / "license", license_status="LICENSE_BLOCKED")).status == "DATASCRAPER_EXPORT_LICENSE_BLOCKED"


def test_import_is_append_only_idempotent_and_versions_changed_batch(tmp_path: Path) -> None:
    export_root = tmp_path / "data-plane"
    _write_export(export_root, batch="B1", value=1)
    adapter = DatascraperAdapter(export_root)
    store = Phase113Store(tmp_path / "stocks" / "phase11_3.sqlite3")
    first = store.import_from(adapter)
    replay = store.import_from(adapter)
    _write_export(export_root, batch="B2", value=2)
    changed = store.import_from(adapter)
    assert first["accepted_record_count"] == 1
    assert replay["batches"][0]["status"] == "DATASCRAPER_EXPORT_DUPLICATE"
    assert changed["accepted_record_count"] == 1
    assert store.counts()["prices"] == 2


def test_phase7_fixture_is_never_a_broker_mirror(tmp_path: Path) -> None:
    store = Phase113Store(tmp_path / "research.sqlite3")
    store.initialize()
    store.append_event("PHASE7_COMPARISON", "SIM-1", {"identifier": "SIM-1", "classification": "SYNTHETIC_FIXTURE_ONLY", "status": "PHASE7_FIXTURE_NOT_BROKER_MIRROR"})
    text = store.path.read_bytes()
    assert b"SYNTHETIC_FIXTURE_ONLY" in text
    assert b"RECONCILED" not in text


def test_no_execution_or_broker_writer_symbols() -> None:
    root = Path(__file__).parents[1] / "src" / "stocks" / "research" / "phase11_3"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    forbidden = ("placeOrder", "cancelOrder", "reqGlobalCancel", "reqIds", "reqAutoOpenOrders", "exerciseOptions")
    assert all(token not in source for token in forbidden)
    assert "execution_authority\": \"NONE" in source


def test_public_contract_contains_no_financial_or_secret_fields() -> None:
    root = Path(__file__).parents[1] / "src" / "stocks" / "research" / "phase11_3"
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in root.glob("*.py") if path.name != "audit.py")
    assert "netliquidation" not in source
    assert "availablefunds" not in source
    assert "buyingpower" not in source
    assert "api_token" not in source


def test_listing_identity_override_is_source_grounded() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads((root / "config" / "research" / "phase11_3_listing_identity_overrides.json").read_text(encoding="utf-8"))
    for symbol in ("APLD", "CPHI", "NBIS", "ONDS"):
        override = payload["overrides"][symbol]
        assert override["coverage_gate_eligible"] is True
        assert override["effective_start"]
        assert all(source.startswith("https://") for source in override["sources"])


def test_delisted_controls_do_not_silently_pass_as_complete() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads((root / "config" / "research" / "phase11_3_listing_identity_overrides.json").read_text(encoding="utf-8"))
    for symbol in ("AAB", "AAAID", "AAALF", "AAALY", "AAARF", "AAC-U"):
        override = payload["overrides"][symbol]
        assert override["coverage_gate_eligible"] is False
        assert override["classification"].startswith("DELISTED_HISTORY_")


def test_shariah_price_join_is_strictly_point_in_time() -> None:
    from decimal import Decimal

    values = [("2020-01-02", Decimal("10")), ("2020-01-06", Decimal("12"))]
    assert _price_before(values, "2020-01-05") == ("2020-01-02", Decimal("10"))
    assert _price_before(values, "2020-01-06") == ("2020-01-02", Decimal("10"))
    assert _price_before(values, "2019-12-31") == (None, None)


def test_shariah_ratio_fails_closed_without_denominator() -> None:
    from decimal import Decimal

    assert _safe_ratio(Decimal("10"), Decimal("100")) == Decimal("0.1")
    assert _safe_ratio(Decimal("10"), None) is None
    assert _safe_ratio(Decimal("10"), Decimal("0")) is None


def test_sec_acceptance_override_is_exact_and_source_grounded() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads((root / "config" / "research" / "phase11_3_sec_acceptance_overrides.json").read_text(encoding="utf-8"))
    override = next(iter(payload["overrides"].values()))
    assert override["accepted_at"] == "2022-02-24T16:13:45-05:00"
    assert override["source"].startswith("https://www.sec.gov/Archives/edgar/")
