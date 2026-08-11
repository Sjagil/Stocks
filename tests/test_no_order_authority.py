from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_no_ibkr_write_methods_in_phase1_source() -> None:
    allowed = {
        PROJECT_ROOT / "src" / "stocks" / "ibkr" / "paper_execution" / "submission.py": {"place" + "Order"},
        PROJECT_ROOT / "src" / "stocks" / "ibkr" / "paper_execution" / "cancellation.py": {"cancel" + "Order"},
        PROJECT_ROOT / "src" / "stocks" / "ibkr" / "paper_execution" / "order_ids.py": {"req" + "Ids"},
        PROJECT_ROOT / "src" / "stocks" / "live" / "submission.py": {"place" + "Order"},
    }
    forbidden = ("place" + "Order", "cancel" + "Order", "req" + "Global" + "Cancel", "req" + "Ids")
    scanned_files = [
        PROJECT_ROOT / "main.py",
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
    ]

    for path in scanned_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in allowed.get(path, set()):
                continue
            assert token not in text, f"{token} found in {path}"


def test_no_market_or_historical_data_request_methods_in_entrypoint_or_source() -> None:
    globally_forbidden = (
        "req" + "Mkt" + "Data",
        "req" + "Real" + "Time" + "Bars",
    )
    historical_token = "req" + "Historical" + "Data"
    historical_allowlist = {PROJECT_ROOT / "src" / "stocks" / "data" / "ibkr_historical.py"}
    market_data_allowlist = {
        PROJECT_ROOT / "src" / "stocks" / "live" / "quote.py",
        PROJECT_ROOT / "src" / "stocks" / "context" / "realtime_equity.py",
    }
    scanned_files = [
        PROJECT_ROOT / "main.py",
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
    ]

    for path in scanned_files:
        text = path.read_text(encoding="utf-8")
        for token in globally_forbidden:
            if path not in market_data_allowlist:
                assert token not in text, f"{token} found in {path}"
        if path not in historical_allowlist:
            assert historical_token not in text, f"{historical_token} found in {path}"
    for source in market_data_allowlist:
        market_source = source.read_text(encoding="utf-8")
        assert market_source.count("req" + "Mkt" + "Data") == 1
        assert "req" + "Real" + "Time" + "Bars" not in market_source


def test_no_account_identifiers_in_phase1_source() -> None:
    scanned_files = [
        PROJECT_ROOT / "main.py",
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
    ]

    for path in scanned_files:
        text = path.read_text(encoding="utf-8")
        assert "DU" + "123" not in text
        assert "DU" + "999" not in text
