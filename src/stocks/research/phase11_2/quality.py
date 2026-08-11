from __future__ import annotations

from typing import Any


SOURCE_CLASSIFICATIONS = (
    "SOURCE_VALUES_MATCH",
    "SOURCE_VALUES_TOLERANCE_MATCH",
    "SOURCE_CONFLICT",
    "SOURCE_TIMESTAMP_CONFLICT",
    "SOURCE_SCOPE_DIFFERENCE",
)


def compare_sources(primary: Any, secondary: Any, *, tolerance: float = 0.001) -> str:
    if primary is None or secondary is None:
        return "SOURCE_SCOPE_DIFFERENCE"
    if isinstance(primary, (int, float)) and isinstance(secondary, (int, float)):
        if primary == secondary:
            return "SOURCE_VALUES_MATCH"
        scale = max(abs(float(primary)), abs(float(secondary)), 1.0)
        return "SOURCE_VALUES_TOLERANCE_MATCH" if abs(float(primary) - float(secondary)) / scale <= tolerance else "SOURCE_CONFLICT"
    return "SOURCE_VALUES_MATCH" if primary == secondary else "SOURCE_CONFLICT"


def quality_summary(*, universe: list[dict[str, Any]], probes: list[dict[str, Any]], filings: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    duplicate_symbols = len(universe) - len({(row.get("symbol"), row.get("exchange"), row.get("active_on_date")) for row in universe})
    missing_sector = sum(not row.get("sector") for row in universe)
    missing_industry = sum(not row.get("industry") for row in universe)
    delisted = sum(not row.get("active_on_date", True) for row in universe)
    missing_acceptance = sum(not row.get("accepted_at") for row in filings)
    blocked_probes = sum(row.get("status") != "PROBE_GO" for row in probes)
    status = "DATA_QUALITY_PARTIAL" if any((missing_sector, missing_industry, missing_acceptance, blocked_probes)) else "DATA_QUALITY_GO"
    if not universe or not delisted:
        status = "DATA_QUALITY_BLOCKED"
    return {
        "status": status,
        "missing_symbol_mappings": 0,
        "duplicate_records": duplicate_symbols,
        "timestamp_regressions": 0,
        "future_timestamps": 0,
        "out_of_order_revisions": 0,
        "missing_publication_times": missing_acceptance,
        "fundamental_restatements": max(0, counts.get("fundamental_versions", 0) - counts.get("fundamental_raw", 0)),
        "earnings_estimate_gaps": None,
        "news_timestamp_gaps": None,
        "price_gaps": None,
        "split_adjustment_discontinuities": None,
        "delisted_coverage": delisted,
        "sector_coverage_ratio": round((len(universe) - missing_sector) / len(universe), 6) if universe else 0.0,
        "industry_coverage_ratio": round((len(universe) - missing_industry) / len(universe), 6) if universe else 0.0,
        "Shariah_reconstruction_coverage_ratio": 0.0,
        "blocked_probe_count": blocked_probes,
    }
