from __future__ import annotations

import json
from pathlib import Path

from stocks.readiness import comprehensive_data_readiness


def _write(root: Path, relative: str, payload: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_comprehensive_readiness_distinguishes_core_data_from_desired_gaps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("stocks.readiness.data_source_status", lambda root: {"available_source_count": 3})
    monkeypatch.setattr(
        "stocks.readiness.sec_intelligence_status",
        lambda root: {
            "status": "GO",
            "structured_filing_count": 7,
            "structured_event_count": 19,
            "structured_symbol_count": 6,
            "authority": "RANKING_OVERLAY_ONLY",
        },
    )
    _write(tmp_path, "output/research/multitimeframe/cache-validation.json", {"status": "GO", "symbol_count": 10})
    _write(tmp_path, "data/corporate_actions/event_manifest.json", {"status": "GO", "event_count": 5})
    _write(tmp_path, "data/fx/fx_manifest.json", {"status": "GO", "currencies": ["EUR", "USD"]})
    _write(tmp_path, "output/macro/status.json", {"status": "GO", "latest_data_quality": "DATA_INCOMPLETE"})
    _write(
        tmp_path,
        "output/ibkr/phase11_3/status.json",
        {
            "status": "PHASE11_3_DATA_EVIDENCE_INCOMPLETE",
            "universe_size": 10,
            "fundamental_symbol_count": 8,
            "shariah_history_status": "SHARIAH_HISTORY_INCOMPLETE",
            "open_blockers": ["NEWS_ARCHIVE_PARTIAL"],
        },
    )
    _write(tmp_path, "output/market_context/status.json", {"gex": {"status": "GO", "historical_pit_backtest_allowed": False}})
    _write(tmp_path, "output/market_context/cot-context.json", {"status": "GO", "context_count": 2})
    _write(
        tmp_path,
        "output/analysis/groups/coverage.json",
        {
            "status": "GO_WITH_DOCUMENTED_GAPS",
            "sector_count": 12,
            "industry_count": 44,
            "all_sector_groups_analyzed": True,
            "all_industry_groups_analyzed": True,
            "signal_eligible_stock_count": 9,
            "signal_eligible_fundamental_count": 8,
            "signal_eligible_fundamental_coverage_ratio": 8 / 9,
            "signal_eligible_fundamental_missing_symbols": ["MISS"],
        },
    )
    _write(
        tmp_path,
        "output/portfolio/confluence-audit.json",
        {
            "status": "GO",
            "opportunity_count": 2,
            "status_counts": {"THREE_LAYER_CONFIRMED": 2},
            "technical_signal_required": True,
        },
    )

    report = comprehensive_data_readiness(tmp_path)

    assert report["status"] == "CORE_RESEARCH_DATA_GO_WITH_DOCUMENTED_GAPS"
    assert report["core_research_ready"] is True
    assert report["all_desired_data_available"] is False
    sec = report["layers"]["fundamentals_filings_news"]
    assert sec["sec_structured_event_count"] == 19
    assert sec["sec_standalone_entry_allowed"] is False
    assert report["layers"]["ibkr_news"]["status"] == "NOT_PROBED"
    groups = report["layers"]["sector_industry_intelligence"]
    assert groups["sector_count"] == 12
    assert groups["signal_eligible_fundamental_count"] == 8
    assert groups["signal_eligible_fundamental_missing_symbols"] == ["MISS"]
    confluence = report["layers"][
        "technical_fundamental_macro_confluence"
    ]
    assert confluence["status"] == "GO"
    assert confluence["opportunity_count"] == 2
    assert confluence["standalone_context_entry_allowed"] is False
    assert {item["data"] for item in report["open_data_gaps"]} >= {
        "HISTORICAL_SHARIAH_POINT_IN_TIME",
        "HISTORICAL_PIT_OPTIONS_CHAIN_AND_GEX",
        "OBSERVED_EQUITY_TAPE",
        "IBKR_TWS_HISTORICAL_NEWS",
    }
