from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocks.ibkr import contract_queue
from stocks.ibkr.contract_queue import build_opportunity_contract_queue
from stocks.live.portfolio_targets import _contract_map as live_contract_map
from stocks.portfolio.manager import _contract_map as portfolio_contract_map


def test_queue_selects_only_contract_only_watchlist_candidates(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "output" / "portfolio" / "opportunity-funnel.json",
        {
            "content_hash": "A" * 64,
            "watchlist_candidates": [
                _candidate("ANET", 2, ["CONTRACT_IDENTITY_REQUIRED"]),
                _candidate(
                    "TXN",
                    1,
                    ["NOT_IN_CURRENT_STRATEGY_SIGNAL_SET"],
                ),
                _candidate(
                    "CSCO",
                    3,
                    [
                        "CONTRACT_IDENTITY_REQUIRED",
                        "MATERIAL_NEWS_RISK_REVIEW_REQUIRED",
                    ],
                ),
            ],
        },
    )
    _write_json(
        tmp_path / "output" / "screener" / "latest-preview.json",
        {
            "content_hash": "B" * 64,
            "records": [
                {
                    "symbol": "ANET",
                    "currency": "USD",
                    "exchange": "NYSE",
                }
            ],
        },
    )

    report = build_opportunity_contract_queue(tmp_path)

    assert report["status"] == "GO"
    assert report["request_count"] == 1
    assert report["requests"][0] == {
        "symbol": "ANET",
        "asset_class": "stock",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NYSE",
        "portfolio_rank": 2,
        "queue_reason": "SOLE_BLOCKER_CONTRACT_IDENTITY_REQUIRED",
    }
    assert report["execution_authority"] == "NONE"
    assert report["broker_calls"] == 0
    assert report["broker_writes"] == 0


def test_queue_keeps_missing_primary_exchange_visible_and_unqueued(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "output" / "portfolio" / "opportunity-funnel.json",
        {
            "watchlist_candidates": [
                _candidate("ANET", 1, ["CONTRACT_IDENTITY_REQUIRED"])
            ]
        },
    )
    _write_json(
        tmp_path / "output" / "screener" / "latest-preview.json",
        {"records": [{"symbol": "ANET", "currency": "USD", "exchange": "SMART"}]},
    )

    report = build_opportunity_contract_queue(tmp_path)

    assert report["status"] == "GO_EMPTY"
    assert report["request_count"] == 0
    assert report["skipped"] == [
        {
            "symbol": "ANET",
            "rank": 1,
            "reasons": ["PRIMARY_EXCHANGE_REQUIRED"],
        }
    ]


def test_queue_blocks_when_source_artifacts_are_missing(tmp_path: Path) -> None:
    report = build_opportunity_contract_queue(tmp_path)

    assert report["status"] == "SOURCE_ARTIFACT_BLOCKED"
    assert report["blockers"] == [
        "CURRENT_SCREENER_PREVIEW_REQUIRED",
        "OPPORTUNITY_FUNNEL_REQUIRED",
    ]
    assert report["broker_calls"] == 0


def test_queue_recovers_active_portfolio_broker_symbol_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_json(
        tmp_path / "output/portfolio/opportunity-funnel.json",
        {"watchlist_candidates": []},
    )
    _write_json(
        tmp_path / "output/screener/latest-preview.json",
        {"records": []},
    )
    _write_json(
        tmp_path / "output/portfolio/active_portfolio_plan.json",
        {
            "opportunities": {
                "opportunities": [
                    {
                        "ticker": "U-UN.TO",
                        "asset_type": "COMMODITY_CLOSED_END_TRUST",
                        "execution_blockers": [
                            "CONTRACT_IDENTITY_REQUIRED"
                        ],
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(
        contract_queue,
        "broad_asset_metadata",
        lambda _root: {
            "U-UN.TO": {
                "asset_type": "COMMODITY_CLOSED_END_TRUST",
                "broker_symbol": "U.UN",
                "currency": "CAD",
                "primary_exchange": "TSE",
            }
        },
    )

    report = build_opportunity_contract_queue(tmp_path)

    assert report["status"] == "GO"
    assert report["requests"] == [
        {
            "symbol": "U.UN",
            "asset_class": "commodity_etf",
            "currency": "CAD",
            "exchange": "SMART",
            "primary_exchange": "TSE",
            "portfolio_rank": 1,
            "queue_reason": "SOLE_BLOCKER_CONTRACT_IDENTITY_REQUIRED",
            "candidate_source": "CURRENT_ACTIVE_PORTFOLIO",
            "portfolio_symbol": "U-UN.TO",
        }
    ]
    assert report["broker_calls"] == 0


def test_portfolio_contract_maps_preserve_provider_to_broker_alias(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output/ibkr/contracts/stocks.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "U.UN",
                "security_type": "STK",
                "currency": "CAD",
                "con_id": 12345,
                "contract_hash": "HASH",
            }
        ]
    ).to_parquet(path, index=False)
    metadata = {"U-UN.TO": {"broker_symbol": "U.UN"}}

    manager_map = portfolio_contract_map(tmp_path, metadata)
    controlled_map = live_contract_map(path, asset_metadata=metadata)

    assert manager_map["U-UN.TO"]["con_id"] == 12345
    assert manager_map["U-UN.TO"]["broker_symbol"] == "U.UN"
    assert controlled_map["U-UN.TO"]["con_id"] == 12345
    assert controlled_map["U-UN.TO"]["portfolio_symbol"] == "U-UN.TO"


def _candidate(symbol: str, rank: int, reasons: list[str]) -> dict[str, object]:
    return {
        "symbol": symbol,
        "rank": rank,
        "asset_type": "STOCK",
        "rejection_reasons": reasons,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
