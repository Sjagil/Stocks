from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from stocks.analysis.theme_contracts import collect_theme_contracts


def _fixture(root: Path, *, read_only: bool = True) -> None:
    config = root / "config/themes/frontier_technology_energy_v1.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "themes": {
                    "quantum": {
                        "instruments": [
                            {
                                "symbol": "IONQ",
                                "ibkr_contract": {
                                    "symbol": "IONQ",
                                    "asset_class": "stock",
                                    "currency": "USD",
                                    "exchange": "SMART",
                                    "primary_exchange": "NYSE",
                                },
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (root / ".env.ibkr.live").write_text(
        "\n".join(
            [
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7496",
                "IBKR_CLIENT_ID=91",
                "IBKR_RECON_CLIENT_ID=92",
                "IBKR_QUOTE_CLIENT_ID=93",
                f"IBKR_READ_ONLY={'true' if read_only else 'false'}",
                "IBKR_ORDER_AUTHORITY=NONE",
                "IBKR_ALLOW_ORDER_TRANSMISSION=false",
                "IBKR_LIVE_TRADING_ENABLED=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_theme_contracts_resolve_with_strict_read_only_authority(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)

    class Result:
        def as_dict(self) -> dict:
            return {
                "status": "RESOLVED",
                "source": "TEST_READ_ONLY",
                "cache_hit": False,
                "returned_match_count": 1,
                "reason": "unique match",
                "resolved_contract": {
                    "contract": {
                        "conId": 123,
                        "symbol": "IONQ",
                        "localSymbol": "IONQ",
                        "secType": "STK",
                        "currency": "USD",
                        "exchange": "SMART",
                        "primaryExchange": "NYSE",
                    },
                    "contract_hash": "A" * 64,
                    "resolved_at": "2026-08-08T12:00:00+00:00",
                    "cache_ttl_seconds": 604800,
                },
                "read_only_ibkr_calls": {"req_contract_details": 1},
                "financial_calls": {
                    "place_order": 0,
                    "cancel_order": 0,
                    "global_cancel": 0,
                },
            }

    class Resolver:
        def resolve(self, request):
            assert request.symbol == "IONQ"
            assert request.primary_exchange == "NYSE"
            return Result()

    report = collect_theme_contracts(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
        resolver_factory=lambda _settings, _layout: Resolver(),
    )

    assert report["status"] == "GO"
    assert report["coverage_ratio"] == 1.0
    assert report["req_contract_details_calls"] == 1
    assert report["results"][0]["contract_identity"]["con_id"] == 123
    assert report["financial_calls"]["place_order"] == 0
    assert report["execution_authority"] == "NONE"
    assert report["live_trading_allowed"] is False


def test_theme_contracts_fail_closed_when_live_env_is_not_read_only(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, read_only=False)

    report = collect_theme_contracts(tmp_path)

    assert report["status"] == "PREFLIGHT_BLOCKED"
    assert "READ_ONLY_MODE_REQUIRED" in report["blockers"]
    assert report["req_contract_details_calls"] == 0
    assert report["financial_calls"]["place_order"] == 0
