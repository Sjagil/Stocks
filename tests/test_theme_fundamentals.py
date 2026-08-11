from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.analysis.theme_fundamentals import (
    _fundamental_data_quality,
    _fundamental_metrics,
    _parse_relevant_companyfacts,
    collect_theme_fundamentals,
)
from stocks.research.phase11_2.foundation import parse_submissions


class FakeClient:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads

    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[Any | None, int | None, int, str | None]:
        del headers
        payload = self.payloads.get(url)
        return payload, 200 if payload is not None else 404, 1, None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sec_theme_refresh_separates_companies_from_vehicles(
    tmp_path: Path,
) -> None:
    config = {
        "themes": {
            "quantum": {
                "instruments": [
                    {
                        "symbol": "AAA",
                        "business_maturity": "OPERATIONAL_COMPANY",
                    },
                    {
                        "symbol": "ETF",
                        "business_maturity": "DIVERSIFIED_FUND",
                    },
                ]
            }
        }
    }
    _write_json(
        tmp_path / "config/themes/frontier_technology_energy_v1.json",
        config,
    )
    tickers = {"0": {"ticker": "AAA", "cik_str": 123}}
    submissions_url = "https://data.sec.gov/submissions/CIK0000000123.json"
    facts_url = (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000123.json"
    )
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-K"],
                "acceptanceDateTime": [
                    "2026-02-01T10:00:00Z",
                    "2025-02-01T10:00:00Z",
                ],
                "accessionNumber": ["A1", "A0"],
                "filingDate": ["2026-02-01", "2025-02-01"],
                "reportDate": ["2025-12-31", "2024-12-31"],
            }
        }
    }
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 100.0,
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "filed": "2025-02-01",
                                "accn": "A0",
                                "form": "10-K",
                                "fy": 2024,
                                "fp": "FY",
                            },
                            {
                                "val": 120.0,
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "accn": "A1",
                                "form": "10-K",
                                "fy": 2025,
                                "fp": "FY",
                            },
                        ]
                    }
                }
            }
        }
    }
    client = FakeClient(
        {
            "https://www.sec.gov/files/company_tickers.json": tickers,
            submissions_url: submissions,
            facts_url: facts,
        }
    )

    report = collect_theme_fundamentals(
        tmp_path,
        client=client,
        now=datetime(2026, 8, 8, tzinfo=UTC),
    )

    assert report["status"] == "GO_WITH_DOCUMENTED_GAPS"
    assert report["fundamental_required_count"] == 1
    assert report["fundamental_available_count"] == 1
    assert report["decision_usable_count"] == 0
    assert report["not_applicable_vehicle_count"] == 1
    assert report["instruments"]["AAA"]["annual_revenue_growth"] == 0.2
    assert report["instruments"]["AAA"]["data_quality"]["status"] == (
        "LIMITED_CORE_METRICS"
    )
    assert report["instruments"]["AAA"]["reporting_currency"] == "USD"
    assert report["instruments"]["ETF"]["status"] == (
        "NOT_APPLICABLE_VEHICLE"
    )
    assert report["execution_authority"] == "NONE"
    assert report["broker_calls"] == 0


def test_sec_theme_refresh_supports_ifrs_40f_companyfacts(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "config/themes/frontier_technology_energy_v1.json",
        {
            "themes": {
                "nuclear": {
                    "instruments": [
                        {
                            "symbol": "CAN",
                            "business_maturity": "OPERATIONAL_COMPANY",
                        }
                    ]
                }
            }
        },
    )
    tickers = {"0": {"ticker": "CAN", "cik_str": 456}}
    submissions_url = "https://data.sec.gov/submissions/CIK0000000456.json"
    facts_url = (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000456.json"
    )
    submissions = {
        "filings": {
            "recent": {
                "form": ["40-F", "40-F"],
                "acceptanceDateTime": [
                    "2026-03-01T10:00:00Z",
                    "2025-03-01T10:00:00Z",
                ],
                "accessionNumber": ["F1", "F0"],
                "filingDate": ["2026-03-01", "2025-03-01"],
                "reportDate": ["2025-12-31", "2024-12-31"],
            }
        }
    }
    facts = {
        "facts": {
            "ifrs-full": {
                "Revenue": {
                    "units": {
                        "CAD": [
                            {
                                "val": 200.0,
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "filed": "2025-03-01",
                                "accn": "F0",
                                "form": "40-F",
                                "fy": 2024,
                                "fp": "FY",
                            },
                            {
                                "val": 250.0,
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "filed": "2026-03-01",
                                "accn": "F1",
                                "form": "40-F",
                                "fy": 2025,
                                "fp": "FY",
                            },
                        ]
                    }
                }
            }
        }
    }
    report = collect_theme_fundamentals(
        tmp_path,
        client=FakeClient(
            {
                "https://www.sec.gov/files/company_tickers.json": tickers,
                submissions_url: submissions,
                facts_url: facts,
            }
        ),
        now=datetime(2026, 8, 8, tzinfo=UTC),
    )

    instrument = report["instruments"]["CAN"]
    assert report["status"] == "GO_WITH_DOCUMENTED_GAPS"
    assert instrument["status"] == "AVAILABLE"
    assert instrument["data_quality"]["decision_usable"] is False
    assert instrument["reporting_currency"] == "CAD"
    assert instrument["annual_revenue_growth"] == 0.25


def test_extreme_ratios_are_flagged_for_review_not_silently_scored() -> None:
    quality = _fundamental_data_quality(
        {
            "latest_accepted_at": "2026-08-01T00:00:00+00:00",
            "annual_revenue_growth": 8.0,
            "annual_net_margin": -12.0,
            "cash_to_assets": 0.5,
            "debt_to_assets": 0.1,
        },
        datetime(2026, 8, 8, tzinfo=UTC),
    )

    assert quality["status"] == "REVIEW_REQUIRED"
    assert quality["decision_usable"] is False
    assert quality["anomalous_metric_fields"] == [
        "annual_net_margin",
        "annual_revenue_growth",
    ]


def test_relevant_companyfacts_are_not_lost_after_global_parser_cap() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-K"],
                "acceptanceDateTime": [
                    "2026-02-01T10:00:00Z",
                    "2025-02-01T10:00:00Z",
                ],
                "accessionNumber": ["A1", "A0"],
                "filingDate": ["2026-02-01", "2025-02-01"],
                "reportDate": ["2025-12-31", "2024-12-31"],
            }
        }
    }

    def annual(value: float, year: int, accession: str) -> dict[str, Any]:
        return {
            "val": value,
            "start": f"{year}-01-01",
            "end": f"{year}-12-31",
            "filed": f"{year + 1}-02-01",
            "accn": accession,
            "form": "10-K",
            "fy": year,
            "fp": "FY",
        }

    irrelevant = {
        f"AUnrelatedConcept{index:04d}": {
            "units": {
                "USD": [annual(float(index), 2025, "A1")]
                * 20
            }
        }
        for index in range(260)
    }
    relevant = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {
                "USD": [
                    annual(100.0, 2024, "A0"),
                    annual(120.0, 2025, "A1"),
                    annual(9_999.0, 2026, "UNKNOWN"),
                ]
            }
        },
        "NetIncomeLoss": {
            "units": {"USD": [annual(12.0, 2025, "A1")]}
        },
        "NetCashProvidedByUsedInOperatingActivities": {
            "units": {"USD": [annual(24.0, 2025, "A1")]}
        },
        "PaymentsToAcquirePropertyPlantAndEquipment": {
            "units": {"USD": [annual(6.0, 2025, "A1")]}
        },
        "Assets": {
            "units": {
                "USD": [
                    {
                        **annual(200.0, 2025, "A1"),
                        "start": None,
                    }
                ]
            }
        },
        "CashAndCashEquivalentsAtCarryingValue": {
            "units": {
                "USD": [
                    {
                        **annual(40.0, 2025, "A1"),
                        "start": None,
                    }
                ]
            }
        },
        "DebtLongtermAndShorttermCombinedAmount": {
            "units": {
                "USD": [
                    {
                        **annual(20.0, 2025, "A1"),
                        "start": None,
                    }
                ]
            }
        },
    }
    payload = {"facts": {"us-gaap": {**irrelevant, **relevant}}}

    facts = _parse_relevant_companyfacts(
        payload,
        parse_submissions(submissions),
        submissions,
    )
    metrics = _fundamental_metrics(facts)
    quality = _fundamental_data_quality(
        metrics,
        datetime(2026, 8, 8, tzinfo=UTC),
    )

    assert len(facts) == 9
    assert {row["tag"] for row in facts} == set(relevant)
    assert metrics["latest_annual_revenue"] == 120.0
    assert metrics["annual_revenue_growth"] == 0.2
    assert metrics["annual_net_margin"] == 0.1
    assert metrics["annual_free_cash_flow_margin"] == 0.15
    assert metrics["cash_to_assets"] == 0.2
    assert metrics["debt_to_assets"] == 0.1
    assert quality["status"] == "GO"
    assert quality["decision_usable"] is True
