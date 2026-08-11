from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values

from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_2.foundation import (
    parse_submissions,
    resolve_cik,
)
from stocks.research.phase11_2.providers import (
    SEC_BASE,
    SEC_TICKERS_URL,
    SafeJsonClient,
)


CONFIG_PATH = Path("config/themes/frontier_technology_energy_v1.json")
PRIVATE_ROOT = Path("data/research/themes/private/sec")
OUTPUT_PATH = Path("output/analysis/themes/fundamental-coverage.json")
VEHICLE_MARKERS = ("FUND", "VEHICLE")
CORE_METRIC_FIELDS = (
    "annual_revenue_growth",
    "annual_net_margin",
    "annual_free_cash_flow_margin",
    "quarter_revenue_yoy_growth",
    "quarter_net_margin",
    "cash_to_assets",
    "debt_to_assets",
)
REVENUE_TAGS = {
    "ContractsRevenue",
    "Revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "RevenueFromContractsWithCustomers",
    "Revenues",
    "SalesRevenueGoodsNet",
    "SalesRevenueNet",
    "SalesRevenueServicesNet",
}
NET_INCOME_TAGS = {
    "NetIncomeLoss",
    "ProfitLoss",
    "ProfitLossAttributableToOwnersOfParent",
}
OPERATING_CASH_FLOW_TAGS = {
    "CashFlowsFromUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
}
CAPEX_TAGS = {
    "PaymentsForAdditionsToPropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
}
ASSET_TAGS = {"Assets"}
CASH_TAGS = {
    "CashAndCashEquivalents",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
}
AGGREGATE_DEBT_TAGS = {
    "Borrowings",
    "DebtLongtermAndShorttermCombinedAmount",
}
CURRENT_DEBT_TAGS = {
    "CurrentPortionOfLongtermBorrowings",
    "DebtCurrent",
    "LongTermDebtCurrent",
}
LONG_DEBT_TAGS = {
    "LongtermBorrowings",
    "LongTermDebt",
    "LongTermDebtNoncurrent",
}
RELEVANT_FUNDAMENTAL_TAGS = frozenset().union(
    REVENUE_TAGS,
    NET_INCOME_TAGS,
    OPERATING_CASH_FLOW_TAGS,
    CAPEX_TAGS,
    ASSET_TAGS,
    CASH_TAGS,
    AGGREGATE_DEBT_TAGS,
    CURRENT_DEBT_TAGS,
    LONG_DEBT_TAGS,
)
RELEVANT_FILING_FORMS = {
    "10-K",
    "10-Q",
    "20-F",
    "40-F",
    "40-F/A",
    "6-K",
}


class JsonClient(Protocol):
    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[Any | None, int | None, int, str | None]: ...


def collect_theme_fundamentals(
    project_root: Path,
    *,
    client: JsonClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    config = _read_json(project_root / CONFIG_PATH)
    instruments = _instruments(config)
    if not instruments:
        return _blocked("THEME_CONFIG_UNAVAILABLE")
    collected_at = now or datetime.now(UTC)
    provider = client or SafeJsonClient(
        user_agent=_sec_user_agent(project_root),
        timeout_seconds=30,
        max_attempts=2,
        minimum_interval=0.2,
    )
    ticker_payload, ticker_status, ticker_attempts, ticker_error = (
        provider.get_json(SEC_TICKERS_URL)
    )
    rows: dict[str, dict[str, Any]] = {}
    provider_calls = 1
    for instrument in instruments:
        symbol = instrument["symbol"]
        if not _fundamentals_required(instrument):
            rows[symbol] = {
                "status": "NOT_APPLICABLE_VEHICLE",
                "fundamentals_required": False,
                "source": "THEME_MANIFEST_CLASSIFICATION",
            }
            continue
        cik = resolve_cik(ticker_payload, symbol)
        if not cik:
            rows[symbol] = {
                "status": "CIK_MAPPING_UNAVAILABLE",
                "fundamentals_required": True,
                "source": "SEC_EDGAR",
            }
            continue
        submissions_url = f"{SEC_BASE}/submissions/CIK{cik}.json"
        facts_url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        submissions, submissions_status, submissions_attempts, submissions_error = (
            provider.get_json(submissions_url)
        )
        companyfacts, facts_status, facts_attempts, facts_error = (
            provider.get_json(facts_url)
        )
        provider_calls += 2
        filings = parse_submissions(submissions)
        facts = _parse_relevant_companyfacts(
            companyfacts,
            filings,
            submissions,
        )
        _store_private(
            project_root,
            symbol,
            cik,
            submissions,
            companyfacts,
            collected_at,
        )
        metrics = _fundamental_metrics(facts)
        data_quality = _fundamental_data_quality(metrics, collected_at)
        rows[symbol] = {
            "status": (
                "AVAILABLE"
                if metrics.get("selected_fact_count", 0) > 0
                else "PROVIDER_AVAILABLE_UNSUPPORTED_OR_EMPTY_FACTS"
                if facts_status == 200
                else "PROVIDER_ERROR"
            ),
            "fundamentals_required": True,
            "source": "SEC_EDGAR_COMPANYFACTS",
            "fact_extraction_semantics": (
                "RELEVANT_CONCEPT_ALLOWLIST_NO_GLOBAL_TRUNCATION"
            ),
            "cik_fingerprint": f"cik-sha256:{stable_hash(cik)[:12].lower()}",
            "submissions_http_status": submissions_status,
            "companyfacts_http_status": facts_status,
            "attempt_count": submissions_attempts + facts_attempts,
            "error_classes": sorted(
                {
                    str(error)
                    for error in (submissions_error, facts_error)
                    if error
                }
            ),
            **metrics,
            "data_quality": data_quality,
        }
    required = [row for row in rows.values() if row["fundamentals_required"]]
    available = [row for row in required if row["status"] == "AVAILABLE"]
    decision_usable = [
        row
        for row in available
        if row.get("data_quality", {}).get("decision_usable")
    ]
    report = {
        "schema": "frontier_theme_sec_fundamental_coverage_v1",
        "status": (
            "GO"
            if required
            and len(available) == len(required)
            and len(decision_usable) == len(required)
            else "GO_WITH_DOCUMENTED_GAPS"
        ),
        "generated_at": collected_at.isoformat(),
        "config_hash": stable_hash(config),
        "ticker_mapping_http_status": ticker_status,
        "ticker_mapping_attempt_count": ticker_attempts,
        "ticker_mapping_error_class": ticker_error,
        "instrument_count": len(rows),
        "fundamental_required_count": len(required),
        "fundamental_available_count": len(available),
        "fundamental_coverage_ratio": _ratio(len(available), len(required)),
        "decision_usable_count": len(decision_usable),
        "decision_usable_ratio": _ratio(len(decision_usable), len(required)),
        "fact_extraction_semantics": (
            "RELEVANT_CONCEPT_ALLOWLIST_NO_GLOBAL_TRUNCATION"
        ),
        "quality_status_counts": dict(
            sorted(
                Counter(
                    str(row.get("data_quality", {}).get("status") or "UNKNOWN")
                    for row in required
                ).items()
            )
        ),
        "not_applicable_vehicle_count": sum(
            row["status"] == "NOT_APPLICABLE_VEHICLE" for row in rows.values()
        ),
        "missing_symbols": sorted(
            symbol
            for symbol, row in rows.items()
            if row["fundamentals_required"] and row["status"] != "AVAILABLE"
        ),
        "instruments": rows,
        "provider_calls": provider_calls,
        "broker_calls": 0,
        "orders_generated": 0,
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(report)
    _atomic_json(project_root / OUTPUT_PATH, report)
    return report


def _fundamental_data_quality(
    metrics: dict[str, Any],
    collected_at: datetime,
) -> dict[str, Any]:
    present = [key for key in CORE_METRIC_FIELDS if metrics.get(key) is not None]
    anomalous: set[str] = set()
    thresholds = {
        "annual_revenue_growth": 5.0,
        "quarter_revenue_yoy_growth": 5.0,
        "annual_net_margin": 3.0,
        "quarter_net_margin": 3.0,
        "annual_free_cash_flow_margin": 3.0,
    }
    for key, maximum_absolute in thresholds.items():
        value = _number(metrics.get(key))
        if value is not None and abs(value) > maximum_absolute:
            anomalous.add(key)
    for key in ("cash_to_assets", "debt_to_assets"):
        value = _number(metrics.get(key))
        if value is not None and (value < 0 or value > 2.0):
            anomalous.add(key)

    accepted = _parse_datetime(metrics.get("latest_accepted_at"))
    age_days = (
        (collected_at.astimezone(UTC) - accepted).total_seconds() / 86400.0
        if accepted is not None
        else None
    )
    stale = age_days is None or age_days > 550
    usable = [key for key in present if key not in anomalous]
    review_reasons = []
    if stale:
        review_reasons.append("LATEST_ACCEPTED_FILING_STALE_OR_UNKNOWN")
    if anomalous:
        review_reasons.append("EXTREME_RATIO_DENOMINATOR_REVIEW_REQUIRED")
    if len(usable) < 3:
        review_reasons.append("INSUFFICIENT_CORE_METRIC_COVERAGE")
    status = (
        "STALE"
        if stale
        else "REVIEW_REQUIRED"
        if anomalous
        else "LIMITED_CORE_METRICS"
        if len(usable) < 3
        else "GO"
    )
    return {
        "status": status,
        "decision_usable": status == "GO",
        "core_metric_count": len(present),
        "usable_core_metric_count": len(usable),
        "core_metric_total": len(CORE_METRIC_FIELDS),
        "metric_completeness_ratio": _ratio(
            len(present), len(CORE_METRIC_FIELDS)
        ),
        "anomalous_metric_fields": sorted(anomalous),
        "latest_accepted_age_days": (
            None if age_days is None else round(age_days, 3)
        ),
        "review_reasons": review_reasons,
        "semantics": (
            "PROVIDER_AVAILABILITY_IS_SEPARATE_FROM_DECISION_USABILITY; "
            "EXTREME_RATIOS_ARE_NOT_SILENTLY_SCORED"
        ),
    }


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fundamental_metrics(facts: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in facts if row.get("accepted_at")]
    reporting_currency = _reporting_currency(valid)
    annual = [
        row
        for row in valid
        if row.get("fiscal_period") == "FY"
        and row.get("form") in {"10-K", "20-F", "40-F", "40-F/A"}
        and _duration_in_range(row, 300, 430)
    ]
    quarters = [
        row
        for row in valid
        if row.get("fiscal_period") in {"Q1", "Q2", "Q3"}
        and row.get("form") in {"10-Q", "6-K"}
        and _duration_in_range(row, 60, 120)
    ]
    latest_annual_revenue, prior_annual_revenue = _latest_two_duration(
        annual,
        REVENUE_TAGS,
        reporting_currency,
    )
    annual_net_income, _ = _latest_two_duration(
        annual,
        NET_INCOME_TAGS,
        reporting_currency,
    )
    operating_cash_flow, _ = _latest_two_duration(
        annual,
        OPERATING_CASH_FLOW_TAGS,
        reporting_currency,
    )
    capex, _ = _latest_two_duration(
        annual,
        CAPEX_TAGS,
        reporting_currency,
    )
    quarter_revenue, prior_quarter_revenue = _latest_two_quarters(
        quarters,
        REVENUE_TAGS,
        reporting_currency,
    )
    quarter_net_income, _ = _latest_two_quarters(
        quarters,
        NET_INCOME_TAGS,
        reporting_currency,
    )
    assets = _latest_instant(valid, ASSET_TAGS, reporting_currency)
    cash = _latest_instant(
        valid,
        CASH_TAGS,
        reporting_currency,
    )
    aggregate_debt = _latest_instant(
        valid,
        AGGREGATE_DEBT_TAGS,
        reporting_currency,
    )
    current_debt = _latest_instant(
        valid,
        CURRENT_DEBT_TAGS,
        reporting_currency,
    ) or 0.0
    long_debt = _latest_instant(
        valid,
        LONG_DEBT_TAGS,
        reporting_currency,
    ) or 0.0
    debt = (
        aggregate_debt
        if aggregate_debt is not None
        else current_debt + long_debt if current_debt or long_debt else None
    )
    free_cash_flow = (
        None
        if operating_cash_flow is None or capex is None
        else operating_cash_flow - abs(capex)
    )
    latest_accepted = max(
        (str(row["accepted_at"]) for row in valid),
        default=None,
    )
    return {
        "provider_fact_count": len(facts),
        "selected_fact_count": len(valid),
        "reporting_currency": reporting_currency,
        "latest_accepted_at": latest_accepted,
        "latest_annual_revenue": _finite(latest_annual_revenue),
        "annual_revenue_growth": _growth(
            latest_annual_revenue, prior_annual_revenue
        ),
        "annual_net_income": _finite(annual_net_income),
        "annual_net_margin": _divide(annual_net_income, latest_annual_revenue),
        "annual_free_cash_flow": _finite(free_cash_flow),
        "annual_free_cash_flow_margin": _divide(
            free_cash_flow, latest_annual_revenue
        ),
        "latest_quarter_revenue": _finite(quarter_revenue),
        "quarter_revenue_yoy_growth": _growth(
            quarter_revenue, prior_quarter_revenue
        ),
        "latest_quarter_net_income": _finite(quarter_net_income),
        "quarter_net_margin": _divide(quarter_net_income, quarter_revenue),
        "assets": _finite(assets),
        "cash": _finite(cash),
        "debt": _finite(debt),
        "cash_to_assets": _divide(cash, assets),
        "debt_to_assets": _divide(debt, assets),
        "semantics": (
            "LATEST_CAUSALLY_ACCEPTED_SEC_FACTS; VALUES_MAY_REQUIRE_ISSUER_"
            "SPECIFIC_ACCOUNTING_REVIEW"
        ),
    }


def _latest_two_duration(
    rows: list[dict[str, Any]],
    concepts: set[str],
    reporting_currency: str | None,
) -> tuple[float | None, float | None]:
    return _latest_two(
        rows,
        concepts,
        reporting_currency,
        require_quarter_match=False,
    )


def _latest_two_quarters(
    rows: list[dict[str, Any]],
    concepts: set[str],
    reporting_currency: str | None,
) -> tuple[float | None, float | None]:
    return _latest_two(
        rows,
        concepts,
        reporting_currency,
        require_quarter_match=True,
    )


def _latest_two(
    rows: list[dict[str, Any]],
    concepts: set[str],
    reporting_currency: str | None,
    *,
    require_quarter_match: bool,
) -> tuple[float | None, float | None]:
    candidates = [
        row
        for row in rows
        if row.get("tag") in concepts
        and row.get("unit") == reporting_currency
        and _number(row.get("value")) is not None
    ]
    candidates.sort(
        key=lambda row: (
            str(row.get("period_end") or ""),
            str(row.get("accepted_at") or ""),
        ),
        reverse=True,
    )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in candidates:
        key = (
            row.get("period_end"),
            row.get("fiscal_period") if require_quarter_match else None,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    if not unique:
        return None, None
    latest = unique[0]
    prior = next(
        (
            row
            for row in unique[1:]
            if not require_quarter_match
            or row.get("fiscal_period") == latest.get("fiscal_period")
        ),
        None,
    )
    return _number(latest.get("value")), _number(
        prior.get("value") if prior else None
    )


def _latest_instant(
    rows: list[dict[str, Any]],
    concepts: set[str],
    reporting_currency: str | None,
) -> float | None:
    candidates = [
        row
        for row in rows
        if row.get("tag") in concepts
        and row.get("unit") == reporting_currency
        and _number(row.get("value")) is not None
    ]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda row: (
            str(row.get("period_end") or ""),
            str(row.get("accepted_at") or ""),
        ),
    )
    return _number(latest.get("value"))


def _parse_ifrs_companyfacts(
    payload: Any,
    filings: list[dict[str, Any]],
    submissions: Any,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    accepted_by_accession = {
        row["accession_hash"]: row.get("accepted_at") for row in filings
    }
    accepted_by_accession.update(_foreign_acceptance_map(submissions))
    concepts = (payload.get("facts") or {}).get("ifrs-full") or {}
    if not isinstance(concepts, dict):
        return []
    rows: list[dict[str, Any]] = []
    for tag, concept in concepts.items():
        units = concept.get("units") if isinstance(concept, dict) else None
        if not isinstance(units, dict):
            continue
        for unit, values in units.items():
            if not isinstance(values, list):
                continue
            for value in values[-20:]:
                if not isinstance(value, dict) or value.get("form") not in {
                    "40-F",
                    "40-F/A",
                    "6-K",
                }:
                    continue
                accession_hash = stable_hash(str(value.get("accn", "")))
                fact = {
                    "taxonomy": "ifrs-full",
                    "tag": tag,
                    "unit": unit,
                    "value": value.get("val"),
                    "period_start": value.get("start"),
                    "period_end": value.get("end"),
                    "filed_at": value.get("filed"),
                    "accepted_at": accepted_by_accession.get(accession_hash),
                    "accession_hash": accession_hash,
                    "form": value.get("form"),
                    "fiscal_year": value.get("fy"),
                    "fiscal_period": value.get("fp"),
                }
                fact["version_hash"] = stable_hash(fact)
                rows.append(fact)
    return rows[:5_000]


def _foreign_acceptance_map(submissions: Any) -> dict[str, str | None]:
    if not isinstance(submissions, dict):
        return {}
    recent = ((submissions.get("filings") or {}).get("recent") or {})
    forms = recent.get("form") or []
    accepted = recent.get("acceptanceDateTime") or []
    accessions = recent.get("accessionNumber") or []
    mapping: dict[str, str | None] = {}
    for index, form in enumerate(forms):
        if form not in {"40-F", "40-F/A", "6-K"}:
            continue
        accession = str(accessions[index]) if index < len(accessions) else ""
        accepted_at = str(accepted[index]) if index < len(accepted) else ""
        if accepted_at.endswith("Z"):
            accepted_at = accepted_at[:-1] + "+00:00"
        mapping[stable_hash(accession)] = accepted_at or None
    return mapping


def _reporting_currency(rows: list[dict[str, Any]]) -> str | None:
    counts = Counter(
        str(row.get("unit"))
        for row in rows
        if row.get("unit") in {"USD", "CAD", "EUR", "GBP"}
    )
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _duration_in_range(
    row: dict[str, Any],
    minimum_days: int,
    maximum_days: int,
) -> bool:
    try:
        start = date.fromisoformat(str(row.get("period_start")))
        end = date.fromisoformat(str(row.get("period_end")))
    except ValueError:
        return False
    return minimum_days <= (end - start).days <= maximum_days


def _parse_relevant_companyfacts(
    payload: Any,
    filings: list[dict[str, Any]],
    submissions: Any,
) -> list[dict[str, Any]]:
    """Extract only decision-relevant facts without a global row cap."""
    if not isinstance(payload, dict):
        return []
    accepted_by_accession = {
        row["accession_hash"]: row.get("accepted_at") for row in filings
    }
    accepted_by_accession.update(_foreign_acceptance_map(submissions))
    facts = payload.get("facts") or {}
    if not isinstance(facts, dict):
        return []

    rows: list[dict[str, Any]] = []
    for taxonomy, concepts in facts.items():
        if not isinstance(concepts, dict):
            continue
        for tag, concept in concepts.items():
            if tag not in RELEVANT_FUNDAMENTAL_TAGS:
                continue
            units = concept.get("units") if isinstance(concept, dict) else None
            if not isinstance(units, dict):
                continue
            for unit, values in units.items():
                if not isinstance(values, list):
                    continue
                for value in values:
                    if (
                        not isinstance(value, dict)
                        or value.get("form") not in RELEVANT_FILING_FORMS
                    ):
                        continue
                    accession_hash = stable_hash(str(value.get("accn", "")))
                    fact = {
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "unit": unit,
                        "value": value.get("val"),
                        "period_start": value.get("start"),
                        "period_end": value.get("end"),
                        "filed_at": value.get("filed"),
                        "accepted_at": accepted_by_accession.get(
                            accession_hash
                        ),
                        "accession_hash": accession_hash,
                        "form": value.get("form"),
                        "fiscal_year": value.get("fy"),
                        "fiscal_period": value.get("fp"),
                    }
                    fact["version_hash"] = stable_hash(fact)
                    rows.append(fact)
    rows.sort(
        key=lambda row: (
            str(row.get("taxonomy") or ""),
            str(row.get("tag") or ""),
            str(row.get("unit") or ""),
            str(row.get("period_end") or ""),
            str(row.get("accepted_at") or ""),
            str(row.get("version_hash") or ""),
        )
    )
    return rows


def _instruments(config: dict[str, Any]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for theme_id, definition in config.get("themes", {}).items():
        for raw in definition.get("instruments", []):
            symbol = str(raw.get("symbol") or "").upper()
            if not symbol:
                continue
            selected[symbol] = {
                **raw,
                "symbol": symbol,
                "theme": theme_id,
            }
    return [selected[symbol] for symbol in sorted(selected)]


def _fundamentals_required(instrument: dict[str, Any]) -> bool:
    maturity = str(instrument.get("business_maturity") or "").upper()
    return not any(marker in maturity for marker in VEHICLE_MARKERS)


def _store_private(
    project_root: Path,
    symbol: str,
    cik: str,
    submissions: Any,
    companyfacts: Any,
    collected_at: datetime,
) -> None:
    root = project_root / PRIVATE_ROOT / f"symbol={symbol}"
    root.mkdir(parents=True, exist_ok=True)
    for dataset, payload in (
        ("submissions", submissions),
        ("companyfacts", companyfacts),
    ):
        if payload is None:
            continue
        payload_hash = stable_hash(payload)
        path = root / f"{dataset}-{payload_hash}.json"
        if path.exists():
            continue
        private = {
            "schema": "frontier_theme_sec_raw_snapshot_v1",
            "symbol": symbol,
            "cik": cik,
            "dataset": dataset,
            "collected_at": collected_at.isoformat(),
            "payload_hash": payload_hash,
            "payload": payload,
        }
        path.write_text(
            json.dumps(private, sort_keys=True, default=str),
            encoding="utf-8",
        )


def _sec_user_agent(project_root: Path) -> str:
    values = (
        dotenv_values(project_root / ".env")
        if (project_root / ".env").is_file()
        else {}
    )
    value = os.environ.get("SEC_USER_AGENT") or values.get("SEC_USER_AGENT")
    return str(value or "StocksThemeResearch/1.0 research-contact@example.com")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite(value: float | None) -> float | None:
    return round(value, 4) if value is not None and math.isfinite(value) else None


def _divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return round(numerator / denominator, 6)


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0.0:
        return None
    return round(current / abs(previous) - 1.0, 6)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "frontier_theme_sec_fundamental_coverage_v1",
        "status": "BLOCKED",
        "reason": reason,
        "standalone_entry_allowed": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
