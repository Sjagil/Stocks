from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from stocks.execution.idempotency import stable_hash


@dataclass(frozen=True)
class ShariahMethodology:
    methodology_id: str = "AAOIFI_RESEARCH_RECONSTRUCTION_V1"
    methodology_version: str = "1.0.0"
    business_activity_rules: str = "exclude prohibited primary business activities"
    debt_ratio_definition: str = "interest-bearing debt / market capitalization"
    debt_ratio_threshold: Decimal = Decimal("0.30")
    cash_interest_ratio_definition: str = "interest-bearing cash and securities / market capitalization"
    cash_interest_ratio_threshold: Decimal = Decimal("0.30")
    receivables_ratio_definition: str = "accounts receivable / total assets"
    receivables_ratio_threshold: Decimal = Decimal("0.49")
    income_impurity_threshold: Decimal = Decimal("0.05")
    market_value_denominator_rules: str = "decision-date market capitalization; no future price"
    screen_validity_days: int = 90

    @property
    def methodology_hash(self) -> str:
        return stable_hash(asdict(self))


def reconstruct_screen(
    *,
    screened_at: str,
    financial_statement_available_at: str | None,
    price_denominator_date: str | None,
    business_activity_result: bool | None,
    debt_ratio: Decimal | None,
    cash_interest_ratio: Decimal | None,
    receivables_ratio: Decimal | None,
    non_permissible_income_ratio: Decimal | None,
) -> dict[str, Any]:
    method = ShariahMethodology()
    values = (debt_ratio, cash_interest_ratio, receivables_ratio, non_permissible_income_ratio)
    if not financial_statement_available_at or not price_denominator_date or business_activity_result is None or any(value is None for value in values):
        status = "SHARIAH_DATA_INCOMPLETE"
    elif financial_statement_available_at > screened_at or price_denominator_date > screened_at:
        status = "SHARIAH_DATA_INCOMPLETE"
    elif not business_activity_result:
        status = "SHARIAH_INELIGIBLE_PIT"
    else:
        assert debt_ratio is not None
        assert cash_interest_ratio is not None
        assert receivables_ratio is not None
        assert non_permissible_income_ratio is not None
        if (
            debt_ratio > method.debt_ratio_threshold
            or cash_interest_ratio > method.cash_interest_ratio_threshold
            or receivables_ratio > method.receivables_ratio_threshold
            or non_permissible_income_ratio > method.income_impurity_threshold
        ):
            status = "SHARIAH_INELIGIBLE_PIT"
        else:
            status = "SHARIAH_ELIGIBLE_PIT"
    payload = {
        "screened_at": screened_at,
        "financial_statement_available_at": financial_statement_available_at,
        "price_denominator_date": price_denominator_date,
        "methodology_version": method.methodology_version,
        "business_activity_result": business_activity_result,
        "debt_ratio": str(debt_ratio) if debt_ratio is not None else None,
        "cash_interest_ratio": str(cash_interest_ratio) if cash_interest_ratio is not None else None,
        "receivables_ratio": str(receivables_ratio) if receivables_ratio is not None else None,
        "non_permissible_income_ratio": str(non_permissible_income_ratio) if non_permissible_income_ratio is not None else None,
        "final_status": status,
        "expiry": (datetime.fromisoformat(screened_at) + timedelta(days=method.screen_validity_days)).isoformat(),
    }
    payload["source_hash"] = stable_hash(payload)
    return payload


def screen_status_at(screen: dict[str, Any], decision_time: str) -> str:
    status = str(screen.get("final_status", "SHARIAH_DATA_INCOMPLETE"))
    expiry = screen.get("expiry")
    if status == "SHARIAH_ELIGIBLE_PIT" and expiry and datetime.fromisoformat(decision_time) > datetime.fromisoformat(str(expiry)):
        return "SHARIAH_STATUS_STALE"
    return status
