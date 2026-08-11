from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from stocks.research.phase11_2.shariah import ShariahMethodology, reconstruct_screen

from .export_manifest import stable_hash
from .import_audit import Phase113Store, database_path


BALANCE_CONCEPTS = {
    "Assets",
    "AccountsReceivableNetCurrent",
    "LongTermDebtCurrent",
    "LongTermDebtNoncurrent",
    "ShortTermBorrowings",
    "CashAndCashEquivalentsAtCarryingValue",
    "MarketableSecuritiesCurrent",
    "EntityCommonStockSharesOutstanding",
}


def build_pit_shariah_screens(root: Path, actuals: Iterable[dict[str, Any]]) -> dict[str, Any]:
    prices = _price_index(root)
    by_filing: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in actuals:
        accepted_at = str(row.get("accepted_at") or "")
        accession_hash = str(row.get("accession_hash") or "")
        if accepted_at and accession_hash:
            by_filing[(str(row.get("symbol")), accession_hash)].append(row)

    events: list[tuple[str, dict[str, Any]]] = []
    final_statuses: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    for (symbol, accession_hash), rows in by_filing.items():
        accepted_at = max(str(row["accepted_at"]) for row in rows)
        facts = _latest_balance_facts(rows)
        price_date, close = _price_before(prices.get(symbol, []), accepted_at[:10])
        shares = _decimal_value(facts.get("EntityCommonStockSharesOutstanding"))
        market_cap = close * shares if close is not None and shares is not None and shares > 0 else None
        debt = _sum_available(facts, ("LongTermDebtCurrent", "LongTermDebtNoncurrent", "ShortTermBorrowings"))
        cash = _sum_available(facts, ("CashAndCashEquivalentsAtCarryingValue", "MarketableSecuritiesCurrent"))
        receivables = _decimal_value(facts.get("AccountsReceivableNetCurrent"))
        assets = _decimal_value(facts.get("Assets"))
        debt_ratio = _safe_ratio(debt, market_cap)
        cash_ratio = _safe_ratio(cash, market_cap)
        receivables_ratio = _safe_ratio(receivables, assets)
        screen = reconstruct_screen(
            screened_at=accepted_at,
            financial_statement_available_at=accepted_at,
            price_denominator_date=price_date,
            business_activity_result=None,
            debt_ratio=debt_ratio,
            cash_interest_ratio=cash_ratio,
            receivables_ratio=receivables_ratio,
            non_permissible_income_ratio=None,
        )
        available = {
            "market_cap": market_cap is not None,
            "debt_ratio": debt_ratio is not None,
            "cash_interest_ratio": cash_ratio is not None,
            "receivables_ratio": receivables_ratio is not None,
            "business_activity": False,
            "non_permissible_income_ratio": False,
        }
        component_counts.update(key for key, value in available.items() if value)
        final_statuses[screen["final_status"]] += 1
        private = {
            **screen,
            "symbol": symbol,
            "accession_hash": accession_hash,
            "form": next((row.get("form") for row in rows if row.get("form")), None),
            "period_end": max((str(row.get("period_end")) for row in rows if row.get("period_end")), default=None),
            "available_components": available,
            "classification": "PIT_OBSERVED_COMPONENTS_INCOMPLETE",
            "methodology_id": ShariahMethodology().methodology_id,
            "methodology_hash": ShariahMethodology().methodology_hash,
        }
        events.append((f"{symbol}:{accession_hash}:{accepted_at}", private))
    Phase113Store(database_path(root)).append_events("SHARIAH_SCREEN", events)
    return {
        "screen_count": len(events),
        "symbol_count": len({key[0] for key in by_filing}),
        "final_status_counts": dict(final_statuses),
        "available_component_counts": dict(component_counts),
        "methodology_id": ShariahMethodology().methodology_id,
        "methodology_hash": ShariahMethodology().methodology_hash,
        "private_screen_content_hash": stable_hash([key for key, _ in events]),
    }


def _price_index(root: Path) -> dict[str, list[tuple[str, Decimal]]]:
    result: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)
    for record in Phase113Store(database_path(root)).iter_records("prices"):
        row = record["payload"]
        try:
            symbol = str(row["symbol"]).removesuffix(".US")
            timestamp = str(row["timestamp"])[:10]
            close = Decimal(str(row["close"]))
            date.fromisoformat(timestamp)
            if close > 0:
                result[symbol].append((timestamp, close))
        except (KeyError, InvalidOperation, ValueError):
            continue
    for symbol, values in result.items():
        result[symbol] = sorted(set(values))
    return result


def _price_before(values: list[tuple[str, Decimal]], target: str) -> tuple[str | None, Decimal | None]:
    if not values:
        return None, None
    index = bisect.bisect_left(values, (target, Decimal("-Infinity"))) - 1
    return values[index] if index >= 0 else (None, None)


def _latest_balance_facts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        concept = str(row.get("concept") or "")
        if concept not in BALANCE_CONCEPTS or str(row.get("unit") or "") not in {"USD", "shares"}:
            continue
        previous = selected.get(concept)
        if previous is None or str(row.get("period_end") or "") > str(previous.get("period_end") or ""):
            selected[concept] = row
    return selected


def _decimal_value(row: dict[str, Any] | None) -> Decimal | None:
    try:
        return Decimal(str(row["value"])) if row is not None else None
    except (InvalidOperation, KeyError, TypeError):
        return None


def _sum_available(facts: dict[str, Any], concepts: tuple[str, ...]) -> Decimal | None:
    values = [_decimal_value(facts.get(concept)) for concept in concepts]
    available = [value for value in values if value is not None]
    return sum(available, Decimal(0)) if available else None


def _safe_ratio(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator
