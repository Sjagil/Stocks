#!/usr/bin/env python3
"""SEC ownership and event intelligence, causally safe and strategy-ready.

This module turns already-collected SEC filings into normalized economic events and
point-in-time features. It deliberately does not fetch from EDGAR. Feed it the raw
XML/HTML/text that your existing SEC store already contains.

Implemented filing families
---------------------------
* Forms 3/4/5: reporting-owner relationships, non-derivative and derivative
  transactions, open-market P/S classification, direct/indirect ownership,
  transaction value, position-relative size, footnotes and Rule 10b5-1 detection.
* Form 144: proposed-sale notices. These remain planned sales and are never treated
  as completed sales unless a later Form 4/5 sale supports them.
* Schedule 13D/13G: beneficial-ownership snapshots, amendments, percentage changes,
  active-ownership context and conservative activist-intent keyword detection.
* Form 13F-HR: delayed institutional position snapshots, new/increased/decreased/
  closed context relative to the manager's previous accepted report.
* Form 8-K: item-number classification, materiality, conservative directional risk
  flags and content-backed mover attribution.

Hard invariants
---------------
1. A filing is unavailable before its SEC accepted timestamp.
2. Report/transaction dates never override accepted_at for causal availability.
3. Insider/ownership/event intelligence is an overlay, never a standalone entry.
4. Derivative awards, gifts, tax withholding and option exercises do not masquerade
   as open-market insider conviction.
5. Form 144 is a notice of proposed sale, not proof that a sale occurred.
6. 13F is delayed context and never receives event-timing authority.

The code uses only the Python standard library and SQLite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

PARSER_VERSION = "1.0.0"
UTC = timezone.utc

OPEN_MARKET_CODES = {"P", "S"}
INSIDER_CODE_LABELS: dict[str, str] = {
    "P": "open_market_purchase",
    "S": "open_market_sale",
    "A": "grant_award_or_other_acquisition",
    "D": "disposition_to_issuer",
    "F": "tax_or_exercise_price_withholding",
    "M": "option_or_derivative_exercise",
    "C": "conversion_of_derivative",
    "E": "expiration_of_short_derivative",
    "G": "gift",
    "J": "other",
    "K": "equity_swap_or_similar",
    "U": "tender_of_shares",
    "W": "will_or_laws_of_descent",
    "X": "exercise_of_in_the_money_derivative",
    "Z": "deposit_or_withdrawal_from_voting_trust",
}

ROLE_WEIGHTS: dict[str, float] = {
    "ceo": 1.50,
    "cfo": 1.40,
    "coo": 1.30,
    "president": 1.25,
    "officer": 1.15,
    "director": 1.00,
    "ten_percent_owner": 0.80,
    "other": 0.70,
}

ITEM_8K_MAP: dict[str, dict[str, Any]] = {
    "1.01": {"category": "material_agreement", "materiality": 0.70, "base_direction": 0.00},
    "1.02": {"category": "agreement_termination", "materiality": 0.75, "base_direction": -0.20},
    "1.03": {"category": "bankruptcy_or_receivership", "materiality": 1.00, "base_direction": -1.00},
    "2.01": {"category": "acquisition_or_disposition", "materiality": 0.90, "base_direction": 0.00},
    "2.02": {"category": "earnings_or_financial_condition", "materiality": 0.90, "base_direction": 0.00},
    "2.03": {"category": "new_direct_obligation", "materiality": 0.75, "base_direction": -0.05},
    "2.04": {"category": "default_or_acceleration_trigger", "materiality": 1.00, "base_direction": -0.95},
    "2.05": {"category": "exit_or_disposal_costs", "materiality": 0.80, "base_direction": -0.45},
    "2.06": {"category": "material_impairment", "materiality": 0.90, "base_direction": -0.75},
    "3.01": {"category": "delisting_or_listing_failure", "materiality": 0.95, "base_direction": -0.90},
    "3.02": {"category": "unregistered_equity_sale", "materiality": 0.80, "base_direction": -0.45},
    "3.03": {"category": "security_holder_rights_change", "materiality": 0.75, "base_direction": -0.20},
    "4.01": {"category": "auditor_change", "materiality": 0.75, "base_direction": -0.15},
    "4.02": {"category": "nonreliance_or_restatement", "materiality": 1.00, "base_direction": -0.95},
    "5.01": {"category": "change_in_control", "materiality": 1.00, "base_direction": 0.00},
    "5.02": {"category": "management_or_director_change", "materiality": 0.75, "base_direction": 0.00},
    "5.03": {"category": "charter_bylaw_or_fiscal_year_change", "materiality": 0.55, "base_direction": 0.00},
    "5.07": {"category": "shareholder_vote", "materiality": 0.55, "base_direction": 0.00},
    "5.08": {"category": "director_nomination_deadline", "materiality": 0.45, "base_direction": 0.00},
    "6.01": {"category": "abs_informational_material", "materiality": 0.45, "base_direction": 0.00},
    "6.02": {"category": "change_of_servicer_or_trustee", "materiality": 0.55, "base_direction": 0.00},
    "6.03": {"category": "credit_enhancement_or_support_change", "materiality": 0.70, "base_direction": -0.10},
    "6.04": {"category": "failure_to_make_distribution", "materiality": 0.95, "base_direction": -0.90},
    "6.05": {"category": "securities_act_update", "materiality": 0.50, "base_direction": 0.00},
    "7.01": {"category": "regulation_fd_disclosure", "materiality": 0.60, "base_direction": 0.00},
    "8.01": {"category": "other_material_event", "materiality": 0.60, "base_direction": 0.00},
    "9.01": {"category": "financial_statements_and_exhibits", "materiality": 0.15, "base_direction": 0.00},
}

POSITIVE_8K_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\brais(?:e[sd]?|ing)\s+(?:full[- ]year\s+)?guidance\b", 0.75),
    (r"\bincreas(?:e[sd]?|ing)\s+(?:full[- ]year\s+)?guidance\b", 0.70),
    (r"\bexceed(?:ed|s|ing)?\s+(?:analyst|market|our)?\s*(?:expectations|estimates)\b", 0.55),
    (r"\brecord\s+(?:revenue|sales|earnings|bookings|backlog|cash flow)\b", 0.45),
    (r"\bspecial\s+dividend\b", 0.45),
    (r"\bshare\s+repurchase\s+(?:program|authorization)\b", 0.35),
    (r"\bcompleted\s+(?:the\s+)?acquisition\b", 0.25),
)

NEGATIVE_8K_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\bbankrupt(?:cy)?\b|\bchapter\s+(?:7|11)\b", -1.00),
    (r"\bgoing\s+concern\b", -0.90),
    (r"\bdefault(?:ed)?\b|\bacceleration\s+of\s+(?:debt|obligations)\b", -0.90),
    (r"\bmaterial\s+weakness\b", -0.75),
    (r"\brestat(?:e|ed|ement|ing)\b|\bshould\s+no\s+longer\s+be\s+relied\s+upon\b", -0.90),
    (r"\bimpairment\s+(?:charge|loss)\b", -0.70),
    (r"\blower(?:ed|s|ing)?\s+(?:full[- ]year\s+)?guidance\b", -0.75),
    (r"\bwithdraw(?:s|n|ing)?\s+(?:full[- ]year\s+)?guidance\b", -0.80),
    (r"\bmiss(?:ed|es|ing)?\s+(?:analyst|market)?\s*(?:expectations|estimates)\b", -0.55),
    (r"\bdelist(?:ed|ing)?\b|\bnoncompliance\s+with\s+(?:nasdaq|nyse)\b", -0.85),
    (r"\bresign(?:ed|ation)?\b|\bterminated\s+for\s+cause\b", -0.35),
    (r"\bdata\s+breach\b|\bcybersecurity\s+incident\b", -0.55),
    (r"\bunregistered\s+sale\s+of\s+equity\b|\bprivate\s+placement\b", -0.35),
)

ACTIVIST_PATTERNS: tuple[str, ...] = (
    r"\bseek(?:s|ing)?\s+(?:representation|seats?)\s+on\s+the\s+board\b",
    r"\bnominate\s+(?:one\s+or\s+more\s+)?directors?\b",
    r"\bproxy\s+(?:contest|solicitation)\b",
    r"\bstrategic\s+alternatives\b",
    r"\bchange\s+in\s+(?:the\s+)?board\b",
    r"\bchange\s+in\s+(?:the\s+)?management\b",
    r"\bchange\s+in\s+control\b",
    r"\btender\s+offer\b",
    r"\bmerger\b|\bacquisition\b|\bsale\s+of\s+(?:the\s+)?issuer\b",
    r"\breturn\s+capital\b|\bshare\s+repurchase\b",
)


class SecIntelligenceError(RuntimeError):
    """Base exception for ingestion and parsing failures."""


class CausalityError(SecIntelligenceError):
    """Raised when a filing lacks a usable accepted timestamp."""


@dataclass(frozen=True)
class FilingEnvelope:
    accession: str
    form_type: str
    accepted_at: datetime
    symbol: str | None = None
    issuer_cik: str | None = None
    filer_cik: str | None = None
    filing_date: date | None = None
    report_period: date | None = None
    source_path: str | None = None
    amends_accession: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accepted_at.tzinfo is None:
            raise CausalityError(
                f"Filing {self.accession} accepted_at must be timezone-aware; "
                "use FilingEnvelope.from_mapping with accepted_timezone when the source is naive"
            )
        object.__setattr__(self, "accepted_at", self.accepted_at.astimezone(UTC))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FilingEnvelope":
        accession = str(value.get("accession") or value.get("accession_number") or "").strip()
        form_type = str(value.get("form_type") or value.get("form") or "").strip().upper()
        if not accession:
            raise ValueError("Filing record is missing accession/accession_number")
        if not form_type:
            raise ValueError(f"Filing {accession} is missing form_type/form")
        accepted_raw = value.get("accepted_at") or value.get("acceptance_datetime") or value.get("accepted")
        if not accepted_raw:
            raise CausalityError(f"Filing {accession} has no accepted_at timestamp")
        metadata = dict(value.get("metadata") or {})
        known = {
            "accession", "accession_number", "form_type", "form", "accepted_at",
            "acceptance_datetime", "accepted", "symbol", "ticker", "issuer_cik",
            "cik", "filer_cik", "filing_date", "filed_at", "report_period",
            "period_of_report", "source_path", "primary_document", "content_path",
            "amends_accession", "accepted_timezone", "metadata", "content", "raw_content",
        }
        for key, item in value.items():
            if key not in known and key not in metadata:
                metadata[key] = item
        accepted_timezone = str(
            value.get("accepted_timezone")
            or metadata.get("accepted_timezone")
            or "America/New_York"
        )
        try:
            accepted_tz = ZoneInfo(accepted_timezone)
        except Exception as exc:
            raise ValueError(f"Unknown accepted_timezone: {accepted_timezone!r}") from exc
        return cls(
            accession=accession,
            form_type=form_type,
            accepted_at=parse_datetime(accepted_raw, assume_timezone=accepted_tz),
            symbol=_clean_symbol(value.get("symbol") or value.get("ticker")),
            issuer_cik=_clean_cik(value.get("issuer_cik") or value.get("cik")),
            filer_cik=_clean_cik(value.get("filer_cik")),
            filing_date=parse_date(value.get("filing_date") or value.get("filed_at")),
            report_period=parse_date(value.get("report_period") or value.get("period_of_report")),
            source_path=_none_if_blank(
                value.get("source_path") or value.get("primary_document") or value.get("content_path")
            ),
            amends_accession=_none_if_blank(value.get("amends_accession")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class ParsedEvent:
    event_key: str
    symbol: str | None
    issuer_cik: str | None
    form_group: str
    event_type: str
    effective_at: datetime
    accepted_at: datetime
    person_cik: str | None = None
    person_name: str | None = None
    role: str | None = None
    transaction_code: str | None = None
    direction: str | None = None
    shares: float | None = None
    price: float | None = None
    value_usd: float | None = None
    holdings_after: float | None = None
    ownership_form: str | None = None
    is_derivative: bool = False
    is_planned: bool = False
    is_10b5_1: bool = False
    percent_owned: float | None = None
    item_number: str | None = None
    materiality: float = 0.0
    direction_score: float = 0.0
    confidence: float = 1.0
    payload: Mapping[str, Any] = field(default_factory=dict)

    def as_db_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["effective_at"] = isoformat_z(self.effective_at)
        result["accepted_at"] = isoformat_z(self.accepted_at)
        result["is_derivative"] = int(self.is_derivative)
        result["is_planned"] = int(self.is_planned)
        result["is_10b5_1"] = int(self.is_10b5_1)
        result["payload_json"] = json.dumps(self.payload, sort_keys=True, separators=(",", ":"), default=str)
        result.pop("payload")
        return result


SecurityResolver = Callable[[str | None, str | None], str | None]


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_symbol(value: Any) -> str | None:
    text = _none_if_blank(value)
    return text.upper() if text else None


def _clean_cik(value: Any) -> str | None:
    text = _none_if_blank(value)
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return digits.zfill(10) if digits else text


def parse_datetime(value: Any, assume_timezone: timezone | ZoneInfo = UTC) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Empty datetime")
        text = text.replace("Z", "+00:00")
        compact_match = re.fullmatch(r"(\d{8})(\d{6})", text)
        if compact_match:
            dt = datetime.strptime(text, "%Y%m%d%H%M%S")
        else:
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                parsed = None
                for fmt in (
                    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                    "%m/%d/%Y %H:%M:%S", "%Y-%m-%d",
                ):
                    try:
                        parsed = datetime.strptime(text, fmt)
                        break
                    except ValueError:
                        continue
                if parsed is None:
                    raise
                dt = parsed
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=assume_timezone)
    return dt.astimezone(UTC)


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(f"Unsupported date: {value!r}") from exc


def isoformat_z(value: datetime) -> str:
    return parse_datetime(value).isoformat().replace("+00:00", "Z")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = html.unescape(str(value)).strip()
    if not text or text.lower() in {"n/a", "na", "none", "null", "--", "-"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[$,%\s,]", "", text.strip("()"))
    try:
        result = float(text)
    except ValueError:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        result = float(match.group(0))
    return -result if negative else result


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def _children(node: ET.Element, name: str) -> list[ET.Element]:
    target = name.lower()
    return [child for child in list(node) if _local_name(child.tag).lower() == target]


def _child(node: ET.Element | None, name: str) -> ET.Element | None:
    if node is None:
        return None
    found = _children(node, name)
    return found[0] if found else None


def _descendants(node: ET.Element, *names: str) -> list[ET.Element]:
    targets = {name.lower() for name in names}
    return [item for item in node.iter() if _local_name(item.tag).lower() in targets]


def _first_desc(node: ET.Element | None, *names: str) -> ET.Element | None:
    if node is None:
        return None
    targets = {name.lower() for name in names}
    for item in node.iter():
        if _local_name(item.tag).lower() in targets:
            return item
    return None


def _node_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = " ".join(part.strip() for part in node.itertext() if part and part.strip())
    return text or None


def _first_text(node: ET.Element | None, *names: str) -> str | None:
    return _node_text(_first_desc(node, *names))


def _nested_text(node: ET.Element | None, *path: str) -> str | None:
    current = node
    for name in path:
        current = _child(current, name)
        if current is None:
            return None
    return _node_text(current)


def _value_text(node: ET.Element | None, container: str) -> str | None:
    parent = _child(node, container) if node is not None else None
    if parent is None:
        parent = _first_desc(node, container) if node is not None else None
    return _nested_text(parent, "value") or _node_text(parent)


def _parse_xml(content: str) -> ET.Element | None:
    text = content.lstrip("\ufeff\x00 \t\r\n")
    candidates = [text]
    xml_blocks = re.findall(r"<XML>(.*?)</XML>", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(xml_blocks)
    for candidate in candidates:
        try:
            return ET.fromstring(candidate)
        except ET.ParseError:
            declaration = candidate.find("<?xml")
            if declaration > 0:
                try:
                    return ET.fromstring(candidate[declaration:])
                except ET.ParseError:
                    pass
    return None


def html_to_text(content: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", content)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>|</h\d>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[\t\r\f\v ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _event_key(*parts: Any) -> str:
    normalized = "|".join("" if part is None else str(part).strip().lower() for part in parts)
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:32]


def _bool_text(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "x"}


def _best_role(relationship: ET.Element | None) -> tuple[str, dict[str, Any]]:
    title = (_first_text(relationship, "officerTitle") or "").strip()
    title_lower = title.lower()
    flags = {
        "is_director": _bool_text(_first_text(relationship, "isDirector")),
        "is_officer": _bool_text(_first_text(relationship, "isOfficer")),
        "is_ten_percent_owner": _bool_text(_first_text(relationship, "isTenPercentOwner")),
        "is_other": _bool_text(_first_text(relationship, "isOther")),
        "officer_title": title or None,
        "other_text": _first_text(relationship, "otherText"),
    }
    if re.search(r"\bchief\s+executive\b|\bceo\b", title_lower):
        role = "ceo"
    elif re.search(r"\bchief\s+financial\b|\bcfo\b", title_lower):
        role = "cfo"
    elif re.search(r"\bchief\s+operating\b|\bcoo\b", title_lower):
        role = "coo"
    elif "president" in title_lower:
        role = "president"
    elif flags["is_officer"]:
        role = "officer"
    elif flags["is_director"]:
        role = "director"
    elif flags["is_ten_percent_owner"]:
        role = "ten_percent_owner"
    else:
        role = "other"
    return role, flags


def _extract_footnotes(root: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in _descendants(root, "footnote"):
        footnote_id = node.attrib.get("id") or node.attrib.get("ID")
        text = _node_text(node)
        if footnote_id and text:
            result[footnote_id] = text
    return result


def _transaction_footnote_text(node: ET.Element, footnotes: Mapping[str, str], remarks: str | None) -> str:
    ids: list[str] = []
    for item in _descendants(node, "footnoteId"):
        footnote_id = item.attrib.get("id") or item.attrib.get("ID") or _node_text(item)
        if footnote_id:
            ids.append(footnote_id)
    pieces = [footnotes[item] for item in ids if item in footnotes]
    if remarks:
        pieces.append(remarks)
    return " ".join(pieces).strip()


def _parse_ownership(
    envelope: FilingEnvelope,
    content: str,
    security_resolver: SecurityResolver | None,
) -> list[ParsedEvent]:
    root = _parse_xml(content)
    if root is None:
        raise SecIntelligenceError(f"Ownership filing {envelope.accession} is not parseable XML")
    symbol = envelope.symbol or _clean_symbol(_first_text(root, "issuerTradingSymbol"))
    issuer_cik = envelope.issuer_cik or _clean_cik(_first_text(root, "issuerCik"))
    if security_resolver and not symbol:
        symbol = _clean_symbol(security_resolver(None, issuer_cik))
    issuer_name = _first_text(root, "issuerName")
    period = parse_date(_first_text(root, "periodOfReport")) or envelope.report_period
    remarks = _first_text(root, "remarks")
    footnotes = _extract_footnotes(root)

    owners: list[dict[str, Any]] = []
    for owner_node in _descendants(root, "reportingOwner"):
        owner_id = _first_desc(owner_node, "reportingOwnerId")
        relationship = _first_desc(owner_node, "reportingOwnerRelationship")
        role, flags = _best_role(relationship)
        owners.append(
            {
                "cik": _clean_cik(_first_text(owner_id, "rptOwnerCik")),
                "name": _first_text(owner_id, "rptOwnerName"),
                "role": role,
                "relationship": flags,
            }
        )
    if not owners:
        owners = [{"cik": None, "name": None, "role": "other", "relationship": {}}]
    owner_allocation = 1.0 / max(1, len(owners))

    events: list[ParsedEvent] = []
    transactions: list[tuple[ET.Element, bool]] = []
    transactions.extend((node, False) for node in _descendants(root, "nonDerivativeTransaction"))
    transactions.extend((node, True) for node in _descendants(root, "derivativeTransaction"))

    for tx_index, (tx, is_derivative) in enumerate(transactions):
        tx_date = parse_date(_value_text(tx, "transactionDate")) or period or envelope.accepted_at.date()
        effective_at = datetime.combine(tx_date, time.min, tzinfo=UTC)
        code = (_first_text(_first_desc(tx, "transactionCoding"), "transactionCode") or "").strip().upper() or None
        acquired_disposed = (_value_text(tx, "transactionAcquiredDisposedCode") or "").strip().upper() or None
        if is_derivative:
            shares = safe_float(_value_text(tx, "transactionShares"))
            if shares is None:
                shares = safe_float(_value_text(tx, "transactionTotalValue"))
            price = safe_float(_value_text(tx, "transactionPricePerShare"))
            if price is None:
                price = safe_float(_value_text(tx, "conversionOrExercisePrice"))
            security_title = _value_text(tx, "securityTitle") or _value_text(tx, "derivativeSecurityTitle")
            underlying_title = _value_text(tx, "underlyingSecurityTitle")
            underlying_shares = safe_float(_value_text(tx, "underlyingSecurityShares"))
            holdings_after = safe_float(_value_text(tx, "derivativeSecuritySharesOwnedFollowingTransaction"))
        else:
            shares = safe_float(_value_text(tx, "transactionShares"))
            price = safe_float(_value_text(tx, "transactionPricePerShare"))
            security_title = _value_text(tx, "securityTitle")
            underlying_title = None
            underlying_shares = None
            holdings_after = safe_float(_value_text(tx, "sharesOwnedFollowingTransaction"))
        ownership_form = (_value_text(tx, "directOrIndirectOwnership") or "").strip().upper() or None
        indirect_nature = _value_text(tx, "natureOfOwnership")
        footnote_text = _transaction_footnote_text(tx, footnotes, remarks)
        is_10b5_1 = bool(re.search(r"\b10b5[- ]?1\b", footnote_text, flags=re.IGNORECASE))
        equity_swap = _bool_text(_first_text(_first_desc(tx, "transactionCoding"), "equitySwapInvolved"))
        direction = None
        if acquired_disposed == "A":
            direction = "acquisition"
        elif acquired_disposed == "D":
            direction = "disposition"
        value_usd = abs(shares * price) if shares is not None and price is not None else None
        signed_shares = None
        holdings_before = None
        pct_existing = None
        if shares is not None and acquired_disposed in {"A", "D"}:
            signed_shares = shares if acquired_disposed == "A" else -shares
            if holdings_after is not None:
                holdings_before = holdings_after - signed_shares
                if holdings_before > 0:
                    pct_existing = abs(shares) / holdings_before
        for owner_index, owner in enumerate(owners):
            payload = {
                "issuer_name": issuer_name,
                "security_title": security_title,
                "underlying_title": underlying_title,
                "underlying_shares": underlying_shares,
                "acquired_disposed": acquired_disposed,
                "signed_shares": signed_shares,
                "holdings_before": holdings_before,
                "transaction_fraction_of_prior_holdings": pct_existing,
                "indirect_ownership_nature": indirect_nature,
                "transaction_code_label": INSIDER_CODE_LABELS.get(code or "", "unknown"),
                "is_open_market": code in OPEN_MARKET_CODES and not is_derivative,
                "equity_swap_involved": equity_swap,
                "footnote_text": footnote_text or None,
                "relationship": owner["relationship"],
                "joint_filing_owner_count": len(owners),
                "economic_allocation_weight": owner_allocation,
                "tx_index": tx_index,
                "owner_index": owner_index,
            }
            key = _event_key(
                envelope.accession, "insider_transaction", tx_index, owner_index, owner["cik"],
                tx_date, code, shares, price, security_title, is_derivative,
            )
            events.append(
                ParsedEvent(
                    event_key=key,
                    symbol=symbol,
                    issuer_cik=issuer_cik,
                    form_group="insider_ownership",
                    event_type="insider_transaction",
                    effective_at=effective_at,
                    accepted_at=envelope.accepted_at,
                    person_cik=owner["cik"],
                    person_name=owner["name"],
                    role=owner["role"],
                    transaction_code=code,
                    direction=direction,
                    shares=shares,
                    price=price,
                    value_usd=value_usd,
                    holdings_after=holdings_after,
                    ownership_form=ownership_form,
                    is_derivative=is_derivative,
                    is_10b5_1=is_10b5_1,
                    materiality=0.75 if code in OPEN_MARKET_CODES and not is_derivative else 0.25,
                    confidence=0.98 if code else 0.70,
                    payload=payload,
                )
            )

    # Form 3 and some Form 4/5 filings include holdings without transactions. Store
    # them as baselines, but do not convert them into conviction.
    holdings: list[tuple[ET.Element, bool]] = []
    holdings.extend((node, False) for node in _descendants(root, "nonDerivativeHolding"))
    holdings.extend((node, True) for node in _descendants(root, "derivativeHolding"))
    for holding_index, (holding, is_derivative) in enumerate(holdings):
        security_title = _value_text(holding, "securityTitle") or _value_text(holding, "derivativeSecurityTitle")
        shares = safe_float(_value_text(holding, "sharesOwnedFollowingTransaction"))
        if shares is None:
            shares = safe_float(_value_text(holding, "derivativeSecuritySharesOwnedFollowingTransaction"))
        ownership_form = (_value_text(holding, "directOrIndirectOwnership") or "").strip().upper() or None
        for owner_index, owner in enumerate(owners):
            events.append(
                ParsedEvent(
                    event_key=_event_key(
                        envelope.accession, "insider_holding", holding_index, owner_index,
                        owner["cik"], security_title, shares, is_derivative,
                    ),
                    symbol=symbol,
                    issuer_cik=issuer_cik,
                    form_group="insider_ownership",
                    event_type="insider_holding_snapshot",
                    effective_at=envelope.accepted_at,
                    accepted_at=envelope.accepted_at,
                    person_cik=owner["cik"],
                    person_name=owner["name"],
                    role=owner["role"],
                    shares=shares,
                    holdings_after=shares,
                    ownership_form=ownership_form,
                    is_derivative=is_derivative,
                    materiality=0.10,
                    confidence=0.95,
                    payload={
                        "security_title": security_title,
                        "relationship": owner["relationship"],
                        "baseline_only": True,
                    },
                )
            )
    return events


def _iter_form144_sale_blocks(root: ET.Element) -> Iterator[ET.Element]:
    candidates = _descendants(root, "securitiesInformation", "securitiesToBeSold", "securityToBeSold")
    seen: set[int] = set()
    for node in candidates:
        if id(node) in seen:
            continue
        if _first_desc(node, "unitsToBeSold", "noOfUnitsSold", "numberOfSharesToBeSold") is None:
            continue
        seen.add(id(node))
        yield node
    if not candidates and _first_desc(root, "unitsToBeSold", "numberOfSharesToBeSold") is not None:
        yield root


def _parse_form144(
    envelope: FilingEnvelope,
    content: str,
    security_resolver: SecurityResolver | None,
) -> list[ParsedEvent]:
    root = _parse_xml(content)
    if root is None:
        raise SecIntelligenceError(f"Form 144 filing {envelope.accession} is not parseable XML")
    issuer_cik = envelope.issuer_cik or _clean_cik(_first_text(root, "issuerCik"))
    symbol = envelope.symbol or _clean_symbol(_first_text(root, "issuerTradingSymbol", "tickerSymbol"))
    issuer_name = _first_text(root, "issuerName")
    if security_resolver and not symbol:
        symbol = _clean_symbol(security_resolver(None, issuer_cik))
    person_name = _first_text(
        root,
        "nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold",
        "personName",
        "reportingPersonName",
    )
    person_cik = envelope.filer_cik or _clean_cik(_first_text(root, "filerCik", "reportingPersonCik"))
    events: list[ParsedEvent] = []
    seen_fingerprints: set[tuple[Any, ...]] = set()
    for index, block in enumerate(_iter_form144_sale_blocks(root)):
        security_title = _first_text(block, "securitiesClassTitle", "securityClassTitle", "titleOfClass")
        shares = safe_float(_first_text(block, "unitsToBeSold", "noOfUnitsSold", "numberOfSharesToBeSold"))
        market_value = safe_float(_first_text(block, "aggregateMarketValue", "marketValue"))
        units_outstanding = safe_float(_first_text(block, "unitsOutstanding", "numberOfSharesOutstanding"))
        approx_sale_date = parse_date(_first_text(block, "approxSaleDate", "approximateDateOfSale"))
        broker = _first_text(block, "brokerName", "nameOfBroker")
        exchange = _first_text(block, "securitiesExchangeName", "nameOfExchange")
        pct_outstanding = shares / units_outstanding if shares is not None and units_outstanding and units_outstanding > 0 else None
        fingerprint = (security_title, shares, market_value, approx_sale_date, broker)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        effective_date = approx_sale_date or envelope.accepted_at.date()
        events.append(
            ParsedEvent(
                event_key=_event_key(envelope.accession, "form144", index, *fingerprint),
                symbol=symbol,
                issuer_cik=issuer_cik,
                form_group="planned_sale",
                event_type="form144_proposed_sale",
                effective_at=datetime.combine(effective_date, time.min, tzinfo=UTC),
                accepted_at=envelope.accepted_at,
                person_cik=person_cik,
                person_name=person_name,
                direction="proposed_disposition",
                shares=shares,
                value_usd=market_value,
                is_planned=True,
                percent_owned=pct_outstanding,
                materiality=0.55,
                direction_score=-0.20,
                confidence=0.95,
                payload={
                    "issuer_name": issuer_name,
                    "security_title": security_title,
                    "units_outstanding": units_outstanding,
                    "planned_fraction_of_outstanding": pct_outstanding,
                    "approximate_sale_date": approx_sale_date.isoformat() if approx_sale_date else None,
                    "broker": broker,
                    "exchange": exchange,
                    "completed_sale": False,
                },
            )
        )
    return events


def _resolve_13f_symbol(
    envelope: FilingEnvelope,
    cusip: str | None,
    issuer_name: str | None,
    security_resolver: SecurityResolver | None,
) -> str | None:
    if security_resolver:
        resolved = security_resolver(cusip, None)
        if resolved:
            return _clean_symbol(resolved)
    mapping = envelope.metadata.get("cusip_to_symbol")
    if isinstance(mapping, Mapping) and cusip:
        resolved = mapping.get(cusip) or mapping.get(cusip.upper())
        if resolved:
            return _clean_symbol(resolved)
    if envelope.metadata.get("single_issuer_13f"):
        return envelope.symbol
    return None


def _parse_13f(
    envelope: FilingEnvelope,
    content: str,
    security_resolver: SecurityResolver | None,
) -> list[ParsedEvent]:
    root = _parse_xml(content)
    if root is None:
        raise SecIntelligenceError(f"13F filing {envelope.accession} is not parseable XML")
    manager_cik = envelope.filer_cik or envelope.issuer_cik or _clean_cik(
        _first_text(root, "cik", "filerCik", "filingManagerCik")
    )
    report_period = envelope.report_period or parse_date(
        _first_text(root, "periodOfReport", "reportCalendarOrQuarter")
    )
    multiplier_raw = envelope.metadata.get("thirteen_f_value_multiplier")
    if multiplier_raw is not None:
        value_multiplier = float(multiplier_raw)
    else:
        # SEC's 2023 XML specification changed the information-table value unit
        # from thousands of dollars to the nearest dollar.
        value_multiplier = 1_000.0 if envelope.accepted_at.date() < date(2023, 1, 3) else 1.0
    events: list[ParsedEvent] = []
    info_nodes = _descendants(root, "infoTable")
    for index, info in enumerate(info_nodes):
        issuer_name = _first_text(info, "nameOfIssuer")
        title = _first_text(info, "titleOfClass")
        cusip = (_first_text(info, "cusip") or "").strip().upper() or None
        figi = (_first_text(info, "figi") or "").strip().upper() or None
        reported_value = safe_float(_first_text(info, "value"))
        value_usd = reported_value * value_multiplier if reported_value is not None else None
        shares = safe_float(_first_text(info, "sshPrnamt", "sharesOrPrincipalAmount"))
        shares_type = _first_text(info, "sshPrnamtType")
        put_call = _first_text(info, "putCall")
        discretion = _first_text(info, "investmentDiscretion")
        sole = safe_float(_first_text(info, "Sole", "soleVotingAuthority"))
        shared = safe_float(_first_text(info, "Shared", "sharedVotingAuthority"))
        none = safe_float(_first_text(info, "None", "noVotingAuthority"))
        symbol = _resolve_13f_symbol(envelope, cusip, issuer_name, security_resolver)
        events.append(
            ParsedEvent(
                event_key=_event_key(envelope.accession, "13f_position", index, manager_cik, cusip, figi, shares, value_usd, put_call),
                symbol=symbol,
                issuer_cik=None,
                form_group="institutional_ownership",
                event_type="13f_position_snapshot",
                effective_at=envelope.accepted_at,
                accepted_at=envelope.accepted_at,
                person_cik=manager_cik,
                person_name=(
                    str(
                        envelope.metadata.get(
                            "manager_name"
                        )
                        or ""
                    ).strip()
                    or _first_text(
                        root,
                        "filingManagerName",
                        "nameOfInstitution",
                        "name",
                    )
                ),
                direction="delayed_snapshot",
                shares=shares,
                value_usd=value_usd,
                materiality=0.25,
                confidence=0.98,
                payload={
                    "manager_cik": manager_cik,
                    "report_period": report_period.isoformat() if report_period else None,
                    "issuer_name": issuer_name,
                    "title_of_class": title,
                    "cusip": cusip,
                    "figi": figi,
                    "put_call": put_call,
                    "shares_or_principal_type": shares_type,
                    "investment_discretion": discretion,
                    "voting_sole": sole,
                    "voting_shared": shared,
                    "voting_none": none,
                    "reported_value": reported_value,
                    "reported_value_multiplier": value_multiplier,
                    "delayed_context_only": True,
                },
            )
        )
    return events


def _flatten_xml(root: ET.Element) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for node in root.iter():
        name = _local_name(node.tag).lower()
        text = (node.text or "").strip()
        if text:
            result.setdefault(name, []).append(text)
    return result


def _first_matching_values(flat: Mapping[str, list[str]], patterns: Sequence[str]) -> list[str]:
    for pattern in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        values: list[str] = []
        for key, items in flat.items():
            if regex.search(key):
                values.extend(items)
        if values:
            return values
    return []


def _parse_13dg(
    envelope: FilingEnvelope,
    content: str,
    security_resolver: SecurityResolver | None,
) -> list[ParsedEvent]:
    root = _parse_xml(content)
    plain_text = html_to_text(content)
    flat = _flatten_xml(root) if root is not None else {}
    issuer_cik = envelope.issuer_cik or _clean_cik(
        (_first_matching_values(flat, [r"issuer.*cik", r"subject.*cik"]) or [None])[0]
    )
    symbol = envelope.symbol
    if security_resolver and not symbol:
        symbol = _clean_symbol(security_resolver(None, issuer_cik))
    schedule = "13D" if "13D" in envelope.form_type else "13G"

    names = _first_matching_values(
        flat,
        [r"reportingperson.*name", r"name.*reportingperson", r"reportingowner.*name"],
    )
    ciks = _first_matching_values(flat, [r"reportingperson.*cik", r"reportingowner.*cik"])
    percents_raw = _first_matching_values(
        flat,
        [r"percent.*class", r"percentage.*class", r"row11.*percent", r"percentofclass"],
    )
    shares_raw = _first_matching_values(
        flat,
        [r"aggregate.*beneficiallyowned", r"amount.*beneficiallyowned", r"aggregateamount"],
    )
    if not percents_raw:
        percent_matches = re.findall(
            r"Percent(?:age)?\s+of\s+Class[^\d]{0,100}(\d+(?:\.\d+)?)\s*%?",
            plain_text,
            flags=re.IGNORECASE,
        )
        percents_raw = percent_matches
    if not shares_raw:
        share_matches = re.findall(
            r"Aggregate\s+Amount\s+Beneficially\s+Owned[^\d]{0,100}([\d,]+)",
            plain_text,
            flags=re.IGNORECASE,
        )
        shares_raw = share_matches
    if not names:
        metadata_names = envelope.metadata.get("reporting_person_names")
        if isinstance(metadata_names, list):
            names = [str(item) for item in metadata_names]
        elif metadata_names:
            names = [str(metadata_names)]
    count = max(len(names), len(ciks), len(percents_raw), len(shares_raw), 1)

    lowered = plain_text.lower()
    activist_hits = [pattern for pattern in ACTIVIST_PATTERNS if re.search(pattern, lowered, flags=re.IGNORECASE)]
    active_ownership = schedule == "13D"
    activist_intent = active_ownership and bool(activist_hits)
    events: list[ParsedEvent] = []
    for index in range(count):
        name = names[index] if index < len(names) else (names[0] if len(names) == 1 else None)
        person_cik = _clean_cik(ciks[index] if index < len(ciks) else (ciks[0] if len(ciks) == 1 else None))
        percent_value = safe_float(percents_raw[index] if index < len(percents_raw) else (percents_raw[0] if len(percents_raw) == 1 else None))
        if percent_value is None:
            percent_fraction = None
        elif envelope.metadata.get("ownership_percent_is_fraction"):
            percent_fraction = percent_value
        else:
            percent_fraction = percent_value / 100.0
        shares = safe_float(shares_raw[index] if index < len(shares_raw) else (shares_raw[0] if len(shares_raw) == 1 else None))
        events.append(
            ParsedEvent(
                event_key=_event_key(envelope.accession, schedule, index, person_cik, name, percent_fraction, shares),
                symbol=symbol,
                issuer_cik=issuer_cik,
                form_group="beneficial_ownership",
                event_type="schedule_13d_13g_snapshot",
                effective_at=envelope.accepted_at,
                accepted_at=envelope.accepted_at,
                person_cik=person_cik,
                person_name=name,
                direction="ownership_snapshot",
                shares=shares,
                percent_owned=percent_fraction,
                materiality=0.85 if schedule == "13D" else 0.55,
                direction_score=0.15 if activist_intent else 0.0,
                confidence=0.90 if percent_fraction is not None else 0.65,
                payload={
                    "schedule": schedule,
                    "active_ownership_filing": active_ownership,
                    "activist_intent_detected": activist_intent,
                    "activist_keyword_hits": activist_hits,
                    "amendment": envelope.form_type.endswith("/A"),
                    "ownership_delta_requires_prior_snapshot": True,
                },
            )
        )
    return events


def _extract_8k_items(envelope: FilingEnvelope, content_text: str) -> list[str]:
    metadata_items = envelope.metadata.get("items") or envelope.metadata.get("item_numbers")
    values: list[str] = []
    if isinstance(metadata_items, str):
        values.extend(re.findall(r"\b\d\.\d{2}\b", metadata_items))
    elif isinstance(metadata_items, Sequence):
        for item in metadata_items:
            values.extend(re.findall(r"\b\d\.\d{2}\b", str(item)))
    if not values:
        values.extend(re.findall(r"\bItem\s+(\d\.\d{2})\b", content_text, flags=re.IGNORECASE))
    return sorted(set(values), key=lambda item: tuple(int(part) for part in item.split(".")))


def _content_direction_score(text: str) -> tuple[float, list[str]]:
    lowered = text.lower()
    evidence: list[tuple[str, float]] = []
    for pattern, weight in POSITIVE_8K_PATTERNS + NEGATIVE_8K_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            evidence.append((pattern, weight))
    if not evidence:
        return 0.0, []
    positive = sum(weight for _, weight in evidence if weight > 0)
    negative = sum(weight for _, weight in evidence if weight < 0)
    score = clamp(positive + negative, -1.0, 1.0)
    return score, [pattern for pattern, _ in evidence]


def _parse_8k(
    envelope: FilingEnvelope,
    content: str,
    security_resolver: SecurityResolver | None,
) -> list[ParsedEvent]:
    text = html_to_text(content)
    items = _extract_8k_items(envelope, text)
    if not items:
        items = ["8.01"]
    content_score, evidence = _content_direction_score(text)
    events: list[ParsedEvent] = []
    for item in items:
        spec = ITEM_8K_MAP.get(item, {"category": "unmapped_8k_item", "materiality": 0.35, "base_direction": 0.0})
        base = float(spec["base_direction"])
        if item in {"2.02", "7.01", "8.01", "5.02", "1.01", "2.01"}:
            directional = clamp(base + content_score, -1.0, 1.0)
        elif base < 0:
            directional = min(base, content_score) if content_score < 0 else base
        else:
            directional = base
        confidence = 0.90 if item != "8.01" or envelope.metadata.get("items") else 0.60
        events.append(
            ParsedEvent(
                event_key=_event_key(envelope.accession, "8k", item),
                symbol=envelope.symbol,
                issuer_cik=envelope.issuer_cik,
                form_group="current_event",
                event_type="8k_item",
                effective_at=envelope.accepted_at,
                accepted_at=envelope.accepted_at,
                item_number=item,
                materiality=float(spec["materiality"]),
                direction_score=directional,
                confidence=confidence,
                payload={
                    "category": spec["category"],
                    "content_direction_score": content_score,
                    "content_evidence_patterns": evidence,
                    "content_excerpt": text[:1_500] if text else None,
                    "attribution_eligible": item != "9.01" and float(spec["materiality"]) >= 0.50,
                },
            )
        )
    return events


def parse_filing(
    envelope: FilingEnvelope,
    content: str,
    security_resolver: SecurityResolver | None = None,
) -> list[ParsedEvent]:
    """Parse one filing into normalized economic events.

    The caller is responsible for providing raw content from the existing SEC store.
    A security resolver receives (CUSIP, issuer CIK) and may return a ticker.
    """
    if not content or not content.strip():
        raise ValueError(f"Filing {envelope.accession} has empty content")
    if envelope.accepted_at is None:
        raise CausalityError(f"Filing {envelope.accession} has no accepted_at")
    form = envelope.form_type.upper().strip()
    base_form = form.removesuffix("/A")
    if base_form in {"3", "4", "5"}:
        return _parse_ownership(envelope, content, security_resolver)
    if base_form == "144":
        return _parse_form144(envelope, content, security_resolver)
    if base_form.startswith("13F-HR"):
        return _parse_13f(envelope, content, security_resolver)
    if "13D" in base_form or "13G" in base_form:
        return _parse_13dg(envelope, content, security_resolver)
    if base_form == "8-K":
        return _parse_8k(envelope, content, security_resolver)
    return []


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sec_intel_filings (
    accession TEXT PRIMARY KEY,
    form_type TEXT NOT NULL,
    symbol TEXT,
    issuer_cik TEXT,
    filer_cik TEXT,
    accepted_at TEXT NOT NULL,
    filing_date TEXT,
    report_period TEXT,
    source_path TEXT,
    amends_accession TEXT,
    content_sha256 TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    parsed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sec_intel_filings_form_accept
    ON sec_intel_filings(form_type, accepted_at);
CREATE INDEX IF NOT EXISTS idx_sec_intel_filings_amends
    ON sec_intel_filings(amends_accession, accepted_at);

CREATE TABLE IF NOT EXISTS sec_intel_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accession TEXT NOT NULL REFERENCES sec_intel_filings(accession) ON DELETE CASCADE,
    event_key TEXT NOT NULL,
    symbol TEXT,
    issuer_cik TEXT,
    form_group TEXT NOT NULL,
    event_type TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    person_cik TEXT,
    person_name TEXT,
    role TEXT,
    transaction_code TEXT,
    direction TEXT,
    shares REAL,
    price REAL,
    value_usd REAL,
    holdings_after REAL,
    ownership_form TEXT,
    is_derivative INTEGER NOT NULL DEFAULT 0,
    is_planned INTEGER NOT NULL DEFAULT 0,
    is_10b5_1 INTEGER NOT NULL DEFAULT 0,
    percent_owned REAL,
    item_number TEXT,
    materiality REAL NOT NULL DEFAULT 0,
    direction_score REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 1,
    payload_json TEXT NOT NULL,
    UNIQUE(accession, event_key)
);

CREATE INDEX IF NOT EXISTS idx_sec_intel_events_symbol_accept
    ON sec_intel_events(symbol, accepted_at);
CREATE INDEX IF NOT EXISTS idx_sec_intel_events_type_symbol
    ON sec_intel_events(event_type, symbol, accepted_at);
CREATE INDEX IF NOT EXISTS idx_sec_intel_events_person
    ON sec_intel_events(person_cik, event_type, accepted_at);
"""


def connect_database(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def ingest_filing(
    conn: sqlite3.Connection,
    envelope: FilingEnvelope,
    content: str,
    security_resolver: SecurityResolver | None = None,
) -> int:
    """Parse and transactionally upsert one filing. Returns inserted event count."""
    events = parse_filing(envelope, content, security_resolver)
    ensure_schema(conn)
    now = isoformat_z(datetime.now(tz=UTC))
    with conn:
        conn.execute(
            """
            INSERT INTO sec_intel_filings (
                accession, form_type, symbol, issuer_cik, filer_cik, accepted_at,
                filing_date, report_period, source_path, amends_accession,
                content_sha256, parser_version, metadata_json, parsed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(accession) DO UPDATE SET
                form_type=excluded.form_type,
                symbol=excluded.symbol,
                issuer_cik=excluded.issuer_cik,
                filer_cik=excluded.filer_cik,
                accepted_at=excluded.accepted_at,
                filing_date=excluded.filing_date,
                report_period=excluded.report_period,
                source_path=excluded.source_path,
                amends_accession=excluded.amends_accession,
                content_sha256=excluded.content_sha256,
                parser_version=excluded.parser_version,
                metadata_json=excluded.metadata_json,
                parsed_at=excluded.parsed_at
            """,
            (
                envelope.accession,
                envelope.form_type,
                envelope.symbol,
                envelope.issuer_cik,
                envelope.filer_cik,
                isoformat_z(envelope.accepted_at),
                envelope.filing_date.isoformat() if envelope.filing_date else None,
                envelope.report_period.isoformat() if envelope.report_period else None,
                envelope.source_path,
                envelope.amends_accession,
                sha256_text(content),
                PARSER_VERSION,
                json.dumps(envelope.metadata, sort_keys=True, separators=(",", ":"), default=str),
                now,
            ),
        )
        conn.execute("DELETE FROM sec_intel_events WHERE accession = ?", (envelope.accession,))
        for event in events:
            row = event.as_db_mapping()
            conn.execute(
                """
                INSERT INTO sec_intel_events (
                    accession, event_key, symbol, issuer_cik, form_group, event_type,
                    effective_at, accepted_at, person_cik, person_name, role,
                    transaction_code, direction, shares, price, value_usd,
                    holdings_after, ownership_form, is_derivative, is_planned,
                    is_10b5_1, percent_owned, item_number, materiality,
                    direction_score, confidence, payload_json
                ) VALUES (
                    :accession, :event_key, :symbol, :issuer_cik, :form_group, :event_type,
                    :effective_at, :accepted_at, :person_cik, :person_name, :role,
                    :transaction_code, :direction, :shares, :price, :value_usd,
                    :holdings_after, :ownership_form, :is_derivative, :is_planned,
                    :is_10b5_1, :percent_owned, :item_number, :materiality,
                    :direction_score, :confidence, :payload_json
                )
                """,
                {"accession": envelope.accession, **row},
            )
    return len(events)


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["payload"] = json.loads(result.pop("payload_json") or "{}")
    for key in ("is_derivative", "is_planned", "is_10b5_1"):
        result[key] = bool(result[key])
    result["accepted_at"] = parse_datetime(result["accepted_at"])
    result["effective_at"] = parse_datetime(result["effective_at"])
    return result


def load_causal_events(
    conn: sqlite3.Connection,
    symbol: str,
    as_of: datetime,
    accepted_lookback_days: int = 550,
) -> list[dict[str, Any]]:
    """Load events that were actually public by as_of, excluding superseded filings."""
    cutoff = parse_datetime(as_of)
    start = cutoff - timedelta(days=accepted_lookback_days)
    rows = conn.execute(
        """
        SELECT e.*, f.form_type, f.report_period, f.filer_cik, f.amends_accession
        FROM sec_intel_events e
        JOIN sec_intel_filings f ON f.accession = e.accession
        WHERE e.symbol = ?
          AND e.accepted_at <= ?
          AND e.accepted_at >= ?
          AND NOT EXISTS (
              SELECT 1
              FROM sec_intel_filings amendment
              WHERE amendment.amends_accession = f.accession
                AND amendment.accepted_at <= ?
          )
        ORDER BY e.accepted_at ASC, e.id ASC
        """,
        (_clean_symbol(symbol), isoformat_z(cutoff), isoformat_z(start), isoformat_z(cutoff)),
    ).fetchall()
    return [_row_to_event(row) for row in rows]


def _deduplicate_insider_transactions(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    # Exact semantic duplicates frequently appear in joint/amended filings. Keep the
    # latest accepted observation, but retain distinct owners and changed economics.
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        payload = event["payload"]
        key = (
            event.get("person_cik") or event.get("person_name"),
            event["effective_at"].date(),
            event.get("transaction_code"),
            round(event.get("shares") or 0.0, 8),
            round(event.get("price") or 0.0, 8),
            payload.get("security_title"),
            event.get("is_derivative"),
            event.get("ownership_form"),
        )
        current = latest.get(key)
        if current is None or event["accepted_at"] > current["accepted_at"]:
            latest[key] = event
    return sorted(latest.values(), key=lambda event: (event["effective_at"], event["accepted_at"]))


def _insider_transaction_strength(event: Mapping[str, Any]) -> float:
    code = event.get("transaction_code")
    if code not in OPEN_MARKET_CODES or event.get("is_derivative"):
        return 0.0
    role_weight = ROLE_WEIGHTS.get(str(event.get("role") or "other"), ROLE_WEIGHTS["other"])
    ownership_weight = 1.0 if event.get("ownership_form") == "D" else 0.70
    plan_weight = 0.35 if event.get("is_10b5_1") and code == "S" else (0.70 if event.get("is_10b5_1") else 1.0)
    allocation = safe_float(event.get("payload", {}).get("economic_allocation_weight")) or 1.0
    value = abs(float(event.get("value_usd") or 0.0))
    size_component = clamp(math.log1p(value / 25_000.0) / 3.0, 0.15 if value > 0 else 0.0, 1.60)
    pct_existing = safe_float(event.get("payload", {}).get("transaction_fraction_of_prior_holdings"))
    ownership_component = clamp((pct_existing or 0.0) * 5.0, 0.0, 1.50)
    sign = 1.0 if code == "P" else -1.0
    return sign * role_weight * ownership_weight * plan_weight * allocation * (
        0.75 * size_component + 0.25 * ownership_component
    )


def _window(events: Iterable[dict[str, Any]], as_of: datetime, days: int) -> list[dict[str, Any]]:
    start = as_of - timedelta(days=days)
    return [event for event in events if start <= event["effective_at"] <= as_of]


def _person_key(event: Mapping[str, Any]) -> str:
    return str(event.get("person_cik") or event.get("person_name") or "unknown").lower()


def _allocated_event_value(event: Mapping[str, Any]) -> float:
    allocation = safe_float(event.get("payload", {}).get("economic_allocation_weight")) or 1.0
    return float(event.get("value_usd") or 0.0) * allocation


def _sec_as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value

    if isinstance(value, str) and value.strip():
        try:
            return parse_datetime(value)
        except (TypeError, ValueError):
            return None

    return None


def _sec_normalized_name_tokens(value: Any) -> set[str]:
    import re

    stopwords = {
        "THE",
        "TRUST",
        "TRUSTEE",
        "REVOCABLE",
        "IRREVOCABLE",
        "FAMILY",
        "LIVING",
        "ET",
        "AL",
        "JR",
        "SR",
        "II",
        "III",
    }

    tokens = set(
        re.findall(
            r"[A-Z0-9]+",
            str(value or "").upper(),
        )
    )

    return tokens - stopwords


def _sec_person_name_similarity(left: Any, right: Any) -> float:
    left_tokens = _sec_normalized_name_tokens(left)
    right_tokens = _sec_normalized_name_tokens(right)

    if not left_tokens or not right_tokens:
        return 0.0

    intersection = left_tokens & right_tokens

    return max(
        len(intersection) / len(left_tokens),
        len(intersection) / len(right_tokens),
    )


def _sec_economic_person_key(event: Mapping[str, Any]) -> str:
    tokens = sorted(
        _sec_normalized_name_tokens(
            event.get("person_name")
        )
    )

    if tokens:
        return "|".join(tokens)

    person_cik = str(
        event.get("person_cik") or ""
    ).strip()

    if person_cik:
        return person_cik

    return "UNKNOWN"


def _sec_datetime_key(value: Any) -> str:
    parsed = _sec_as_datetime(value)

    if parsed is not None:
        return parsed.isoformat()

    return str(value or "")


def _sec_aggregate_open_market_transactions(
    events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str, str, str, int],
        dict[str, Any],
    ] = {}

    for event in events:
        if event.get("event_type") != "insider_transaction":
            continue

        transaction_code = str(
            event.get("transaction_code") or ""
        ).upper()

        if transaction_code not in {"P", "S"}:
            continue

        if bool(event.get("is_derivative")):
            continue

        key = (
            str(event.get("accession") or ""),
            _sec_economic_person_key(event),
            _sec_datetime_key(event.get("effective_at")),
            transaction_code,
            str(event.get("ownership_form") or ""),
            str(event.get("role") or ""),
            int(bool(event.get("is_derivative"))),
        )

        aggregate = grouped.get(key)

        if aggregate is None:
            aggregate = dict(event)
            aggregate["shares"] = 0.0
            aggregate["value_usd"] = 0.0
            aggregate["is_10b5_1"] = False
            aggregate["economic_row_count"] = 0
            aggregate["source_event_keys"] = []
            grouped[key] = aggregate

        aggregate["shares"] += float(
            event.get("shares") or 0.0
        )
        aggregate["value_usd"] += float(
            event.get("value_usd") or 0.0
        )
        aggregate["is_10b5_1"] = (
            bool(aggregate.get("is_10b5_1"))
            or bool(event.get("is_10b5_1"))
        )
        aggregate["economic_row_count"] += 1

        event_key = event.get("event_key")
        if event_key:
            aggregate["source_event_keys"].append(
                str(event_key)
            )

    return sorted(
        grouped.values(),
        key=lambda item: (
            _sec_as_datetime(item.get("effective_at"))
            or _sec_as_datetime(item.get("accepted_at"))
            or datetime.min.replace(tzinfo=UTC),
            str(item.get("accession") or ""),
        ),
    )


def _sec_economic_transaction_strength(
    event: Mapping[str, Any],
) -> float:
    transaction_code = str(
        event.get("transaction_code") or ""
    ).upper()

    value_usd = abs(
        float(event.get("value_usd") or 0.0)
    )
    shares = abs(
        float(event.get("shares") or 0.0)
    )

    if value_usd > 0:
        magnitude = math.log1p(
            value_usd / 100_000.0
        ) / 3.5
    elif shares > 0:
        magnitude = math.log1p(
            shares / 1_000.0
        ) / 3.5
    else:
        magnitude = 0.15

    magnitude = clamp(
        magnitude,
        0.15,
        1.50,
    )

    role = str(
        event.get("role") or ""
    ).lower()

    if transaction_code == "P":
        role_multiplier = 1.0

        if role in {"ceo", "cfo"}:
            role_multiplier = 1.25
        elif role == "director":
            role_multiplier = 1.10

        return magnitude * role_multiplier

    if transaction_code == "S":
        sale_weight = (
            0.10
            if bool(event.get("is_10b5_1"))
            else 1.00
        )

        return -magnitude * sale_weight

    return 0.0


def _compute_insider_features(
    events: Sequence[dict[str, Any]],
    as_of: datetime,
) -> dict[str, Any]:
    open_market = _sec_aggregate_open_market_transactions(
        events
    )

    result: dict[str, Any] = {}

    for days in (7, 30, 90):
        recent = _window(
            open_market,
            as_of,
            days,
        )

        purchases = [
            event
            for event in recent
            if str(
                event.get("transaction_code") or ""
            ).upper() == "P"
        ]
        sales = [
            event
            for event in recent
            if str(
                event.get("transaction_code") or ""
            ).upper() == "S"
        ]

        result[f"insider_buy_count_{days}d"] = len(
            purchases
        )
        result[f"insider_sale_count_{days}d"] = len(
            sales
        )

        buy_value = sum(
            float(event.get("value_usd") or 0.0)
            for event in purchases
        )
        sale_value = sum(
            float(event.get("value_usd") or 0.0)
            for event in sales
        )

        result[f"insider_buy_value_{days}d"] = buy_value
        result[f"insider_sale_value_{days}d"] = sale_value
        result[f"insider_net_value_{days}d"] = (
            buy_value - sale_value
        )

        result[
            f"unique_insider_buyers_{days}d"
        ] = len(
            {
                _sec_economic_person_key(event)
                for event in purchases
            }
        )
        result[
            f"unique_insider_sellers_{days}d"
        ] = len(
            {
                _sec_economic_person_key(event)
                for event in sales
            }
        )

    recent30 = _window(
        open_market,
        as_of,
        30,
    )

    purchases30 = [
        event
        for event in recent30
        if str(
            event.get("transaction_code") or ""
        ).upper() == "P"
    ]

    unique_buyers30 = {
        _sec_economic_person_key(event)
        for event in purchases30
    }

    result["cluster_buying_30d"] = (
        len(unique_buyers30) >= 3
        and len(purchases30) >= 3
    )

    result["ceo_cfo_buy_30d"] = any(
        str(
            event.get("transaction_code") or ""
        ).upper() == "P"
        and str(
            event.get("role") or ""
        ).lower() in {"ceo", "cfo"}
        for event in recent30
    )

    recent90 = _window(
        open_market,
        as_of,
        90,
    )

    result[
        "direct_ownership_transaction_ratio_90d"
    ] = (
        sum(
            1
            for event in recent90
            if str(
                event.get("ownership_form") or ""
            ).upper() == "D"
        )
        / max(1, len(recent90))
    )

    sales90 = [
        event
        for event in recent90
        if str(
            event.get("transaction_code") or ""
        ).upper() == "S"
    ]

    total_sale_value = sum(
        abs(float(event.get("value_usd") or 0.0))
        for event in sales90
    )
    planned_sale_value = sum(
        abs(float(event.get("value_usd") or 0.0))
        for event in sales90
        if bool(event.get("is_10b5_1"))
    )

    if total_sale_value > 0:
        result["rule_10b5_1_sale_ratio_90d"] = (
            planned_sale_value / total_sale_value
        )
    else:
        result["rule_10b5_1_sale_ratio_90d"] = 0.0

    person_events: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for event in open_market:
        person_events.setdefault(
            _sec_economic_person_key(event),
            [],
        ).append(event)

    max_streak = 0

    for values in person_events.values():
        streak = 0

        for event in sorted(
            values,
            key=lambda item: (
                _sec_as_datetime(
                    item.get("effective_at")
                )
                or _sec_as_datetime(
                    item.get("accepted_at")
                )
                or datetime.min.replace(tzinfo=UTC)
            ),
            reverse=True,
        ):
            if str(
                event.get("transaction_code") or ""
            ).upper() == "P":
                streak += 1
            else:
                break

        max_streak = max(
            max_streak,
            streak,
        )

    result["max_insider_purchase_streak"] = max_streak

    purchase_strength_by_person: dict[str, float] = {}
    planned_sale_strength_by_person: dict[str, float] = {}
    discretionary_sale_strength_by_person: dict[str, float] = {}

    for event in recent90:
        person_key = _sec_economic_person_key(event)

        transaction_code = str(
            event.get("transaction_code") or ""
        ).upper()

        strength = _sec_economic_transaction_strength(
            event
        )

        if transaction_code == "P":
            purchase_strength_by_person[person_key] = (
                purchase_strength_by_person.get(
                    person_key,
                    0.0,
                )
                + strength
            )

        elif (
            transaction_code == "S"
            and bool(event.get("is_10b5_1"))
        ):
            planned_sale_strength_by_person[person_key] = (
                planned_sale_strength_by_person.get(
                    person_key,
                    0.0,
                )
                + strength
            )

        elif transaction_code == "S":
            discretionary_sale_strength_by_person[person_key] = (
                discretionary_sale_strength_by_person.get(
                    person_key,
                    0.0,
                )
                + strength
            )

    purchase_strength = sum(
        min(value, 1.50)
        for value in purchase_strength_by_person.values()
    )

    planned_sale_strength = sum(
        max(value, -0.25)
        for value in planned_sale_strength_by_person.values()
    )

    discretionary_sale_strength = sum(
        max(value, -0.75)
        for value in discretionary_sale_strength_by_person.values()
    )

    strength90 = (
        purchase_strength
        + planned_sale_strength
        + discretionary_sale_strength
    )

    cluster_bonus = (
        0.35
        if result["cluster_buying_30d"]
        else 0.0
    )
    executive_bonus = (
        0.20
        if result["ceo_cfo_buy_30d"]
        else 0.0
    )

    result["net_insider_conviction"] = math.tanh(
        strength90 / 3.5
        + cluster_bonus
        + executive_bonus
    )

    result["insider_feature_observation_count"] = len(
        open_market
    )

    return result



def _compute_form144_features(
    events: Sequence[dict[str, Any]],
    as_of: datetime,
) -> dict[str, Any]:
    notices = [
        event
        for event in events
        if event.get("event_type")
        == "form144_proposed_sale"
    ]

    economic_transactions = (
        _sec_aggregate_open_market_transactions(events)
    )

    sales = [
        event
        for event in economic_transactions
        if str(
            event.get("transaction_code") or ""
        ).upper() == "S"
        and not bool(event.get("is_derivative"))
    ]

    notice_start = as_of - timedelta(days=90)

    current = [
        event
        for event in notices
        if (
            notice_start
            <= event["accepted_at"]
            <= as_of
        )
    ]

    candidate_pairs: list[
        tuple[float, int, int]
    ] = []

    for notice_index, notice in enumerate(current):
        notice_name = notice.get("person_name")
        notice_shares = abs(
            float(notice.get("shares") or 0.0)
        )
        notice_accepted = _sec_as_datetime(
            notice.get("accepted_at")
        )
        notice_effective = (
            _sec_as_datetime(
                notice.get("effective_at")
            )
            or notice_accepted
        )

        if (
            notice_accepted is None
            or notice_effective is None
            or notice_shares <= 0
        ):
            continue

        for sale_index, sale in enumerate(sales):
            sale_name = sale.get("person_name")
            sale_shares = abs(
                float(sale.get("shares") or 0.0)
            )
            sale_accepted = _sec_as_datetime(
                sale.get("accepted_at")
            )
            sale_effective = (
                _sec_as_datetime(
                    sale.get("effective_at")
                )
                or sale_accepted
            )

            if (
                sale_accepted is None
                or sale_effective is None
                or sale_shares <= 0
            ):
                continue

            accepted_gap_days = (
                sale_accepted - notice_accepted
            ).total_seconds() / 86_400.0

            effective_gap_days = (
                sale_effective - notice_effective
            ).total_seconds() / 86_400.0

            if (
                accepted_gap_days < -1.0
                or accepted_gap_days > 10.0
            ):
                continue

            if abs(effective_gap_days) > 2.0:
                continue

            name_similarity = (
                _sec_person_name_similarity(
                    notice_name,
                    sale_name,
                )
            )

            if name_similarity < 0.50:
                continue

            share_difference = abs(
                notice_shares - sale_shares
            ) / max(
                notice_shares,
                sale_shares,
                1.0,
            )

            if share_difference > 0.15:
                continue

            share_score = max(
                0.0,
                1.0 - share_difference,
            )
            date_score = max(
                0.0,
                1.0
                - abs(effective_gap_days) / 45.0,
            )

            score = (
                0.50 * name_similarity
                + 0.40 * share_score
                + 0.10 * date_score
            )

            # Een Form 4 die iets eerder werd geaccepteerd
            # kan dezelfde economische transactie bevatten,
            # maar krijgt een lagere matchprioriteit.
            if accepted_gap_days < 0:
                score -= 0.15

            candidate_pairs.append(
                (
                    score,
                    notice_index,
                    sale_index,
                )
            )

    candidate_pairs.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    matched_notices: set[int] = set()
    matched_sales: set[int] = set()

    for score, notice_index, sale_index in candidate_pairs:
        if score < 0.65:
            continue

        if notice_index in matched_notices:
            continue

        if sale_index in matched_sales:
            continue

        matched_notices.add(notice_index)
        matched_sales.add(sale_index)

    unconfirmed_value = sum(
        float(notice.get("value_usd") or 0.0)
        for index, notice in enumerate(current)
        if index not in matched_notices
    )

    if unconfirmed_value > 0:
        raw_pressure = math.tanh(
            math.log1p(
                unconfirmed_value / 1_000_000.0
            ) / 5.0
        )
        pressure = -0.30 * raw_pressure
    else:
        pressure = 0.0

    return {
        "form144_notice_count_90d": len(current),
        "form144_planned_sale_value_90d": sum(
            float(item.get("value_usd") or 0.0)
            for item in current
        ),
        "form144_confirmed_by_later_form4_count_90d": (
            len(matched_notices)
        ),
        "form144_unconfirmed_sale_value_90d": (
            unconfirmed_value
        ),
        "form144_unconfirmed_sale_pressure": pressure,
    }



def _compute_13dg_features(events: Sequence[dict[str, Any]], as_of: datetime) -> dict[str, Any]:
    snapshots = [event for event in events if event["event_type"] == "schedule_13d_13g_snapshot"]
    by_person: dict[str, list[dict[str, Any]]] = {}
    for event in snapshots:
        by_person.setdefault(_person_key(event), []).append(event)
    deltas: list[float] = []
    activist = False
    active_filers = 0
    passive_filers = 0
    latest_percentages: list[float] = []
    for values in by_person.values():
        ordered = sorted(values, key=lambda item: item["accepted_at"])
        latest = ordered[-1]
        latest_pct = safe_float(latest.get("percent_owned"))
        if latest_pct is not None:
            latest_percentages.append(latest_pct)
        schedule = latest["payload"].get("schedule")
        if schedule == "13D":
            active_filers += 1
        elif schedule == "13G":
            passive_filers += 1
        activist = activist or bool(latest["payload"].get("activist_intent_detected"))
        if len(ordered) >= 2:
            previous_pct = safe_float(ordered[-2].get("percent_owned"))
            if latest_pct is not None and previous_pct is not None:
                deltas.append(latest_pct - previous_pct)
    net_delta = sum(deltas)
    context = math.tanh(net_delta * 8.0 + (0.30 if activist else 0.0))
    return {
        "beneficial_owner_count": len(by_person),
        "active_13d_filer_count": active_filers,
        "passive_13g_filer_count": passive_filers,
        "activist_intent_detected": activist,
        "beneficial_ownership_net_delta_fraction": net_delta,
        "largest_reported_beneficial_ownership_fraction": max(latest_percentages, default=0.0),
        "beneficial_ownership_context_score": context,
    }


def _13f_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _13f_filing_metadata(
    filing: Mapping[str, Any] | sqlite3.Row,
) -> dict[str, Any]:
    try:
        raw = filing["metadata_json"]
    except (IndexError, KeyError):
        raw = None

    if not raw:
        return {}

    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _13f_amendment_type(
    filing: Mapping[str, Any] | sqlite3.Row,
) -> str:
    metadata = _13f_filing_metadata(
        filing
    )

    value = str(
        metadata.get("amendment_type")
        or ""
    )

    return " ".join(
        value.upper().replace(
            "_",
            " ",
        ).split()
    )


def _effective_13f_accessions_for_period(
    conn: sqlite3.Connection,
    manager_cik: str,
    report_period: str,
    as_of: datetime,
) -> list[str]:
    filings = conn.execute(
        """
        SELECT
            accession,
            form_type,
            accepted_at,
            metadata_json
        FROM sec_intel_filings
        WHERE filer_cik = ?
          AND form_type LIKE '13F-HR%'
          AND report_period = ?
          AND accepted_at <= ?
        ORDER BY accepted_at ASC, accession ASC
        """,
        (
            manager_cik,
            report_period,
            isoformat_z(as_of),
        ),
    ).fetchall()

    effective: list[str] = []
    has_complete_base = False

    for filing in filings:
        metadata = _13f_filing_metadata(
            filing
        )

        form_type = str(
            filing["form_type"] or ""
        ).upper()

        is_amendment = (
            _13f_truthy(
                metadata.get("is_amendment")
            )
            or form_type.endswith("/A")
        )

        amendment_type = (
            _13f_amendment_type(
                filing
            )
        )

        accession = str(
            filing["accession"]
        )

        if not is_amendment:
            # Een latere niet-amendmentfiling binnen
            # hetzelfde kwartaal wordt de nieuwe basis.
            effective = [accession]
            has_complete_base = True
            continue

        if amendment_type == "RESTATEMENT":
            # Restatement vervangt alle eerdere
            # holdings voor dit kwartaal.
            effective = [accession]
            has_complete_base = True
            continue

        if amendment_type == "NEW HOLDINGS":
            # Alleen toevoegen wanneer er een complete
            # oorspronkelijke basis of restatement is.
            if has_complete_base:
                effective.append(accession)
            continue

        # Onbekend amendmenttype: fail-closed.
        # Niet gebruiken als volledige snapshot.

    return effective


def _latest_two_manager_filings(
    conn: sqlite3.Connection,
    manager_cik: str,
    as_of: datetime,
) -> list[dict[str, Any]]:
    periods = conn.execute(
        """
        SELECT
            report_period,
            MAX(accepted_at) AS accepted_at
        FROM sec_intel_filings
        WHERE filer_cik = ?
          AND form_type LIKE '13F-HR%'
          AND accepted_at <= ?
          AND report_period IS NOT NULL
          AND TRIM(report_period) <> ''
        GROUP BY report_period
        ORDER BY report_period DESC
        LIMIT 8
        """,
        (
            manager_cik,
            isoformat_z(as_of),
        ),
    ).fetchall()

    result: list[dict[str, Any]] = []

    for period in periods:
        report_period = str(
            period["report_period"]
        )

        accessions = (
            _effective_13f_accessions_for_period(
                conn,
                manager_cik,
                report_period,
                as_of,
            )
        )

        if not accessions:
            continue

        result.append(
            {
                "report_period": report_period,
                "accepted_at": str(
                    period["accepted_at"]
                ),
                "accessions": accessions,
            }
        )

        if len(result) == 2:
            break

    return result


def _position_for_filing_symbol(
    conn: sqlite3.Connection,
    accession: str,
    symbol: str,
    instrument_type: str = "EQUITY",
) -> sqlite3.Row | None:
    normalized_instrument = (
        str(instrument_type or "EQUITY")
        .strip()
        .upper()
    )

    row = conn.execute(
        """
        SELECT
            SUM(COALESCE(shares, 0)) AS shares,
            SUM(COALESCE(value_usd, 0)) AS value_usd,
            COUNT(*) AS row_count
        FROM sec_intel_events
        WHERE accession = ?
          AND event_type = '13f_position_snapshot'
          AND symbol = ?
          AND UPPER(
                COALESCE(
                    NULLIF(
                        TRIM(
                            json_extract(
                                payload_json,
                                '$.put_call'
                            )
                        ),
                        ''
                    ),
                    'EQUITY'
                )
              ) = ?
        """,
        (
            accession,
            symbol,
            normalized_instrument,
        ),
    ).fetchone()

    if (
        row is None
        or int(row["row_count"] or 0) == 0
    ):
        return None

    return row


def _position_for_manager_period_symbol(
    conn: sqlite3.Connection,
    manager_cik: str,
    report_period: str,
    symbol: str,
    as_of: datetime,
    instrument_type: str = "EQUITY",
) -> dict[str, float] | None:
    accessions = (
        _effective_13f_accessions_for_period(
            conn,
            manager_cik,
            report_period,
            as_of,
        )
    )

    if not accessions:
        return None

    total_shares = 0.0
    total_value = 0.0
    found = False

    for accession in accessions:
        position = (
            _position_for_filing_symbol(
                conn,
                accession,
                symbol,
                instrument_type,
            )
        )

        if position is None:
            continue

        found = True
        total_shares += (
            safe_float(position["shares"])
            or 0.0
        )
        total_value += (
            safe_float(position["value_usd"])
            or 0.0
        )

    if not found:
        return None

    return {
        "shares": total_shares,
        "value_usd": total_value,
    }


def _13f_filing_age_days(
    as_of: datetime,
    accepted_at: str,
) -> float:
    text = str(
        accepted_at or ""
    ).strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(
        text
    )

    reference = as_of

    if (
        parsed.tzinfo is None
        and reference.tzinfo is not None
    ):
        parsed = parsed.replace(
            tzinfo=reference.tzinfo
        )

    elif (
        parsed.tzinfo is not None
        and reference.tzinfo is None
    ):
        reference = reference.replace(
            tzinfo=parsed.tzinfo
        )

    return max(
        0.0,
        (
            reference - parsed
        ).total_seconds()
        / 86_400.0,
    )


def _13f_freshness_weight(
    age_days: float,
) -> float:
    # De eerste 45 dagen na publicatie krijgen
    # geen extra decay. Daarna: halfwaardetijd
    # van 180 dagen, met een ondergrens van 5%.
    decayed = math.exp(
        -math.log(2.0)
        * max(0.0, age_days - 45.0)
        / 180.0
    )

    return max(
        0.05,
        min(1.0, decayed),
    )


def _compute_13f_features(
    conn: sqlite3.Connection,
    symbol: str,
    events: Sequence[dict[str, Any]],
    as_of: datetime,
) -> dict[str, Any]:
    managers = sorted(
        {
            str(event.get("person_cik"))
            for event in events
            if (
                event["event_type"]
                == "13f_position_snapshot"
                and event.get("person_cik")
            )
        }
    )

    relevant_manager_count = 0
    equity_manager_count = 0

    new_positions = 0
    increases = 0
    decreases = 0
    closed = 0
    unchanged = 0

    net_share_change = 0.0
    comparable = 0

    weighted_equity_direction = 0.0
    equity_weight_total = 0.0

    reported_call_value = 0.0
    reported_put_value = 0.0
    call_manager_count = 0
    put_manager_count = 0

    weighted_option_direction = 0.0
    option_weight_total = 0.0

    filing_ages: list[float] = []
    freshness_weights: list[float] = []

    for manager in managers:
        periods = _latest_two_manager_filings(
            conn,
            manager,
            as_of,
        )

        if not periods:
            continue

        current_period = periods[0]

        current_report_period = str(
            current_period["report_period"]
        )

        age_days = _13f_filing_age_days(
            as_of,
            current_period["accepted_at"],
        )

        freshness = _13f_freshness_weight(
            age_days
        )

        # Huidige gewone aandelenpositie.
        current = (
            _position_for_manager_period_symbol(
                conn,
                manager,
                current_report_period,
                symbol,
                as_of,
                "EQUITY",
            )
        )

        # Optieposities blijven volledig gescheiden
        # van gewone aandelen.
        current_call = (
            _position_for_manager_period_symbol(
                conn,
                manager,
                current_report_period,
                symbol,
                as_of,
                "CALL",
            )
        )

        current_put = (
            _position_for_manager_period_symbol(
                conn,
                manager,
                current_report_period,
                symbol,
                as_of,
                "PUT",
            )
        )

        call_value = (
            current_call["value_usd"]
            if current_call
            else 0.0
        )

        put_value = (
            current_put["value_usd"]
            if current_put
            else 0.0
        )

        reported_call_value += call_value
        reported_put_value += put_value

        if call_value > 0:
            call_manager_count += 1

        if put_value > 0:
            put_manager_count += 1

        option_total = (
            call_value + put_value
        )

        if option_total > 0:
            option_direction = (
                call_value - put_value
            ) / option_total

            weighted_option_direction += (
                option_direction
                * freshness
            )

            option_weight_total += freshness

        # Met slechts één kwartaal kan de manager
        # wel actuele exposure hebben, maar nog geen
        # kwartaalvergelijking leveren.
        if len(periods) < 2:
            manager_is_relevant = any(
                (
                    current is not None,
                    current_call is not None,
                    current_put is not None,
                )
            )

            if manager_is_relevant:
                relevant_manager_count += 1
                filing_ages.append(age_days)
                freshness_weights.append(
                    freshness
                )

            continue

        previous_period = periods[1]

        previous = (
            _position_for_manager_period_symbol(
                conn,
                manager,
                str(
                    previous_period[
                        "report_period"
                    ]
                ),
                symbol,
                as_of,
                "EQUITY",
            )
        )

        # Een manager is ook relevant wanneer een
        # eerdere equitypositie nu volledig gesloten is.
        manager_is_relevant = any(
            (
                current is not None,
                previous is not None,
                current_call is not None,
                current_put is not None,
            )
        )

        if manager_is_relevant:
            relevant_manager_count += 1
            filing_ages.append(age_days)
            freshness_weights.append(
                freshness
            )

        # Een option-only manager heeft geen gewone
        # ownershipvergelijking.
        if (
            current is None
            and previous is None
        ):
            continue

        comparable += 1
        equity_manager_count += 1

        current_shares = (
            current["shares"]
            if current
            else None
        )

        previous_shares = (
            previous["shares"]
            if previous
            else None
        )

        direction = 0.0

        if (
            current is not None
            and previous is None
        ):
            new_positions += 1
            direction = 1.0

            net_share_change += (
                current_shares or 0.0
            )

        elif (
            current is None
            and previous is not None
        ):
            closed += 1
            direction = -1.0

            net_share_change -= (
                previous_shares or 0.0
            )

        else:
            delta = (
                (current_shares or 0.0)
                - (previous_shares or 0.0)
            )

            net_share_change += delta

            tolerance = max(
                1.0,
                abs(
                    previous_shares or 0.0
                )
                * 0.001,
            )

            if delta > tolerance:
                increases += 1

                relative_change = (
                    delta
                    / max(
                        abs(
                            previous_shares
                            or 0.0
                        ),
                        1.0,
                    )
                )

                direction = math.tanh(
                    relative_change * 2.0
                )

            elif delta < -tolerance:
                decreases += 1

                relative_change = (
                    delta
                    / max(
                        abs(
                            previous_shares
                            or 0.0
                        ),
                        1.0,
                    )
                )

                direction = math.tanh(
                    relative_change * 2.0
                )

            else:
                unchanged += 1
                direction = 0.0

        weighted_equity_direction += (
            direction * freshness
        )

        equity_weight_total += freshness

    if equity_weight_total > 0:
        equity_context = (
            weighted_equity_direction
            / equity_weight_total
        ) * 0.60
    else:
        equity_context = 0.0

    equity_context = max(
        -0.60,
        min(
            0.60,
            equity_context,
        ),
    )

    if option_weight_total > 0:
        option_context = (
            weighted_option_direction
            / option_weight_total
        ) * 0.35
    else:
        option_context = 0.0

    option_context = max(
        -0.35,
        min(
            0.35,
            option_context,
        ),
    )

    average_age = (
        sum(filing_ages)
        / len(filing_ages)
        if filing_ages
        else 0.0
    )

    average_freshness = (
        sum(freshness_weights)
        / len(freshness_weights)
        if freshness_weights
        else 0.0
    )

    fresh_managers = sum(
        1
        for age in filing_ages
        if age <= 180.0
    )

    stale_managers = sum(
        1
        for age in filing_ages
        if age > 180.0
    )

    return {
        "institutional_manager_count": (
            relevant_manager_count
        ),
        "institutional_equity_manager_count": (
            equity_manager_count
        ),
        "institutional_new_position_count": (
            new_positions
        ),
        "institutional_increase_count": increases,
        "institutional_decrease_count": decreases,
        "institutional_closed_position_count": (
            closed
        ),
        "institutional_unchanged_count": unchanged,
        "institutional_net_share_change": (
            net_share_change
        ),
        "institutional_comparable_manager_count": (
            comparable
        ),
        "institutional_delayed_context_score": (
            equity_context
        ),
        "institutional_reported_call_value_usd": (
            reported_call_value
        ),
        "institutional_reported_put_value_usd": (
            reported_put_value
        ),
        "institutional_call_manager_count": (
            call_manager_count
        ),
        "institutional_put_manager_count": (
            put_manager_count
        ),
        "institutional_options_context_score": (
            option_context
        ),
        "institutional_average_filing_age_days": (
            average_age
        ),
        "institutional_average_freshness_weight": (
            average_freshness
        ),
        "institutional_fresh_manager_count": (
            fresh_managers
        ),
        "institutional_stale_manager_count": (
            stale_managers
        ),
        "institutional_context_is_delayed": True,
    }




def _compute_8k_features(events: Sequence[dict[str, Any]], as_of: datetime) -> dict[str, Any]:
    items = [event for event in events if event["event_type"] == "8k_item"]
    recent30 = _window(items, as_of, 30)
    recent7 = _window(items, as_of, 7)
    weighted = sum(
        float(event.get("direction_score") or 0.0)
        * float(event.get("materiality") or 0.0)
        * float(event.get("confidence") or 0.0)
        for event in recent30
    )
    score = math.tanh(weighted)
    categories = sorted({event["payload"].get("category") for event in recent30 if event["payload"].get("category")})
    material = [event for event in recent30 if float(event.get("materiality") or 0.0) >= 0.70]
    severe_negative = any(
        float(event.get("direction_score") or 0.0) <= -0.75
        and float(event.get("materiality") or 0.0) >= 0.80
        for event in recent30
    )
    return {
        "eight_k_item_count_7d": len(recent7),
        "eight_k_item_count_30d": len(recent30),
        "eight_k_material_item_count_30d": len(material),
        "eight_k_categories_30d": categories,
        "eight_k_context_score": score,
        "eight_k_severe_negative_risk": severe_negative,
    }


def compute_symbol_features(
    conn: sqlite3.Connection,
    symbol: str,
    as_of: datetime | str,
    accepted_lookback_days: int = 550,
) -> dict[str, Any]:
    """Build a causal feature snapshot for one symbol."""
    ensure_schema(conn)
    cutoff = parse_datetime(as_of)
    clean_symbol = _clean_symbol(symbol)
    if not clean_symbol:
        raise ValueError("symbol is required")
    events = load_causal_events(conn, clean_symbol, cutoff, accepted_lookback_days)
    features: dict[str, Any] = {
        "symbol": clean_symbol,
        "as_of": isoformat_z(cutoff),
        "parser_version": PARSER_VERSION,
        "causal_event_count": len(events),
    }
    features.update(_compute_insider_features(events, cutoff))
    features.update(_compute_form144_features(events, cutoff))
    features.update(_compute_13dg_features(events, cutoff))
    features.update(_compute_13f_features(conn, clean_symbol, events, cutoff))
    features.update(_compute_8k_features(events, cutoff))

    components = {
        "insider": float(features.get("net_insider_conviction") or 0.0),
        "beneficial_ownership": float(features.get("beneficial_ownership_context_score") or 0.0),
        "institutional_delayed": float(features.get("institutional_delayed_context_score") or 0.0),
        "current_events": float(features.get("eight_k_context_score") or 0.0),
        "planned_sales": float(features.get("form144_unconfirmed_sale_pressure") or 0.0),
    }
    overlay_raw = (
        0.55 * components["insider"]
        + 0.15 * components["beneficial_ownership"]
        + 0.08 * components["institutional_delayed"]
        + 0.17 * components["current_events"]
        + 0.05 * components["planned_sales"]
    )
    if features.get("eight_k_severe_negative_risk"):
        overlay_raw = min(overlay_raw, -0.35)
    features["overlay_components"] = components
    features["sec_intelligence_score"] = clamp(overlay_raw, -1.0, 1.0)
    features["standalone_entry_allowed"] = False
    features["authority"] = "RANKING_OVERLAY_ONLY"
    return features


def apply_ranking_overlay(
    base_score: float,
    features: Mapping[str, Any],
    base_signal_authorized: bool,
    max_abs_points: float = 4.0,
) -> dict[str, Any]:
    """Apply bounded SEC context without creating authority.

    `base_signal_authorized=False` remains false even if SEC evidence is strongly
    positive. This function only changes ranking among already-valid candidates.
    """
    sec_score = clamp(float(features.get("sec_intelligence_score") or 0.0), -1.0, 1.0)
    overlay_points = sec_score * max_abs_points
    final_score = float(base_score) + overlay_points
    return {
        "base_score": float(base_score),
        "sec_intelligence_score": sec_score,
        "sec_overlay_points": overlay_points,
        "final_rank_score": final_score,
        "base_signal_authorized": bool(base_signal_authorized),
        "entry_authorized": bool(base_signal_authorized),
        "authority_rule": "SEC overlay cannot create an entry; it only ranks an existing valid signal.",
        "severe_negative_event_risk": bool(features.get("eight_k_severe_negative_risk")),
    }


def attribute_mover(
    conn: sqlite3.Connection,
    symbol: str,
    move_time: datetime | str,
    lookback_days: int = 3,
    minimum_materiality: float = 0.50,
) -> dict[str, Any]:
    """Return economically plausible SEC explanations available before a move.

    This replaces the weak rule "any SEC event existed in the preceding three days".
    Routine holdings, derivative grants and Item 9.01-only filings are excluded.
    """
    cutoff = parse_datetime(move_time)
    events = load_causal_events(conn, symbol, cutoff, accepted_lookback_days=max(10, lookback_days + 2))
    start = cutoff - timedelta(days=lookback_days)
    candidates: list[dict[str, Any]] = []
    for event in events:
        if not (start <= event["accepted_at"] <= cutoff):
            continue
        if float(event.get("materiality") or 0.0) < minimum_materiality:
            continue
        eligible = False
        if event["event_type"] == "8k_item":
            eligible = bool(event["payload"].get("attribution_eligible"))
        elif event["event_type"] == "insider_transaction":
            eligible = (
                event.get("transaction_code") in OPEN_MARKET_CODES
                and not event.get("is_derivative")
                and float(event.get("value_usd") or 0.0) >= 100_000.0
            )
        elif event["event_type"] == "schedule_13d_13g_snapshot":
            eligible = bool(event["payload"].get("active_ownership_filing"))
        elif event["event_type"] == "form144_proposed_sale":
            eligible = float(event.get("value_usd") or 0.0) >= 500_000.0
        if not eligible:
            continue
        age_hours = max(0.0, (cutoff - event["accepted_at"]).total_seconds() / 3600.0)
        recency = math.exp(-age_hours / 72.0)
        attribution_score = (
            float(event.get("materiality") or 0.0)
            * float(event.get("confidence") or 0.0)
            * recency
            * (1.0 + 0.35 * abs(float(event.get("direction_score") or 0.0)))
        )
        candidates.append(
            {
                "accession": event["accession"],
                "event_type": event["event_type"],
                "accepted_at": isoformat_z(event["accepted_at"]),
                "item_number": event.get("item_number"),
                "category": event["payload"].get("category"),
                "transaction_code": event.get("transaction_code"),
                "person_name": event.get("person_name"),
                "value_usd": event.get("value_usd"),
                "materiality": event.get("materiality"),
                "direction_score": event.get("direction_score"),
                "attribution_score": attribution_score,
            }
        )
    candidates.sort(key=lambda item: item["attribution_score"], reverse=True)
    return {
        "symbol": _clean_symbol(symbol),
        "move_time": isoformat_z(cutoff),
        "lookback_days": lookback_days,
        "has_plausible_sec_catalyst": bool(candidates),
        "best_candidate": candidates[0] if candidates else None,
        "candidates": candidates[:10],
    }


def ingest_jsonl(
    conn: sqlite3.Connection,
    input_path: str | Path,
    content_root: str | Path | None = None,
    security_resolver: SecurityResolver | None = None,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    root = Path(content_root).resolve() if content_root else None
    ingested = skipped = failed = events_total = 0
    errors: list[dict[str, Any]] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                envelope = FilingEnvelope.from_mapping(record)
                content = record.get("content") or record.get("raw_content")
                if content is None:
                    path_raw = record.get("content_path") or record.get("source_path") or record.get("primary_document")
                    if not path_raw:
                        raise ValueError("record has no content or content path")
                    path = Path(path_raw)
                    if not path.is_absolute() and root:
                        path = root / path
                    content = path.read_text(encoding="utf-8", errors="replace")
                count = ingest_filing(conn, envelope, str(content), security_resolver)
                if count:
                    ingested += 1
                    events_total += count
                else:
                    skipped += 1
            except Exception as exc:  # noqa: BLE001 - batch ingestion must report bad rows
                failed += 1
                errors.append({"line": line_number, "error": str(exc)})
                if not continue_on_error:
                    raise
    return {
        "filings_ingested": ingested,
        "filings_skipped_unsupported": skipped,
        "filings_failed": failed,
        "events_inserted": events_total,
        "errors": errors[:100],
    }


def _self_test() -> dict[str, Any]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    form4 = """<?xml version="1.0"?>
    <ownershipDocument>
      <documentType>4</documentType><periodOfReport>2026-08-01</periodOfReport>
      <issuer><issuerCik>0000320193</issuerCik><issuerName>Example Corp</issuerName><issuerTradingSymbol>EXM</issuerTradingSymbol></issuer>
      <reportingOwner>
        <reportingOwnerId><rptOwnerCik>0000001001</rptOwnerCik><rptOwnerName>Jane Doe</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isDirector>0</isDirector><isOfficer>1</isOfficer><officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable><nonDerivativeTransaction>
        <securityTitle><value>Common Stock</value></securityTitle>
        <transactionDate><value>2026-08-01</value></transactionDate>
        <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>P</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
        <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>50</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
        <postTransactionAmounts><sharesOwnedFollowingTransaction><value>5000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
        <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
      </nonDerivativeTransaction></nonDerivativeTable>
    </ownershipDocument>"""
    form144 = """<?xml version="1.0"?>
    <edgarSubmission><formData><issuerInfo><issuerCik>0000320193</issuerCik><issuerName>Example Corp</issuerName></issuerInfo>
      <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Jane Doe</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
      <securitiesInformation><securitiesClassTitle>Common Stock</securitiesClassTitle><unitsToBeSold>2500</unitsToBeSold>
      <aggregateMarketValue>125000</aggregateMarketValue><unitsOutstanding>1000000</unitsOutstanding><approxSaleDate>2026-08-10</approxSaleDate></securitiesInformation>
    </formData></edgarSubmission>"""
    eight_k = """<html><body><h1>Item 2.04</h1><p>The company received a notice of default and acceleration of debt obligations.</p><h1>Item 9.01</h1></body></html>"""
    ingest_filing(
        conn,
        FilingEnvelope("0001", "4", parse_datetime("2026-08-02T12:00:00Z"), symbol="EXM", issuer_cik="0000320193"),
        form4,
    )
    ingest_filing(
        conn,
        FilingEnvelope("0002", "144", parse_datetime("2026-08-03T12:00:00Z"), symbol="EXM", issuer_cik="0000320193", filer_cik="0000001001"),
        form144,
    )
    ingest_filing(
        conn,
        FilingEnvelope("0003", "8-K", parse_datetime("2026-08-04T12:00:00Z"), symbol="EXM", issuer_cik="0000320193", metadata={"items": ["2.04", "9.01"]}),
        eight_k,
    )
    features = compute_symbol_features(conn, "EXM", "2026-08-05T12:00:00Z")
    assert features["insider_buy_count_7d"] == 1
    assert features["net_insider_conviction"] > 0
    assert features["form144_notice_count_90d"] == 1
    assert features["eight_k_severe_negative_risk"] is True
    assert features["standalone_entry_allowed"] is False
    overlay = apply_ranking_overlay(70.0, features, base_signal_authorized=False)
    assert overlay["entry_authorized"] is False
    attribution = attribute_mover(conn, "EXM", "2026-08-05T12:00:00Z")
    assert attribution["has_plausible_sec_catalyst"] is True
    return {"status": "PASS", "features": features, "overlay": overlay, "attribution": attribution}


def _load_resolver_csv(path: str | Path | None) -> SecurityResolver | None:
    if not path:
        return None
    mapping_by_cusip: dict[str, str] = {}
    mapping_by_cik: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {str(key).strip().lower(): (value or "").strip() for key, value in raw_row.items()}
            symbol = _clean_symbol(row.get("symbol") or row.get("ticker"))
            if not symbol:
                continue
            cusip = (row.get("cusip") or "").upper()
            cik = _clean_cik(row.get("cik") or row.get("issuer_cik"))
            if cusip:
                mapping_by_cusip[cusip] = symbol
            if cik:
                mapping_by_cik[cik] = symbol

    def resolver(cusip: str | None, issuer_cik: str | None) -> str | None:
        if cusip and cusip.upper() in mapping_by_cusip:
            return mapping_by_cusip[cusip.upper()]
        clean_cik = _clean_cik(issuer_cik)
        return mapping_by_cik.get(clean_cik or "")

    return resolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create/upgrade the intelligence SQLite schema")
    init_parser.add_argument("--db", required=True)

    ingest_parser = subparsers.add_parser("ingest-jsonl", help="Ingest existing filing records from JSONL")
    ingest_parser.add_argument("--db", required=True)
    ingest_parser.add_argument("--input", required=True)
    ingest_parser.add_argument("--content-root")
    ingest_parser.add_argument("--security-map", help="CSV with symbol plus CUSIP and/or issuer CIK")
    ingest_parser.add_argument("--continue-on-error", action="store_true")

    feature_parser = subparsers.add_parser("features", help="Compute one causal symbol feature snapshot")
    feature_parser.add_argument("--db", required=True)
    feature_parser.add_argument("--symbol", required=True)
    feature_parser.add_argument("--as-of", required=True, help="ISO timestamp, preferably with timezone")
    feature_parser.add_argument("--lookback-days", type=int, default=550)

    overlay_parser = subparsers.add_parser("overlay", help="Compute bounded ranking overlay")
    overlay_parser.add_argument("--db", required=True)
    overlay_parser.add_argument("--symbol", required=True)
    overlay_parser.add_argument("--as-of", required=True)
    overlay_parser.add_argument("--base-score", type=float, required=True)
    overlay_parser.add_argument("--base-authorized", action="store_true")
    overlay_parser.add_argument("--max-points", type=float, default=4.0)

    attribute_parser = subparsers.add_parser("attribute", help="Find plausible SEC catalysts before a move")
    attribute_parser.add_argument("--db", required=True)
    attribute_parser.add_argument("--symbol", required=True)
    attribute_parser.add_argument("--move-time", required=True)
    attribute_parser.add_argument("--lookback-days", type=int, default=3)
    attribute_parser.add_argument("--minimum-materiality", type=float, default=0.50)

    subparsers.add_parser("self-test", help="Run deterministic parser and feature smoke tests")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "self-test":
        print(json.dumps(_self_test(), indent=2, sort_keys=True, default=str))
        return 0
    conn = connect_database(args.db)
    try:
        ensure_schema(conn)
        if args.command == "init":
            print(json.dumps({"status": "OK", "db": str(args.db), "parser_version": PARSER_VERSION}, indent=2))
        elif args.command == "ingest-jsonl":
            resolver = _load_resolver_csv(args.security_map)
            result = ingest_jsonl(
                conn,
                args.input,
                content_root=args.content_root,
                security_resolver=resolver,
                continue_on_error=args.continue_on_error,
            )
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        elif args.command == "features":
            result = compute_symbol_features(conn, args.symbol, args.as_of, args.lookback_days)
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        elif args.command == "overlay":
            features = compute_symbol_features(conn, args.symbol, args.as_of)
            result = apply_ranking_overlay(
                args.base_score,
                features,
                base_signal_authorized=args.base_authorized,
                max_abs_points=args.max_points,
            )
            print(json.dumps({"features": features, "overlay": result}, indent=2, sort_keys=True, default=str))
        elif args.command == "attribute":
            result = attribute_mover(
                conn,
                args.symbol,
                args.move_time,
                lookback_days=args.lookback_days,
                minimum_materiality=args.minimum_materiality,
            )
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SecIntelligenceError, ValueError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
