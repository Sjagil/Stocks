from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POSITIVE = {"beat", "beats", "growth", "raised", "strong", "record", "improved", "buyback", "acquisition", "demand"}
NEGATIVE = {"miss", "misses", "lower", "cut", "weak", "decline", "lawsuit", "investigation", "departure", "risk"}
UNCERTAIN = {"may", "might", "could", "uncertain", "approximately", "expects", "subject to", "believes"}
FORWARD = {"guidance", "outlook", "expects", "forecast", "next quarter", "full year", "forward-looking"}
SUPPORTED_SEC_FORMS = {"10-K", "10-Q", "8-K", "FORM 4", "FORM 5", "13D", "13G", "13F", "S-1", "DEF 14A"}


@dataclass(frozen=True)
class NewsIntelligenceEngine:
    known_entities: Mapping[str, Iterable[str]] | None = None

    def analyze(self, text: str, *, history: Iterable[str] = ()) -> dict[str, Any]:
        normalized = " ".join(str(text).split())
        if not normalized:
            raise ValueError("news text is required")
        lower = normalized.lower()
        tokens = re.findall(r"[a-z0-9%.-]+", lower)
        positive = sum(token in POSITIVE for token in tokens)
        negative = sum(token in NEGATIVE for token in tokens)
        sentiment = (positive - negative) / max(positive + negative, 1)
        revenue_surprise = _percentage_after(lower, r"(?:revenue|sales)[^.%]{0,40}(?:beat|beats|above)[^0-9]{0,10}")
        guidance = _direction(lower, "guid")
        margin = _direction(lower, "margin")
        demand = _direction(lower, "demand")
        entities = _entities(normalized, self.known_entities or {})
        events = _events(lower)
        topics = _topics(lower)
        novelty = _novelty(normalized, history)
        uncertainty = sum(phrase in lower for phrase in UNCERTAIN) / max(len(UNCERTAIN), 1)
        forward_looking = sum(phrase in lower for phrase in FORWARD) / max(len(FORWARD), 1)
        relevance = float(np.clip(0.25 + 0.15 * len(entities) + 0.10 * len(events) + 0.05 * len(topics), 0, 1))
        component_values = [sentiment * 0.25]
        if revenue_surprise is not None:
            component_values.append(float(np.clip(revenue_surprise * 2, -0.5, 0.5)))
        component_values.extend(value * 0.2 for value in (guidance, margin, demand) if value != 0)
        impact = float(np.clip(sum(component_values), -1, 1))
        return {
            "entities": entities,
            "events": events,
            "topics": topics,
            "sentiment": float(sentiment),
            "revenue_surprise": revenue_surprise,
            "guidance": _label(guidance),
            "margin_outlook": _label(margin),
            "demand_commentary": _label(demand),
            "novelty": novelty,
            "relevance": relevance,
            "uncertainty": float(uncertainty),
            "forward_looking_language": float(forward_looking),
            "overall_impact": impact,
            "simple_positive_negative_only": False,
            "research_only": True,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


class SecFilingIntelligenceEngine:
    def analyze(
        self,
        form: str,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        previous_text: str | None = None,
    ) -> dict[str, Any]:
        canonical_form = str(form).strip().upper().replace("FORM ", "FORM ")
        if canonical_form in {"4", "5"}:
            canonical_form = f"FORM {canonical_form}"
        if canonical_form not in SUPPORTED_SEC_FORMS:
            raise ValueError("unsupported SEC form")
        lower = " ".join(str(text).lower().split())
        metadata = dict(metadata or {})
        features = {
            "insider_buying": _count(lower, ("purchase", "acquired", "open market buy")),
            "insider_selling": _count(lower, ("sale", "disposed", "sold")),
            "ownership_changes": _count(lower, ("beneficial ownership", "percent of class", "voting power")),
            "institutional_accumulation": _count(lower, ("increased position", "additional shares", "accumulation")),
            "guidance_changes": _count(lower, ("revised guidance", "updated outlook", "withdraws guidance")),
            "capital_structure": _count(lower, ("debt issuance", "equity offering", "convertible", "dilution")),
            "buybacks": _count(lower, ("share repurchase", "stock repurchase", "buyback")),
            "management_changes": _count(lower, ("chief executive officer", "chief financial officer", "resignation", "appointed")),
            "mergers_acquisitions": _count(lower, ("merger", "acquisition", "tender offer", "business combination")),
            "litigation": _count(lower, ("litigation", "lawsuit", "legal proceeding", "investigation")),
        }
        risk_factor_change = _text_change(previous_text, text) if previous_text is not None else None
        transaction_code = str(metadata.get("transaction_code", "")).upper()
        if canonical_form in {"FORM 4", "FORM 5"}:
            features["insider_buying"] += int(transaction_code == "P")
            features["insider_selling"] += int(transaction_code == "S")
        signal = (
            0.25 * features["insider_buying"]
            - 0.20 * features["insider_selling"]
            + 0.15 * features["institutional_accumulation"]
            + 0.10 * features["buybacks"]
            - 0.10 * features["litigation"]
            - 0.10 * features["capital_structure"]
        )
        return {
            "form": canonical_form,
            "features": features,
            "risk_factor_change": risk_factor_change,
            "sec_signal": float(np.clip(signal, -1, 1)),
            "accepted_at": metadata.get("accepted_at"),
            "issuer": metadata.get("issuer"),
            "research_only": True,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


def event_study(
    asset_returns: Any,
    market_returns: Any,
    *,
    event_position: int,
    sector_returns: Any | None = None,
    windows: Iterable[int] = (-20, -5, 0, 1, 5, 20),
    estimation_window: int = 60,
) -> dict[int, dict[str, float | None]]:
    asset = np.asarray(asset_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float)
    if len(asset) != len(market) or not 0 <= event_position < len(asset):
        raise ValueError("event-study arrays or position are invalid")
    start = max(0, event_position - estimation_window - 1)
    end = max(start, event_position - 1)
    if end - start < 20:
        raise ValueError("insufficient pre-event estimation history")
    x = np.column_stack([np.ones(end - start), market[start:end]])
    alpha, beta = np.linalg.lstsq(x, asset[start:end], rcond=None)[0]
    abnormal = asset - (alpha + beta * market)
    if sector_returns is not None:
        sector = np.asarray(sector_returns, dtype=float)
        if len(sector) != len(asset):
            raise ValueError("sector returns must align")
        abnormal = abnormal - sector
    result: dict[int, dict[str, float | None]] = {}
    for offset in windows:
        position = event_position + int(offset)
        if position < 0 or position >= len(asset):
            result[int(offset)] = {"abnormal_return": None, "cumulative_abnormal_return": None}
            continue
        left, right = sorted((event_position, position))
        cumulative = float(abnormal[left : right + 1].sum())
        if position < event_position:
            cumulative = -cumulative
        result[int(offset)] = {
            "abnormal_return": float(abnormal[position]),
            "cumulative_abnormal_return": cumulative,
        }
    return result


def _entities(text: str, known: Mapping[str, Iterable[str]]) -> list[str]:
    upper = text.upper()
    found = {
        str(symbol).upper()
        for symbol, aliases in known.items()
        if any(str(alias).upper() in upper for alias in [symbol, *aliases])
    }
    found.update(re.findall(r"\b[A-Z]{2,5}\b", text))
    return sorted(found)


def _events(lower: str) -> list[str]:
    patterns = {
        "EARNINGS": ("earnings", "revenue estimates"),
        "GUIDANCE_CHANGE": ("guidance", "outlook"),
        "MANAGEMENT_CHANGE": ("ceo departure", "resignation", "appointed"),
        "M_AND_A": ("merger", "acquisition"),
        "REGULATORY": ("sec investigation", "antitrust", "regulator"),
        "FED_DECISION": ("federal reserve", "fed decision", "rate decision"),
    }
    return [event for event, phrases in patterns.items() if any(phrase in lower for phrase in phrases)]


def _topics(lower: str) -> list[str]:
    patterns = {
        "REVENUE": ("revenue", "sales"),
        "MARGINS": ("margin",),
        "DEMAND": ("demand", "orders"),
        "CAPITAL": ("buyback", "dividend", "debt"),
        "LEGAL": ("lawsuit", "litigation", "investigation"),
    }
    return [topic for topic, phrases in patterns.items() if any(phrase in lower for phrase in phrases)]


def _percentage_after(text: str, prefix: str) -> float | None:
    match = re.search(prefix + r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if match is None:
        match = re.search(
            r"(?:beat|beats|above)[^.%]{0,30}(?:revenue|sales)[^0-9]{0,30}"
            r"([0-9]+(?:\.[0-9]+)?)\s*%",
            text,
        )
    return float(match.group(1)) / 100 if match else None


def _direction(text: str, subject: str) -> int:
    if subject == "guid" and re.search(r"guid(?:ance|es|ed)[^.!?]{0,80}(?:lower|down|cut|weak)", text):
        return -1
    if subject == "margin" and (
        re.search(r"margin[^.!?]{0,40}(?:lower|down|declin|weak)", text)
        or re.search(r"(?:lower|down|declin|weak)[^.!?]{0,20}margin", text)
    ):
        return -1
    if subject == "demand" and re.search(r"(?:strong|record|improv|robust)[^.!?]{0,25}demand", text):
        return 1
    snippets = [text[max(0, match.start() - 50) : match.end() + 80] for match in re.finditer(subject, text)]
    score = sum(sum(word in snippet for word in POSITIVE) - sum(word in snippet for word in NEGATIVE) for snippet in snippets)
    return 1 if score > 0 else -1 if score < 0 else 0


def _label(value: int) -> str:
    return "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "NEUTRAL_OR_UNAVAILABLE"


def _novelty(text: str, history: Iterable[str]) -> float:
    documents = [str(item) for item in history if str(item).strip()]
    if not documents:
        return 1.0
    matrix = TfidfVectorizer(stop_words="english").fit_transform([*documents, text])
    similarity = cosine_similarity(matrix[-1], matrix[:-1]).max()
    return float(np.clip(1 - similarity, 0, 1))


def _count(text: str, phrases: Iterable[str]) -> int:
    return sum(text.count(phrase) for phrase in phrases)


def _text_change(previous: str, current: str) -> float:
    left = set(re.findall(r"[a-z]{3,}", previous.lower()))
    right = set(re.findall(r"[a-z]{3,}", current.lower()))
    union = left | right
    return float(1 - len(left & right) / len(union)) if union else 0.0
