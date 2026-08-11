from __future__ import annotations

from typing import Any, Mapping


def deterministic_analysis(
    snapshot: Mapping[str, Any],
    *,
    period: str,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    regime = snapshot["regime"]
    scores = snapshot["scores"]
    changes = _changes(scores, None if previous is None else previous.get("scores"))
    positive = _drivers(scores, positive=True)
    negative = _drivers(scores, positive=False)
    conflicts = _conflicts(scores)
    supported = _implications(snapshot, "POSITIVE")
    vulnerable = _implications(snapshot, "NEGATIVE")
    missing = sorted(
        {
            item
            for score in scores.values()
            for item in score.get("missing_inputs", ())
        }
    )
    stale = sorted(
        {
            item
            for score in scores.values()
            for item in score.get("stale_inputs", ())
        }
    )
    paragraphs = [
        (
            f"Het huidige macroregime is {regime['overall_macro_regime']} "
            f"met confidence {regime['confidence']:.0%}."
        ),
        _direction_sentence(scores),
        _risk_sentence(regime, conflicts),
        _breadth_sentence(scores),
    ]
    if missing or stale:
        paragraphs.append(
            "De analyse is beperkt door ontbrekende of stale inputs; "
            "ontbrekende data worden niet als neutraal bewijs behandeld."
        )
    return {
        "schema": "deterministic_macro_analyst_v1",
        "period": period,
        "as_of": snapshot["as_of"],
        "current_regime": regime["overall_macro_regime"],
        "directions": {
            key: regime[key]
            for key in (
                "growth_regime",
                "inflation_regime",
                "liquidity_regime",
                "credit_regime",
                "market_regime",
                "currency_regime",
                "commodity_regime",
            )
        },
        "changes_since_previous": changes,
        "confirming_signals": _confirmations(regime),
        "conflicting_signals": conflicts,
        "risks_increasing": [row["series_id"] for row in negative[:5]],
        "risks_decreasing": [row["series_id"] for row in positive[:5]],
        "top_positive_drivers": positive[:10],
        "top_negative_drivers": negative[:10],
        "relatively_supported": supported,
        "relatively_vulnerable": vulnerable,
        "watch_indicators": list(dict.fromkeys([*missing, *stale]))[:20],
        "missing_inputs": missing,
        "stale_inputs": stale,
        "confidence": regime["confidence"],
        "paragraphs_nl": paragraphs,
        "guaranteed_prediction": False,
        "order_intent": False,
        "automatic_authority_change": False,
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        f"# Macrorapport {analysis['period'].title()}",
        "",
        f"As-of: `{analysis['as_of']}`",
        "",
        *[str(paragraph) for paragraph in analysis["paragraphs_nl"]],
        "",
        "## Positieve drivers",
        "",
    ]
    lines.extend(
        f"- {row['series_id']}: {row['weighted_contribution']:.2f}"
        for row in analysis["top_positive_drivers"][:5]
    )
    lines.extend(["", "## Negatieve drivers", ""])
    lines.extend(
        f"- {row['series_id']}: {row['weighted_contribution']:.2f}"
        for row in analysis["top_negative_drivers"][:5]
    )
    lines.extend(
        [
            "",
            "Dit rapport is researchcontext, geen order of gegarandeerde voorspelling.",
            "",
        ]
    )
    return "\n".join(lines)


def _direction_sentence(scores: Mapping[str, Any]) -> str:
    def label(name: str) -> str:
        value = scores[name].get("value")
        if value is None:
            return "onbekend"
        return "positief" if value > 15 else "negatief" if value < -15 else "neutraal"

    return (
        f"Groei is {label('growth')}, inflatie-ontwikkeling is "
        f"{label('inflation')}, liquiditeit is {label('liquidity')} en "
        f"krediet is {label('credit')}."
    )


def _risk_sentence(regime: Mapping[str, Any], conflicts: list[str]) -> str:
    if conflicts:
        return (
            "Macro- en marktinputs spreken elkaar deels tegen. "
            "Het regime blijft daarom context en geen zelfstandige timingregel."
        )
    return (
        f"Marktbevestiging staat op {regime['market_regime']}; "
        "technische en fundamentele assetfilters blijven vereist."
    )


def _breadth_sentence(scores: Mapping[str, Any]) -> str:
    breadth = scores["breadth"].get("value")
    if breadth is None:
        return "Betrouwbare marktbreedte is niet beschikbaar."
    if breadth < -15:
        return (
            "De marktbreadth verslechtert en maakt brede indexsterkte "
            "minder betrouwbaar."
        )
    if breadth > 15:
        return "De marktbreadth bevestigt een relatief brede marktdeelname."
    return "De marktbreadth geeft geen duidelijke bevestiging."


def _changes(
    scores: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if previous is None:
        return []
    result = []
    for name, score in scores.items():
        current = score.get("value")
        prior = previous.get(name, {}).get("value")
        if current is None or prior is None:
            continue
        difference = float(current) - float(prior)
        if abs(difference) >= 5:
            result.append(
                {"score": name, "previous": prior, "current": current, "change": difference}
            )
    return sorted(result, key=lambda row: abs(row["change"]), reverse=True)


def _drivers(scores: Mapping[str, Any], *, positive: bool) -> list[dict[str, Any]]:
    key = "positive_contributions" if positive else "negative_contributions"
    rows = [dict(row) for score in scores.values() for row in score.get(key, ())]
    return sorted(
        rows,
        key=lambda row: row["weighted_contribution"],
        reverse=positive,
    )


def _conflicts(scores: Mapping[str, Any]) -> list[str]:
    pairs = (
        ("growth", "breadth"),
        ("liquidity", "credit"),
        ("risk_appetite", "financial_stress"),
    )
    result = []
    for left, right in pairs:
        a = scores[left].get("value")
        b = scores[right].get("value")
        if a is not None and b is not None and a * b < -225:
            result.append(f"{left.upper()}_VERSUS_{right.upper()}")
    return result


def _confirmations(regime: Mapping[str, Any]) -> list[str]:
    result = []
    if regime["growth_regime"] != "UNKNOWN" and regime["market_regime"] != "UNKNOWN":
        result.append(
            f"GROWTH_{regime['growth_regime']}_MARKET_{regime['market_regime']}"
        )
    if regime["liquidity_regime"] != "UNKNOWN" and regime["credit_regime"] != "UNKNOWN":
        result.append(
            f"LIQUIDITY_{regime['liquidity_regime']}_CREDIT_{regime['credit_regime']}"
        )
    return result


def _implications(snapshot: Mapping[str, Any], status: str) -> list[dict[str, Any]]:
    sections = snapshot["implications"]
    rows = [
        dict(value)
        for section in ("sectors_and_asset_classes", "regions")
        for value in sections[section].values()
        if value["macro_support"] == status
    ]
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)
