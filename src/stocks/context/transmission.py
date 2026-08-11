from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.market.context import load_market_context_map


CONFIG_PATH = Path("config/context/asset_transmission_v1.json")
UNIVERSE_PATH = Path("output/universe/instruments.parquet")


def build_asset_context(
    project_root: Path,
    *,
    symbols: Iterable[str] | None = None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    now = _utc(observed_at or datetime.now(UTC))
    config = _read_json(project_root / CONFIG_PATH)
    _validate_config(config)
    macro = _read_json(project_root / "output/macro/score.json")
    cot = _read_json(
        project_root / "output/market_context/cot-context.json"
    )
    news = _read_json(
        project_root
        / "output/notifications/market-intelligence-digest.json"
    )
    market = load_market_context_map(project_root)
    selected = _symbols(project_root, symbols, config)
    raw_macro_scores = macro.get("scores")
    macro_scores: dict[str, Any] = (
        {str(key): value for key, value in raw_macro_scores.items()}
        if isinstance(raw_macro_scores, dict)
        else {}
    )
    cot_map = {
        str(row.get("market_id")): row
        for row in cot.get("contexts", [])
        if isinstance(row, dict) and row.get("market_id")
    }
    news_map = _news_by_symbol(news)
    metadata = _metadata_map(project_root)
    contexts: list[dict[str, Any]] = []
    for symbol in selected:
        group_id, mapping_source = _transmission_group(
            symbol,
            config=config,
            metadata=metadata.get(symbol, {}),
        )
        group = config["groups"].get(group_id, config["groups"]["broad_equity"])
        sensitivities = group.get("sensitivities", {})
        if not isinstance(sensitivities, dict):
            sensitivities = {}
        macro_component = _macro_component(
            sensitivities, macro_scores
        )
        cot_component = _cot_component(
            cot_map.get(str(group.get("cot_market") or ""))
        )
        symbol_market = market.get(symbol, {})
        gex_component = _market_component(symbol_market.get("gex"), "GEX")
        flow_component = _market_component(
            symbol_market.get("orderflow"), "ORDERFLOW"
        )
        event_component = _event_component(
            news_map.get(symbol, []),
            global_event_risk=bool(news.get("event_risk_within_24h")),
        )
        components = {
            "macro": macro_component,
            "cot": cot_component,
            "gex": gex_component,
            "orderflow": flow_component,
        }
        bias_score, bias_confidence = _weighted_bias(components)
        contexts.append(
            {
                "symbol": symbol,
                "transmission_group": group_id,
                "transmission_mapping_source": mapping_source,
                "macro_sensitivities": group.get("sensitivities", {}),
                "cot_market": group.get("cot_market"),
                "components": components,
                "event_risk": event_component,
                "asset_bias_score": round(bias_score, 8),
                "asset_bias_confidence": round(bias_confidence, 8),
                "bias_classification": _classification(bias_score),
                "gex_applicable": gex_component["status"] not in {
                    "UNAVAILABLE",
                    "STALE_CONTEXT_BLOCKED",
                },
                "observed_orderflow_available": bool(
                    symbol_market.get("orderflow", {}).get(
                        "observed_aggressor_volume"
                    )
                ),
                "standalone_entry_authority": False,
                "strategy_authority": "NONE",
                "execution_authority": "NONE",
            }
        )
    payload = {
        "schema": "asset_specific_context_transmission_v1",
        "status": "GO" if contexts else "DATA_UNAVAILABLE",
        "generated_at": now.isoformat(),
        "config_version": config["version"],
        "context_count": len(contexts),
        "transmission_group_counts": _counts(
            row["transmission_group"] for row in contexts
        ),
        "mapping_source_counts": _counts(
            row["transmission_mapping_source"] for row in contexts
        ),
        "contexts": contexts,
        "architecture": "CONTEXT_TO_BIAS_TO_SETUP_TO_ENTRY_TO_RISK_EXIT",
        "limitations": [
            "COT is weekly asset-class context, never an intraday trigger",
            "GEX dealer direction is estimated, never observed",
            "bar-flow is not observed aggressor flow",
            "missing components reduce confidence instead of becoming neutral evidence",
        ],
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_json(
        project_root / "output/market_context/asset-context.json", payload
    )
    return payload


def _macro_component(
    sensitivities: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    contributions: list[dict[str, Any]] = []
    numerator = 0.0
    denominator = 0.0
    confidence_numerator = 0.0
    for family, raw_weight in sensitivities.items():
        weight = _number(raw_weight)
        score = scores.get(family)
        if not isinstance(score, dict) or score.get("value") is None:
            contributions.append(
                {
                    "family": family,
                    "sensitivity": weight,
                    "status": "UNAVAILABLE",
                }
            )
            continue
        value = _clamp(_number(score.get("value")) / 100.0, -1.0, 1.0)
        confidence = _clamp(_number(score.get("confidence")), 0.0, 1.0)
        contribution = weight * value
        numerator += contribution * confidence
        denominator += abs(weight) * confidence
        confidence_numerator += abs(weight) * confidence
        contributions.append(
            {
                "family": family,
                "sensitivity": weight,
                "normalized_score": round(value, 8),
                "confidence": round(confidence, 8),
                "weighted_contribution": round(contribution, 8),
                "status": score.get("status", "UNKNOWN"),
            }
        )
    total_weight = sum(abs(_number(value)) for value in sensitivities.values())
    confidence = confidence_numerator / total_weight if total_weight else 0.0
    return {
        "status": "AVAILABLE" if denominator else "UNAVAILABLE",
        "score": round(numerator / denominator, 8) if denominator else 0.0,
        "confidence": round(confidence, 8),
        "contributions": contributions,
        "standalone_entry_authority": False,
    }


def _cot_component(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "status": "UNAVAILABLE",
            "score": 0.0,
            "confidence": 0.0,
            "standalone_entry_authority": False,
        }
    status = str(row.get("status", "UNAVAILABLE"))
    return {
        "status": status,
        "market_id": row.get("market_id"),
        "report_date": row.get("report_date"),
        "available_at": row.get("available_at"),
        "score": (
            _clamp(_number(row.get("positioning_score")), -1.0, 1.0)
            if status == "CONTEXT_AVAILABLE"
            else 0.0
        ),
        "confidence": (
            _clamp(_number(row.get("confidence")), 0.0, 1.0)
            if status == "CONTEXT_AVAILABLE"
            else 0.0
        ),
        "standalone_entry_authority": False,
    }


def _market_component(row: Any, name: str) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("status") in {None, "UNAVAILABLE"}:
        return {
            "status": "UNAVAILABLE",
            "score": 0.0,
            "confidence": 0.0,
            "data_class": None,
            "standalone_entry_authority": False,
        }
    confidence = _clamp(_number(row.get("confidence")), 0.0, 1.0)
    ranking = _clamp(_number(row.get("ranking_score"), 0.5), 0.0, 1.0)
    status = str(row.get("status"))
    data_class = (
        "OBSERVED_TRADE_FLOW"
        if name == "ORDERFLOW" and row.get("observed_aggressor_volume")
        else "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW"
        if name == "ORDERFLOW"
        else "ESTIMATED_DEALER_GEX"
    )
    return {
        "status": status,
        "score": round(2.0 * (ranking - 0.5), 8),
        "confidence": round(confidence, 8),
        "data_class": data_class,
        "standalone_entry_authority": False,
    }


def _event_component(
    rows: list[dict[str, Any]], *, global_event_risk: bool
) -> dict[str, Any]:
    risk = 0.15 if global_event_risk else 0.0
    negative_high = False
    for row in rows:
        importance = str(row.get("importance", "LOW"))
        direction = str(row.get("direction", "MIXED_OR_UNCLEAR"))
        if importance == "HIGH":
            risk = max(risk, 0.8)
            negative_high = negative_high or direction == "NEGATIVE_INFERENCE"
        elif importance == "MEDIUM":
            risk = max(risk, 0.4)
    return {
        "risk_score": round(risk, 8),
        "negative_high_impact": negative_high,
        "news_count": len(rows),
        "blocks_new_entry": negative_high,
    }


def _weighted_bias(
    components: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    weights = {"macro": 0.50, "cot": 0.20, "gex": 0.15, "orderflow": 0.15}
    numerator = 0.0
    denominator = 0.0
    confidence = 0.0
    for name, weight in weights.items():
        row = components[name]
        component_confidence = _clamp(_number(row.get("confidence")), 0.0, 1.0)
        if component_confidence <= 0:
            continue
        effective = weight * component_confidence
        numerator += _clamp(_number(row.get("score")), -1.0, 1.0) * effective
        denominator += effective
        confidence += effective
    return (
        _clamp(numerator / denominator, -1.0, 1.0) if denominator else 0.0,
        _clamp(confidence / sum(weights.values()), 0.0, 1.0),
    )


def _news_by_symbol(news: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in news.get("important_news", []):
        if not isinstance(row, dict):
            continue
        for symbol in row.get("symbols", []):
            result.setdefault(str(symbol).upper(), []).append(row)
    return result


def _symbols(
    project_root: Path,
    supplied: Iterable[str] | None,
    config: dict[str, Any],
) -> list[str]:
    if supplied is not None:
        values = [str(value).strip().upper() for value in supplied]
        return sorted(set(value for value in values if value))
    signals = _read_json(project_root / "output/signals/latest_signals.json")
    signal_symbols = {
        str(row.get("ticker") or row.get("asset") or "").upper()
        for row in signals.get("signals", [])
        if isinstance(row, dict)
    }
    signal_symbols.discard("")
    return sorted(signal_symbols or set(config["symbols"]))


def _metadata_map(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / UNIVERSE_PATH
    if not path.is_file():
        return {}
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return {}
    required = {"symbol", "sector", "industry"}
    if not required.issubset(frame.columns):
        return {}
    active = (
        frame.loc[frame["active_listing"].astype(bool)]
        if "active_listing" in frame
        else frame
    )
    return {
        str(row["symbol"]).upper(): row.to_dict()
        for _, row in active.drop_duplicates("symbol", keep="last").iterrows()
    }


def _transmission_group(
    symbol: str,
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[str, str]:
    explicit = config["symbols"].get(symbol)
    if explicit in config["groups"]:
        return str(explicit), "EXPLICIT_SYMBOL_MAPPING"
    sector = _key(metadata.get("sector"))
    industry = _key(metadata.get("industry"))
    region = _key(metadata.get("region"))
    sleeve = _key(metadata.get("sleeve"))
    text = "_".join((sector, industry, region, sleeve))
    if "BOND" in sleeve or any(
        token in sector
        for token in ("TREASURY", "AGGREGATE_BOND", "CREDIT")
    ):
        if "SHORT" in sector or "SHORT" in industry:
            return "short_duration_bond", "UNIVERSE_CLASSIFICATION_MAPPING"
        return "long_duration_bond", "UNIVERSE_CLASSIFICATION_MAPPING"
    rules = (
        (("GOLD",), "gold"),
        (("SILVER",), "silver"),
        (("OIL", "GAS", "ENERGY"), "energy"),
        (("AGRICULT", "FARM"), "agriculture"),
        (("BROAD_COMMODITY",), "broad_commodity"),
        (("COPPER", "STEEL", "METAL", "MINING", "MATERIAL"), "materials_equity"),
        (("SEMICONDUCTOR", "SOFTWARE", "TECHNOLOGY"), "technology_equity"),
        (("BANK", "INSURANCE", "FINANCIAL", "CAPITAL_MARKET", "CREDIT_SERVICE", "ASSET_MANAGEMENT"), "financial_equity"),
        (("BIOTECH", "MEDICAL", "HEALTH", "DRUG", "PHARMA"), "healthcare_equity"),
        (("UTILIT",), "utility_equity"),
        (("REIT", "REAL_ESTATE"), "real_estate_equity"),
        (("CONSUMER_CYCLICAL", "CONSUMER_DISCRETIONARY", "AUTO", "RETAIL", "RESTAURANT", "LEISURE"), "consumer_cyclical_equity"),
        (("CONSUMER_DEFENSIVE", "PACKAGED_FOOD", "HOUSEHOLD"), "defensive_equity"),
        (("COMMUNICATION", "TELECOM", "ADVERTISING", "ENTERTAINMENT"), "communication_equity"),
        (("INDUSTRIAL", "AEROSPACE", "MACHINERY", "TRANSPORT", "CONSTRUCTION", "SHIPPING"), "industrial_equity"),
        (("EMERGING", "CHINA", "INDIA"), "emerging_equity"),
    )
    for tokens, group_id in rules:
        if any(token in text for token in tokens) and group_id in config["groups"]:
            return group_id, "UNIVERSE_CLASSIFICATION_MAPPING"
    return "broad_equity", "BROAD_EQUITY_DEFAULT"


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _key(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("&", "AND")


def _classification(score: float) -> str:
    if score >= 0.25:
        return "SUPPORTIVE"
    if score <= -0.25:
        return "ADVERSE"
    return "NEUTRAL_OR_MIXED"


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "asset_context_transmission_v1":
        raise ValueError("ASSET_TRANSMISSION_SCHEMA_INVALID")
    if not isinstance(config.get("groups"), dict) or not isinstance(
        config.get("symbols"), dict
    ):
        raise ValueError("ASSET_TRANSMISSION_MAPPING_INVALID")
    if "broad_equity" not in config["groups"]:
        raise ValueError("ASSET_TRANSMISSION_DEFAULT_GROUP_MISSING")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = ["build_asset_context"]
