from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.screener.models import ShariahSnapshot


THEME_CONFIG_PATH = Path("config/themes/frontier_technology_energy_v1.json")
SCREENER_CONFIG_PATH = Path("config/screener/daily_screener_v1.json")
ENTRY_SHORTLIST_PATH = Path("output/market_context/entry-shortlist.json")
REVIEW_EVIDENCE_PATH = Path(
    "data/research/themes/private/shariah-review-evidence.json"
)
OUTPUT_PATH = Path("output/analysis/themes/shariah-coverage.json")


def collect_theme_shariah_coverage(
    project_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Apply the existing expiring attestation gate to configured themes."""
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    theme_config = _read_json(project_root / THEME_CONFIG_PATH)
    instruments = _instruments(theme_config)
    if not instruments:
        return _publish(project_root, _blocked(observed_at, "THEME_CONFIG_UNAVAILABLE"))

    screener_config = _read_json(project_root / SCREENER_CONFIG_PATH)
    attestation_relative = screener_config.get("shariah_attestations_path")
    if not attestation_relative:
        return _publish(
            project_root,
            _blocked(observed_at, "SHARIAH_ATTESTATION_CONFIG_UNAVAILABLE"),
        )
    attestation_path = project_root / str(attestation_relative)
    attestation_payload = _read_json(attestation_path)
    attestations = {
        str(item.get("symbol") or "").upper(): item
        for item in attestation_payload.get("attestations", [])
        if item.get("symbol")
    }
    forward = _forward_priority(project_root)
    review_evidence = _review_evidence(project_root, observed_at)

    rows = []
    for instrument in instruments:
        symbol = instrument["symbol"]
        asset_class = str(
            instrument.get("ibkr_contract", {}).get("asset_class") or "stock"
        ).upper()
        result = _evaluate_attestation(
            symbol,
            asset_class,
            attestations.get(symbol),
            observed_at,
        )
        priority = forward.get(symbol, {})
        external_review = review_evidence.get(symbol, {})
        rows.append(
            {
                "symbol": symbol,
                "theme": instrument["theme"],
                "asset_class": asset_class,
                **result,
                "review_priority": (
                    "P1_CURRENT_FORWARD_SETUP"
                    if priority.get("hard_veto_pass")
                    else "P2_CURRENT_OBSERVATION"
                    if priority.get("observed")
                    else "P3_THEME_COVERAGE"
                ),
                "current_forward_state": priority.get("state"),
                "external_review": external_review,
            }
        )

    eligible = [row for row in rows if row["currently_eligible"]]
    review = [row for row in rows if not row["currently_eligible"]]
    review.sort(
        key=lambda row: (
            row["review_priority"],
            row["theme"],
            row["symbol"],
        )
    )
    status_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    report = {
        "schema": "frontier_theme_current_shariah_coverage_v1",
        "status": "GO" if len(eligible) == len(rows) else "GO_WITH_REVIEW_REQUIRED",
        "generated_at": observed_at.isoformat(),
        "methodology": (
            "EXISTING_MANUALLY_REVIEWED_EXPIRING_ATTESTATIONS_ONLY; "
            "NO_AUTOMATED_COMPLIANCE_INFERENCE"
        ),
        "instrument_count": len(rows),
        "currently_eligible_count": len(eligible),
        "review_required_count": len(review),
        "coverage_ratio": _ratio(len(eligible), len(rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "external_review_conflict_count": sum(
            row["external_review"].get("status") == "DUAL_SOURCE_CONFLICT"
            for row in rows
        ),
        "instruments": rows,
        "review_queue": [
            {
                "symbol": row["symbol"],
                "theme": row["theme"],
                "asset_class": row["asset_class"],
                "status": row["status"],
                "review_priority": row["review_priority"],
                "current_forward_state": row["current_forward_state"],
                "external_review_status": row["external_review"].get(
                    "status", "NOT_REVIEWED"
                ),
            }
            for row in review
        ],
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "provider_calls": 0,
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    return _publish(project_root, report)


def _evaluate_attestation(
    symbol: str,
    asset_class: str,
    attestation: dict[str, Any] | None,
    observed_at: datetime,
) -> dict[str, Any]:
    if not attestation:
        return {
            "status": (
                "SHARIAH_PRODUCT_ATTESTATION_REQUIRED"
                if asset_class != "STOCK"
                else "SHARIAH_ATTESTATION_REQUIRED"
            ),
            "currently_eligible": False,
            "screened_at": None,
            "expires_at": None,
            "methodology": None,
            "source": None,
            "evidence_reference_count": 0,
        }
    screened_at = _as_utc(attestation.get("screened_at"))
    expires_at = _as_utc(attestation.get("expires_at"))
    snapshot = ShariahSnapshot(
        status=str(attestation.get("status") or "SHARIAH_DATA_INCOMPLETE"),
        screened_at=screened_at,
        expires_at=expires_at,
        methodology=_text(attestation.get("methodology")),
        source=_text(attestation.get("source")),
    )
    eligible = snapshot.eligible_at(observed_at)
    status = snapshot.status
    if screened_at is None or expires_at is None:
        status = "SHARIAH_ATTESTATION_INVALID_DATES"
    elif screened_at > observed_at:
        status = "SHARIAH_ATTESTATION_NOT_YET_EFFECTIVE"
    elif expires_at < observed_at:
        status = "SHARIAH_ATTESTATION_EXPIRED"
    elif not eligible and status in {"SHARIAH_COMPLIANT", "SHARIAH_ELIGIBLE_PIT"}:
        status = "SHARIAH_ATTESTATION_NOT_CURRENTLY_ELIGIBLE"
    return {
        "status": status,
        "currently_eligible": eligible,
        "screened_at": screened_at.isoformat() if screened_at else None,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "methodology": snapshot.methodology,
        "source": snapshot.source,
        "evidence_reference_count": len(attestation.get("evidence") or []),
    }


def _forward_priority(project_root: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(project_root / ENTRY_SHORTLIST_PATH)
    output: dict[str, dict[str, Any]] = {}
    for row in payload.get("observations", []):
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        contract = row.get("decision_contract") or {}
        candidate = {
            "observed": True,
            "hard_veto_pass": bool(contract.get("hard_veto_pass")),
            "state": row.get("state"),
        }
        current = output.get(symbol)
        if current is None or candidate["hard_veto_pass"]:
            output[symbol] = candidate
    return output


def _review_evidence(
    project_root: Path,
    observed_at: datetime,
) -> dict[str, dict[str, Any]]:
    payload = _read_json(project_root / REVIEW_EVIDENCE_PATH)
    valid_until = _as_utc(payload.get("valid_until"))
    stale = valid_until is None or valid_until < observed_at
    output = {}
    for row in payload.get("rows", []):
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        sources = [
            {
                "provider": source.get("provider"),
                "as_of": source.get("as_of"),
                "status": source.get("status"),
                "url": source.get("url"),
            }
            for source in row.get("sources", [])
            if isinstance(source, dict)
        ]
        output[symbol] = {
            "status": (
                "STALE_REVIEW_EVIDENCE"
                if stale
                else str(row.get("review_outcome") or "REVIEW_INCOMPLETE")
            ),
            "observed_at": payload.get("observed_at"),
            "valid_until": payload.get("valid_until"),
            "source_count": len(sources),
            "sources": sources,
            "attestation_effect": "NONE",
        }
    return output


def _instruments(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for theme, definition in (config.get("themes") or {}).items():
        for item in definition.get("instruments", []):
            symbol = str(item.get("symbol") or "").upper()
            if symbol:
                rows.append({**item, "symbol": symbol, "theme": str(theme)})
    return rows


def _blocked(observed_at: datetime, reason: str) -> dict[str, Any]:
    return {
        "schema": "frontier_theme_current_shariah_coverage_v1",
        "status": "BLOCKED",
        "generated_at": observed_at.isoformat(),
        "blockers": [reason],
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "provider_calls": 0,
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _publish(project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    output = project_root / OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
